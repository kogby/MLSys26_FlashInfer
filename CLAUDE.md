# FlashInfer AI Kernel Generation Contest @ MLSys 2026 - 專案筆記

## 專案目的

參加 FlashInfer AI Kernel Generation Contest @ MLSys 2026，為 NVIDIA Blackwell GPU 撰寫高效能 GPU kernel。
選擇賽道：**`sparse_attention`**（稀疏注意力機制）。

評測框架：[flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench)

## 目錄結構

```
MLSys26_FlashInfer/
├── config.toml                          # 賽道與團隊設定（目前選擇 sparse_attention + triton）
├── solution/
│   ├── triton/
│   │   ├── kernel_sparse_attention_baseline.py   # Sparse Attention baseline
│   │   └── kernel_topk_indexer_baseline.py       # TopK Indexer baseline
│   └── cuda/
│       ├── kernel.cu                    # CUDA kernel（空模板）
│       └── binding.py                   # CUDA Python 綁定（空模板）
├── dsa_sparse_attention_ref.py          # Sparse Attention 參考實作
├── dsa_topk_indexer_ref.py              # TopK Indexer 參考實作
├── scripts/
│   ├── pack_solution.py                 # 打包 solution.json
│   ├── run_local.py                     # 本地 GPU 測試
│   └── run_modal.py                     # 雲端 Modal (B200) 測試
└── images/                              # Logo 圖片
```

## 核心元件

### 1. Sparse Attention Kernel（主要任務）

- **輸入**:
  - `q_nope`: `[num_tokens, 16, 512]` bfloat16 — 不含位置編碼的 query
  - `q_pe`: `[num_tokens, 16, 64]` bfloat16 — 含位置編碼的 query
  - `ckv_cache`: `[num_pages, 64, 512]` bfloat16 — 分頁式壓縮 KV cache
  - `kpe_cache`: `[num_pages, 64, 64]` bfloat16 — 分頁式 KV 位置編碼 cache
  - `sparse_indices`: `[num_tokens, 2048]` int32 — 選中的 token 索引（-1 = 無效）
  - `sm_scale`: float — softmax 縮放因子
- **運算**: 對選中的 2048 個 token 做 attention
  - `logits = Q_nope · K_ckv^T + Q_pe · K_pe^T`
  - softmax 後加權求和
- **輸出**:
  - `output`: `[num_tokens, 16, 512]` bfloat16 — attention 輸出
  - `lse`: `[num_tokens, 16]` float32 — log-sum-exp (base 2)
- **關鍵維度**: 16 heads, head_dim=512 (ckv) + 64 (kpe), page_size=64

### 2. TopK Indexer Kernel（輔助任務）

- **輸入**:
  - `q_index_fp8`: `[batch_size, 64, 128]` fp8 — 查詢向量
  - `k_index_cache_fp8`: `[num_pages, 64, 1, 132]` int8 — FP8 KV cache（deep_gemm 格式，128 bytes data + 4 bytes scale）
  - `weights`: `[batch_size, 64]` f32 — 每個 head 的 learned weights
  - `seq_lens`: `[batch_size]` int32 — 序列長度
  - `block_table`: `[batch_size, max_num_pages]` int32 — page table
- **運算**:
  - 反量化 FP8 → float32
  - 計算 attention score: `scores = Q · K^T`
  - ReLU 激活
  - 加權求和跨 heads: `final = Σ(weights_h × ReLU(scores_h))`
  - 取 top-2048 個 token
- **輸出**:
  - `topk_indices`: `[batch_size, 2048]` int32 — 全域 token 索引

### 3. 兩者的關係（兩階段稀疏注意力）

```
Stage 1: TopK Indexer
  FP8 量化的 Q/K → 粗略 attention score → 選出最重要的 2048 個 token 索引

Stage 2: Sparse Attention
  用精確的 bfloat16 Q/KV → 對選中的 2048 個 token 做完整 attention 計算
```

這是類似 DeepSeek MLA + NSA 的設計：先用低精度快速篩選，再用高精度精確計算。

## DPS（Destination Passing Style）

FlashInfer-Bench 預設使用 DPS 風格：輸入和輸出都作為函數參數傳入，輸出是預先分配好的 tensor，kernel 直接寫入。
這樣可以避免量測到 tensor 分配的開銷，得到更準確的效能數據。

## 目前狀態

- 目前只有 **PyTorch baseline**（逐 token 的 for-loop 實作），效能極低
- 最終目標：用 **Triton**（或 CUDA）寫出高效的 GPU kernel 取代 baseline

## 開發流程

```bash
# 1. 實作 kernel
#    編輯 solution/triton/kernel.py

# 2. 打包
python scripts/pack_solution.py

# 3. 本地測試
python scripts/run_local.py

# 4. 雲端測試（B200 GPU）
modal run scripts/run_modal.py
```
