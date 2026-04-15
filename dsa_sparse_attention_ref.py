import torch
import flashinfer.decode

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOPK = 2048


def _get_workspace(device):
    key = str(device)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.zeros(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buf
    return buf


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens = q_nope.shape[0]
    num_pages, page_size, _ = ckv_cache.shape
    device = q_nope.device

    if isinstance(sm_scale, torch.Tensor):
        bmm1_scale = float(sm_scale.item())
    else:
        bmm1_scale = float(sm_scale)

    query = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)  # [T, 1, H, ckv+kpe]
    kv_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)    # [num_pages, page_size, ckv+kpe]
    block_tables = sparse_indices.unsqueeze(1)              # [T, 1, topk]

    # seq_lens = number of valid (non -1) entries per token
    # The kernel only reads the first seq_lens entries from block_tables;
    # valid entries are already contiguous at the front.
    seq_lens = (sparse_indices != -1).sum(dim=1).to(torch.int32)
    max_seq_len = int(seq_lens.max().item())
    workspace = _get_workspace(device)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=TOPK,
        bmm1_scale=bmm1_scale,
    )
    output = output.squeeze(1)  # [T, H, ckv]

    return (output,)
