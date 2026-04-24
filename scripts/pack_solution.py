"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


TRACK_DIRS = ["dsa_indexer", "dsa_attention", "moe"]


def load_config(track_dir: Path) -> dict:
    """Load configuration from <track_dir>/config.toml."""
    config_path = track_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(track: str, output_path: Path = None) -> Path:
    """Pack the solution for a given track into a Solution JSON.

    `track` must match one of the immediate subdirectories that contain a
    config.toml (e.g. "dsa_indexer", "dsa_attention", "moe") — this is the
    layout the official evaluation pipeline expects per FAQ.md.
    """
    track_dir = PROJECT_ROOT / track
    if not track_dir.is_dir():
        raise FileNotFoundError(
            f"Track directory not found: {track_dir}. "
            f"Valid tracks: {TRACK_DIRS}"
        )

    config = load_config(track_dir)

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory based on language (relative to the track dir).
    if language == "triton":
        source_dir = track_dir / "solution" / "triton"
    elif language == "cuda":
        source_dir = track_dir / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Respect optional destination_passing_style override from config.toml.
    # Default True (framework default) — set False in config for value-returning
    # style kernels (e.g. topk indexer definition uses return-tuple semantics).
    dps = build_config.get("destination_passing_style", True)

    # Create build spec
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / f"solution_{track}.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "track",
        choices=TRACK_DIRS,
        help=f"Which track to pack (one of: {TRACK_DIRS})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution JSON (default: ./solution_<track>.json)",
    )
    args = parser.parse_args()

    try:
        pack_solution(args.track, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
