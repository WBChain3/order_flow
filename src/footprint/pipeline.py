from __future__ import annotations

import argparse
import sys


def run_pipeline(
    tick_path: str,
    output_dir: str,
    tick_size: float | None = None,
) -> None:
    print(f"Pipeline not yet implemented.")
    print(f"  Input:  {tick_path}")
    print(f"  Output: {output_dir}")
    if tick_size:
        print(f"  Tick size: {tick_size}")
    print("Stage 5 will wire this to the full end-to-end pipeline.")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build footprint tensors from tick data parquet files."
    )
    parser.add_argument("--input", required=True, help="Path to input parquet tick file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--tick-size", type=float, default=None, help="Instrument tick size")
    args = parser.parse_args()
    run_pipeline(
        tick_path=args.input,
        output_dir=args.output,
        tick_size=args.tick_size,
    )


if __name__ == "__main__":
    main()
