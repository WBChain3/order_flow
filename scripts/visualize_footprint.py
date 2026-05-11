#!/usr/bin/env python3
"""
Visualizes a footprint array as a heatmap.

Usage:
    python scripts/visualize_footprint.py --footprint path/to/footprint.npy
    python scripts/visualize_footprint.py --dataset path/to/dataset.npz --index 0
"""

import argparse
import sys

import numpy as np


def visualize_footprint(footprint: np.ndarray, title: str = "Footprint") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization. Install with: pip install matplotlib")
        sys.exit(1)

    channel_labels = ["Bid Volume", "Ask Volume", "Delta", "Total Volume"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    for c in range(4):
        ax = axes[c // 2, c % 2]
        im = ax.imshow(footprint[c], aspect="auto", cmap="viridis")
        ax.set_title(channel_labels[c])
        ax.set_xlabel("Time Bucket")
        ax.set_ylabel("Price Level")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize footprint arrays")
    parser.add_argument("--footprint", type=str, help="Path to .npy footprint file")
    parser.add_argument("--dataset", type=str, help="Path to .npz dataset file")
    parser.add_argument("--index", type=int, default=0, help="Sample index in dataset")
    args = parser.parse_args()

    if args.footprint:
        footprint = np.load(args.footprint)
        visualize_footprint(footprint, title=f"Footprint: {args.footprint}")

    if args.dataset:
        data = np.load(args.dataset)
        samples = data["samples"]
        idx = args.index
        if idx >= len(samples):
            print(f"Index {idx} out of range. Dataset has {len(samples)} samples.")
            sys.exit(1)
        sample = samples[idx]
        visualize_footprint(sample, title=f"Dataset Sample {idx}")


if __name__ == "__main__":
    main()
