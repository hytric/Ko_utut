#!/usr/bin/env python3
# split_manifest.py

import argparse
import os
import json
import random
from pathlib import Path

def split_manifest(input_manifest, train_ratio, val_ratio, output_dir, seed=None):
    """
    Splits a manifest file into training and validation sets.

    Args:
        input_manifest (str): Path to the input manifest file.
        train_ratio (float): Proportion of data to be used for training.
        val_ratio (float): Proportion of data to be used for validation.
        output_dir (str): Directory where the split manifests will be saved.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    # Read all lines from the input manifest
    with open(input_manifest, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total entries in manifest: {total}")

    # Shuffle the lines
    random.shuffle(lines)

    # Calculate split indices
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split the lines
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output manifest paths
    train_manifest_path = Path(output_dir) / "train_manifest.json"
    val_manifest_path = Path(output_dir) / "val_manifest.json"

    # Write training manifest
    with open(train_manifest_path, 'w') as f:
        f.writelines(train_lines)
    print(f"Training manifest saved to: {train_manifest_path} ({len(train_lines)} entries)")

    # Write validation manifest
    with open(val_manifest_path, 'w') as f:
        f.writelines(val_lines)
    print(f"Validation manifest saved to: {val_manifest_path} ({len(val_lines)} entries)")

def main():
    parser = argparse.ArgumentParser(description="Split a manifest file into training and validation sets.")
    parser.add_argument('--input_manifest', type=str, required=True, help="Path to the input manifest file.")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Proportion of data for training (default: 0.8).")
    parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of data for validation (default: 0.2).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the split manifests.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility (optional).")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio
    if not 0.99 < total_ratio < 1.01:
        raise ValueError(f"The sum of train_ratio and val_ratio should be approximately 1.0. Got {total_ratio}")

    split_manifest(args.input_manifest, args.train_ratio, args.val_ratio, args.output_dir, args.seed)

if __name__ == "__main__":
    main()
