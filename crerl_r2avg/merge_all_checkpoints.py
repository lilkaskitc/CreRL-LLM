#!/usr/bin/env python3
"""
Merge all LoRA checkpoints from a training run.

This script:
1. Finds all checkpoints in a training output directory
2. Merges each LoRA checkpoint with the base model
3. Saves merged checkpoints to a specified directory

Usage:
    python merge_all_checkpoints.py \
        --train_dir /path/to/training/output \
        --base_model Qwen/Qwen2.5-Math-7B \
        --merged_dir ./output_merged \
        --eval_every 10
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge all LoRA checkpoints with base model"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to training output directory containing LoRA checkpoints"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Base model name or path"
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default="./output_merged",
        help="Base output directory for merged checkpoints"
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Merge every N steps (e.g., every 10 steps)"
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default="checkpoint-*",
        help="Glob pattern for checkpoint directories"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoints that have already been merged"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually running"
    )
    return parser.parse_args()


def find_checkpoints(train_dir: str, pattern: str = "checkpoint-*") -> List[Tuple[int, str]]:
    """
    Find all checkpoint directories and extract step numbers.

    Returns:
        List of (step, checkpoint_path) tuples, sorted by step
    """
    train_path = Path(train_dir)
    if not train_path.exists():
        raise ValueError(f"Training directory does not exist: {train_dir}")

    checkpoints = []
    for checkpoint_dir in train_path.glob(pattern):
        if checkpoint_dir.is_dir():
            # Extract step number from directory name
            match = re.search(r'checkpoint-(\d+)', checkpoint_dir.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, str(checkpoint_dir)))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def filter_checkpoints(
    checkpoints: List[Tuple[int, str]],
    eval_every: int
) -> List[Tuple[int, str]]:
    """
    Filter checkpoints to merge only every N steps.
    """
    if eval_every <= 0:
        return checkpoints

    # Always include the first and last checkpoint
    if len(checkpoints) <= 1:
        return checkpoints

    filtered = [checkpoints[0]]  # Always include first

    for i, (step, path) in enumerate(checkpoints[1:-1], 1):
        if step % eval_every == 0:
            filtered.append((step, path))

    # Always include last if not already included
    if checkpoints[-1] not in filtered:
        filtered.append(checkpoints[-1])

    return filtered


def check_if_merged(merged_dir: str, step: int) -> bool:
    """
    Check if a checkpoint has already been merged.
    """
    merged_path = Path(merged_dir) / f"checkpoint-{step}"
    return merged_path.exists() and (merged_path / "config.json").exists()


def merge_checkpoint(
    checkpoint_path: str,
    base_model: str,
    merged_output_path: str,
    dry_run: bool = False
) -> bool:
    """
    Merge a LoRA checkpoint with the base model.

    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        print(f"[DRY RUN] Would merge {checkpoint_path} -> {merged_output_path}")
        return True

    print(f"  Base model: {base_model}")
    print(f"  Output: {merged_output_path}")

    try:
        # Load base model
        print(f"  Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="bfloat16",
            device_map="auto",
        )

        # Load LoRA adapters
        print(f"  Loading LoRA adapters from {checkpoint_path}...")
        model = PeftModel.from_pretrained(base, checkpoint_path)

        # Merge and unload
        print(f"  Merging LoRA weights...")
        model = model.merge_and_unload()

        # Save merged model
        print(f"  Saving merged model to {merged_output_path}...")
        os.makedirs(merged_output_path, exist_ok=True)
        model.save_pretrained(merged_output_path)

        # Save tokenizer
        print(f"  Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.save_pretrained(merged_output_path)

        print(f"  Successfully merged checkpoint!")
        return True

    except Exception as e:
        print(f"  Error merging checkpoint: {e}")
        return False


def main():
    args = parse_args()

    # Set environment variables for memory efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Find checkpoints
    print(f"Searching for checkpoints in: {args.train_dir}")
    all_checkpoints = find_checkpoints(args.train_dir, args.checkpoint_pattern)
    print(f"Found {len(all_checkpoints)} checkpoints")

    if not all_checkpoints:
        print("No checkpoints found!")
        sys.exit(1)

    # Filter to merge every N steps
    checkpoints = filter_checkpoints(all_checkpoints, args.eval_every)
    print(f"Will merge {len(checkpoints)} checkpoints (every {args.eval_every} steps)")

    for step, path in checkpoints:
        print(f"  - Step {step}: {path}")

    # Create merged output directory
    merged_base = Path(args.merged_dir) / Path(args.train_dir).name
    merged_base.mkdir(parents=True, exist_ok=True)
    print(f"\nMerged checkpoints will be saved to: {merged_base}")

    # Merge each checkpoint
    merged_count = 0
    skipped_count = 0
    failed_count = 0

    for step, checkpoint_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: step={step}")
        print(f"Path: {checkpoint_path}")
        print(f"{'='*60}")

        # Check if already merged
        merged_output_path = merged_base / f"checkpoint-{step}"
        if args.skip_existing and check_if_merged(str(merged_base), step):
            print(f"Skipping step {step} (already merged)")
            skipped_count += 1
            continue

        # Merge checkpoint
        success = merge_checkpoint(
            checkpoint_path=checkpoint_path,
            base_model=args.base_model,
            merged_output_path=str(merged_output_path),
            dry_run=args.dry_run
        )

        if success:
            merged_count += 1
        else:
            failed_count += 1
            print(f"Failed to merge checkpoint for step {step}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Merged: {merged_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"\nMerged checkpoints saved to: {merged_base}")


if __name__ == "__main__":
    main()
