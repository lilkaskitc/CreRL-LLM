#!/usr/bin/env python3
"""
Evaluate all merged checkpoints from a training run.

This script:
1. Finds all merged checkpoints in a directory
2. Evaluates each checkpoint using lighteval
3. Logs all metrics to a single wandb run for plotting accuracy vs training steps

Usage:
    python eval_all_merged_checkpoints.py \
        --merged_dir /path/to/merged/checkpoints \
        --wandb_project lighteval \
        --wandb_run_name eval_my_model \
        --eval_every 10
"""

import argparse
import subprocess
import sys
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all merged checkpoints and log accuracy across steps to wandb"
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        required=True,
        help="Path to directory containing merged checkpoints"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lighteval",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name (default: eval_<merged_dir_name>)"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default="272f1b038ff8ae26ef5980d3a79e786f9631ee29",
        help="WandB API key"
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Evaluate every N steps (e.g., every 10 steps)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="aime25m|0,aime24m|0,math_500m|0,minervamath|0,gsm8km|0,amc23|0,olympiadbench|0",
        help="Comma-separated list of tasks to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_eval",
        help="Base output directory for evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1536,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoints that have already been evaluated"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually running"
    )
    return parser.parse_args()


def find_checkpoints(merged_dir: str) -> List[Tuple[int, str]]:
    """
    Find all merged checkpoint directories and extract step numbers.

    Returns:
        List of (step, checkpoint_path) tuples, sorted by step
    """
    merged_path = Path(merged_dir)
    if not merged_path.exists():
        raise ValueError(f"Merged directory does not exist: {merged_dir}")

    checkpoints = []
    for checkpoint_dir in merged_path.glob("checkpoint-*"):
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
    Filter checkpoints to evaluate only every N steps.
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


def check_if_evaluated(output_dir: str, step: int) -> bool:
    """
    Check if a checkpoint has already been evaluated by looking for results.
    """
    eval_dir = Path(output_dir) / f"eval_step_{step}"
    # Look for any results_*.json file in the results subdirectory
    results_dir = eval_dir / "results"
    if results_dir.exists():
        # Recursively search for results_*.json files
        for json_file in results_dir.rglob("results_*.json"):
            return True
    return False


def run_lighteval(
    model_path: str,
    tasks: str,
    output_dir: str,
    batch_size: int = 16,
    max_length: int = 1536,
    max_new_tokens: int = 1024,
    dry_run: bool = False
) -> Tuple[str, str, int]:
    """
    Run lighteval evaluation and return stdout, stderr, returncode.
    Output is streamed in real-time to show progress bars.
    """
    cmd = [
        "accelerate", "launch",
        "--num_processes=1",
        "--mixed_precision=bf16",
        "-m", "lighteval", "accelerate",
        f"model_name={model_path},model_parallel=true,batch_size={batch_size},max_length={max_length},generation_parameters={{max_new_tokens:{max_new_tokens}}},cache_dir=None",
        tasks,
        "--custom-tasks", "./trl/eval.py",
        "--output-dir", output_dir,
        "--save-details",
    ]

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return "", "", 0

    print(f"Running: {' '.join(cmd)}")
    sys.stdout.flush()

    env = {
        **os.environ,
        "TORCH_COMPILE_DISABLE": "1",
        "ACCELERATE_USE_DEEPSPEED": "false",
        "ACCELERATE_NUM_PROCESSES": "1",
        "TRANSFORMERS_DISABLE_SDPA": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    # Use Popen for real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout for proper progress bar display
        text=True,
        env=env,
        bufsize=1,  # Line buffered
    )

    # Stream output in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print immediately
        sys.stdout.flush()
        output_lines.append(line)

    process.wait()

    output = ''.join(output_lines)
    return output, output, process.returncode  # Return same for both since we merged streams


def parse_results(results_path: str) -> Dict[str, float]:
    """
    Parse lighteval results JSON to extract accuracy metrics.
    
    The lighteval results are saved as results_*.json in a nested directory structure:
    eval_step_X/results/<full_path_to_checkpoint>/results_*.json
    """
    metrics = {}
    eval_dir = Path(results_path)
    
    # Look for results_*.json files in the results subdirectory
    results_dir = eval_dir / "results"
    if results_dir.exists():
        # Recursively search for results_*.json files
        json_files = list(results_dir.rglob("results_*.json"))
        if json_files:
            # Use the most recently modified file
            json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            results_file = json_files[0]
        else:
            return metrics
    else:
        # Fallback: look for any JSON in the results_path
        results_file = eval_dir / "results.json"
        if not results_file.exists():
            json_files = list(Path(results_path).glob("*.json"))
            if json_files:
                results_file = json_files[0]
            else:
                return metrics

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Navigate results structure - lighteval uses 'results' key
        task_results = results.get('results', {})

        for task_name, task_data in task_results.items():
            if isinstance(task_data, dict):
                # Clean task name: remove |0 suffix and replace with underscore
                clean_task = task_name.replace('|', '_').replace(':', '_')
                
                for key, value in task_data.items():
                    # Extract accuracy metrics (acc@1, extractive_match, etc.)
                    if isinstance(value, (int, float)) and not key.endswith('_stderr'):
                        metrics[f'{clean_task}_{key}'] = float(value)
                        
    except Exception as e:
        print(f"Warning: Could not parse results: {e}")
        import traceback
        traceback.print_exc()

    return metrics


def initialize_wandb(args, merged_dir: str):
    """
    Initialize wandb for logging.
    """
    try:
        import wandb

        run_name = args.wandb_run_name or f"eval_{Path(merged_dir).name}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "merged_dir": merged_dir,
                "eval_every": args.eval_every,
                "tasks": args.tasks,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "max_new_tokens": args.max_new_tokens,
            }
        )

        print(f"Initialized wandb run: {run_name}")
        return wandb

    except ImportError:
        print("Warning: wandb not installed. Will save results locally only.")
        return None


def log_metrics_to_wandb(wandb, metrics: Dict[str, float], step: int):
    """
    Log metrics to wandb with step information.
    """
    if wandb is None:
        return

    # Create log dict with step as x-axis
    log_dict = {"eval/step": step}

    # Add all accuracy metrics
    for key, value in metrics.items():
        if 'acc' in key.lower():
            log_dict[f"eval/accuracy/{key}"] = value

    # Also compute average accuracy if multiple tasks
    acc_values = [v for k, v in metrics.items() if 'acc' in k.lower()]
    if acc_values:
        log_dict["eval/avg_accuracy"] = sum(acc_values) / len(acc_values)

    wandb.log(log_dict)
    print(f"Logged to wandb: step={step}, metrics={list(metrics.keys())}")


def save_results_locally(results: List[Dict], output_path: str):
    """
    Save evaluation results to a JSON file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {output_path}")


def main():
    args = parse_args()

    # Set environment variables
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_START_METHOD"] = "thread"

    # Find checkpoints
    print(f"Searching for merged checkpoints in: {args.merged_dir}")
    all_checkpoints = find_checkpoints(args.merged_dir)
    print(f"Found {len(all_checkpoints)} merged checkpoints")

    if not all_checkpoints:
        print("No merged checkpoints found!")
        sys.exit(1)

    # Filter to evaluate every N steps
    checkpoints = filter_checkpoints(all_checkpoints, args.eval_every)
    print(f"Will evaluate {len(checkpoints)} checkpoints (every {args.eval_every} steps)")

    for step, path in checkpoints:
        print(f"  - Step {step}: {path}")

    # Initialize wandb
    wandb = initialize_wandb(args, args.merged_dir)

    # Store all results
    all_results = []

    # Evaluate each checkpoint
    for step, checkpoint_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: step={step}")
        print(f"Path: {checkpoint_path}")
        print(f"{'='*60}")

        # Check if already evaluated
        eval_output_dir = os.path.join(args.output_dir, f"eval_step_{step}")
        if args.skip_existing and check_if_evaluated(eval_output_dir, step):
            print(f"Skipping step {step} (already evaluated)")
            metrics = parse_results(eval_output_dir)
            if metrics:
                all_results.append({"step": step, "checkpoint": checkpoint_path, "metrics": metrics})
                log_metrics_to_wandb(wandb, metrics, step)
            continue

        # Run evaluation
        stdout, stderr, returncode = run_lighteval(
            model_path=checkpoint_path,
            tasks=args.tasks,
            output_dir=eval_output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            dry_run=args.dry_run
        )

        if returncode != 0 and not args.dry_run:
            print(f"Evaluation failed for step {step}: return code {returncode}")
            print(f"STDERR (full):")
            print(stderr)
            print(f"STDOUT (last 1000 chars):")
            print(stdout[-1000:] if len(stdout) > 1000 else stdout)
            continue

        # Parse results
        metrics = parse_results(eval_output_dir)

        if not metrics:
            print(f"Warning: Could not extract metrics for step {step}")
            print(f"STDOUT preview: {stdout[:500]}")
            continue

        print(f"Metrics for step {step}: {metrics}")

        # Store results
        all_results.append({
            "step": step,
            "checkpoint": checkpoint_path,
            "metrics": metrics
        })

        # Log to wandb
        log_metrics_to_wandb(wandb, metrics, step)

    # Save results locally
    results_file = os.path.join(args.output_dir, "eval_results_all.json")
    save_results_locally(all_results, results_file)

    # Finish wandb
    if wandb:
        wandb.finish()
        print(f"\nWandB run finished. View results at: https://wandb.ai/")

    print(f"\nEvaluation complete! Results saved to: {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"Step {result['step']}: {result['metrics']}")


if __name__ == "__main__":
    main()
