#!/usr/bin/env python3
"""
Parse existing evaluation results and generate a plot of accuracy vs steps.

This script:
1. Finds all eval_step_* directories
2. Parses the results_*.json files
3. Saves consolidated results to eval_results_all.json
4. Generates a plot of accuracy vs training steps
5. Logs to wandb if not already logged

Usage:
    python eval_all_plot_results.py \
        --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/crerl_r2avg/eval_log/aggcrerl_r2avg_grpo_lora4g_5496617 \
        --wandb_project lighteval \
        --wandb_run_name eval_merged_aggcrerl_r2avg_grpo_lora4g_5496617
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse eval results and plot accuracy vs steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory containing eval_step_* folders"
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
        help="WandB run name"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default="272f1b038ff8ae26ef5980d3a79e786f9631ee29",
        help="WandB API key"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the plot"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip logging to wandb"
    )
    return parser.parse_args()


def parse_results(results_path: str) -> Dict[str, float]:
    """
    Parse lighteval results JSON to extract accuracy metrics.
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
        print(f"Warning: Could not parse results from {results_file}: {e}")

    return metrics


def find_all_eval_results(output_dir: str) -> List[Tuple[int, str, Dict[str, float]]]:
    """
    Find all eval_step_* directories and parse their results.
    
    Returns:
        List of (step, eval_dir_path, metrics) tuples, sorted by step
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    results = []
    
    # Find all eval_step_* directories
    for eval_dir in sorted(output_path.glob("eval_step_*")):
        if eval_dir.is_dir():
            # Extract step number from directory name
            match = re.search(r'eval_step_(\d+)', eval_dir.name)
            if match:
                step = int(match.group(1))
                metrics = parse_results(str(eval_dir))
                
                if metrics:
                    results.append((step, str(eval_dir), metrics))
                    print(f"Found results for step {step}: {list(metrics.keys())[:5]}...")
                else:
                    print(f"Warning: No metrics found for step {step} at {eval_dir}")
    
    # Sort by step number
    results.sort(key=lambda x: x[0])
    return results


def save_results(results: List[Tuple[int, str, Dict]], output_path: str):
    """
    Save consolidated results to JSON.
    """
    all_results = []
    for step, eval_dir, metrics in results:
        all_results.append({
            "step": step,
            "eval_dir": eval_dir,
            "metrics": metrics
        })
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved consolidated results to: {output_path}")
    return all_results


def create_plot(results: List[Tuple[int, str, Dict]], output_dir: str):
    """
    Create a plot of accuracy vs training steps.
    """
    if not results:
        print("No results to plot!")
        return
    
    # Extract steps and accuracy metrics
    steps = [r[0] for r in results]
    
    # Find all accuracy keys (acc@1 metrics for different tasks)
    all_keys = set()
    for _, _, metrics in results:
        all_keys.update(metrics.keys())
    
    # Filter to accuracy-related metrics
    acc_keys = sorted([k for k in all_keys if 'acc' in k.lower() or 'extractive' in k.lower()])
    
    if not acc_keys:
        print("No accuracy metrics found to plot!")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10.colors
    color_idx = 0
    
    # Plot each accuracy metric
    for key in acc_keys:
        values = []
        valid_steps = []
        
        for step, _, metrics in results:
            if key in metrics:
                values.append(metrics[key])
                valid_steps.append(step)
        
        if values:
            # Use a clean label name
            label = key.replace('_acc@1', '').replace('_extractive_match', '')
            label = label.replace('_', ' ').title()
            
            ax.plot(valid_steps, values, marker='o', label=label, 
                   color=colors[color_idx % len(colors)], linewidth=2, markersize=8)
            color_idx += 1
    
    # Add average accuracy line
    avg_values = []
    for step, _, metrics in results:
        acc_vals = [v for k, v in metrics.items() if 'acc@1' in k.lower()]
        if acc_vals:
            avg_values.append(sum(acc_vals) / len(acc_vals))
        else:
            avg_values.append(None)
    
    if any(v is not None for v in avg_values):
        ax.plot(steps, avg_values, marker='s', label='Average Accuracy', 
               color='black', linewidth=2, markersize=10, linestyle='--')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Evaluation Accuracy vs Training Steps', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'eval_accuracy_vs_steps.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {plot_path}")


def log_to_wandb(results: List[Tuple[int, str, Dict]], args):
    """
    Log results to wandb.
    """
    if args.no_wandb:
        print("Skipping wandb logging (--no-wandb flag set)")
        return
    
    try:
        import wandb
        
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        
        run_name = args.wandb_run_name or f"eval_results_{Path(args.output_dir).name}"
        
        # Check if run already exists by trying to resume
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                resume="allow"
            )
            print(f"Connected to wandb run: {run_name}")
        except Exception as e:
            print(f"Could not connect to existing wandb run: {e}")
            wandb.init(
                project=args.wandb_project,
                name=run_name
            )
        
        # Log all metrics with step
        for step, _, metrics in results:
            log_dict = {"eval/step": step}
            
            # Add all accuracy metrics
            for key, value in metrics.items():
                if 'acc' in key.lower() or 'extractive' in key.lower():
                    log_dict[f"eval/accuracy/{key}"] = value
            
            # Add average accuracy
            acc_vals = [v for k, v in metrics.items() if 'acc@1' in k.lower()]
            if acc_vals:
                log_dict["eval/avg_accuracy"] = sum(acc_vals) / len(acc_vals)
            
            wandb.log(log_dict, step=step)
            print(f"Logged step {step} to wandb")
        
        # Log the plot
        plot_path = os.path.join(args.output_dir, 'eval_accuracy_vs_steps.png')
        if os.path.exists(plot_path):
            wandb.log({"eval/accuracy_plot": wandb.Image(plot_path)})
            print("Logged plot to wandb")
        
        wandb.finish()
        print("WandB logging complete!")
        
    except ImportError:
        print("Warning: wandb not installed. Skipping wandb logging.")
    except Exception as e:
        print(f"Warning: Could not log to wandb: {e}")


def main():
    args = parse_args()
    
    print(f"Searching for evaluation results in: {args.output_dir}")
    
    # Find all eval results
    results = find_all_eval_results(args.output_dir)
    
    if not results:
        print("No evaluation results found!")
        sys.exit(1)
    
    print(f"\nFound {len(results)} evaluation results")
    
    # Save consolidated results
    results_file = os.path.join(args.output_dir, "eval_results_all.json")
    all_results = save_results(results, results_file)
    
    # Create plot
    if not args.no_plot:
        print("\nCreating plot...")
        create_plot(results, args.output_dir)
    
    # Log to wandb
    print("\nLogging to wandb...")
    log_to_wandb(results, args)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for step, _, metrics in results:
        # Show key metrics
        acc_metrics = {k: v for k, v in metrics.items() if 'acc@1' in k.lower()}
        print(f"Step {step}: avg_acc={sum(acc_metrics.values())/len(acc_metrics):.4f} ({len(acc_metrics)} tasks)")
        for task, val in sorted(acc_metrics.items()):
            print(f"  - {task}: {val:.4f}")


if __name__ == "__main__":
    main()
