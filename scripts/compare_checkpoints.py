#!/usr/bin/env python
"""
Compare performance of multiple checkpoints.

Usage:
    python compare_checkpoints.py --checkpoints 60000 70000 --num-episodes 10
"""

import argparse
import subprocess
import json
from pathlib import Path
import pandas as pd


def run_evaluation(checkpoint_dir, num_episodes=10, max_steps=500, seed=42):
    """Run evaluation and capture results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_dir}")
    print(f"{'='*60}\n")
    
    cmd = [
        "python", "eval_smolvla_lerobot.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--num-episodes", str(num_episodes),
        "--max-steps", str(max_steps),
        "--seed", str(seed),
        "--no-gui",  # Run without GUI for faster evaluation
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 min timeout per checkpoint
        )
        
        # Parse output for success rate
        output = result.stdout + result.stderr
        
        # Look for success rate in output
        success_rate = None
        avg_steps = None
        for line in output.split('\n'):
            if 'Success rate:' in line or 'success rate:' in line:
                try:
                    parts = line.split(':')[1].strip()
                    success_rate = float(parts.replace('%', '').split()[0])
                except:
                    pass
            if 'Average steps:' in line or 'avg steps:' in line:
                try:
                    avg_steps = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
        
        return {
            'checkpoint': checkpoint_dir.name,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'output': output,
        }
    
    except subprocess.TimeoutExpired:
        print(f"⚠️  Evaluation timeout for {checkpoint_dir}")
        return {
            'checkpoint': checkpoint_dir.name,
            'success_rate': None,
            'avg_steps': None,
            'output': 'TIMEOUT',
        }
    except Exception as e:
        print(f"❌ Error evaluating {checkpoint_dir}: {e}")
        return {
            'checkpoint': checkpoint_dir.name,
            'success_rate': None,
            'avg_steps': None,
            'output': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints")
    parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                        help="Checkpoint steps to compare (e.g., 60000 70000)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/smolvla_full",
                        help="Base checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes per checkpoint")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="checkpoint_comparison.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    base_dir = Path(args.checkpoint_dir)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Checkpoint Comparison Evaluation")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    print(f"Checkpoints to evaluate: {args.checkpoints}")
    print(f"Episodes per checkpoint: {args.num_episodes}")
    print(f"{'='*60}\n")
    
    # Evaluate each checkpoint
    for checkpoint_step in args.checkpoints:
        checkpoint_dir = base_dir / f"checkpoint-{checkpoint_step}"
        
        if not checkpoint_dir.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_dir}")
            continue
        
        result = run_evaluation(
            checkpoint_dir, 
            args.num_episodes, 
            args.max_steps,
            args.seed
        )
        results.append(result)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    if results:
        # Create pandas dataframe for nice display
        df = pd.DataFrame([{
            'Checkpoint': r['checkpoint'],
            'Success Rate (%)': r['success_rate'] if r['success_rate'] is not None else 'N/A',
            'Avg Steps': r['avg_steps'] if r['avg_steps'] is not None else 'N/A',
        } for r in results])
        
        print(df.to_string(index=False))
        print()
        
        # Find best checkpoint
        valid_results = [r for r in results if r['success_rate'] is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x['success_rate'])
            print(f"🏆 Best checkpoint: {best['checkpoint']} ({best['success_rate']:.1f}% success)")
        
        # Save results
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")
    else:
        print("❌ No results to display")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
