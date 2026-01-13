#!/usr/bin/env python

"""
Train SmolVLA policy on LeRobot-format UR5 grasping dataset.

This script uses the native LeRobot dataset format with proper preprocessors
for fast training. Based on the working train.py script approach.

Usage:
    python train_smolvla_lerobot_v2.py --repo-id local/ur5_smolvla_grasp
"""

import os
import sys

# Suppress warnings and set environment variables BEFORE any imports
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP duplicate library error on macOS

import argparse
from pathlib import Path

import logging
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("lerobot").setLevel(logging.WARNING)

import torch
import wandb

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA on LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="local/ur5_smolvla_grasp", help="Dataset repo ID")
    parser.add_argument("--root", type=str, default="../../datasets/lerobot", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/smolvla_lerobot", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--training-steps", type=int, default=5000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action chunk size")
    parser.add_argument("--checkpoint-freq", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--log-freq", type=int, default=10, help="Log frequency")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset to N samples for testing")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming from latest checkpoint")
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find latest checkpoint if resuming
    starting_step = 0
    pretrained_path = "lerobot/smolvla_base"
    optimizer_state_path = None
    scheduler_state_path = None
    
    if not args.no_resume:
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.name.split('-')[1]))
            starting_step = int(latest.name.split('-')[1])
            pretrained_path = str(latest)
            optimizer_state_path = latest / "optimizer.pt"
            scheduler_state_path = latest / "scheduler.pt"
            print(f"🔄 Resuming from step {starting_step}: {pretrained_path}")
        else:
            print("ℹ️  No checkpoints found, starting fresh")
    else:
        print("Starting fresh training (--no-resume flag set)")
    
    # Load dataset metadata FIRST (this is the key difference!)
    print(f"\nLoading dataset metadata: {args.repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    
    # Convert features to policy format
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print("\nInput features:")
    for key, feature in input_features.items():
        print(f"  {key}: {feature.shape}")
    print("\nOutput features:")
    for key, feature in output_features.items():
        print(f"  {key}: {feature.shape}")
    
    # Load SmolVLA policy
    print(f"\nLoading SmolVLA policy from: {pretrained_path}")
    policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    print("Policy loaded successfully!")
    
    policy.train()
    policy.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nPolicy: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Create preprocessor/postprocessor (this handles all the data conversion!)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, 
        dataset_stats=dataset_metadata.stats
    )
    
    # Configure delta_timestamps for proper temporal alignment
    delta_timestamps = {}
    for key in input_features.keys():
        if key.startswith("observation.image"):
            delta_timestamps[key] = [i / dataset_metadata.fps for i in range(-policy.config.n_obs_steps + 1, 1)]
        elif key.startswith("observation."):
            delta_timestamps[key] = [i / dataset_metadata.fps for i in range(-policy.config.n_obs_steps + 1, 1)]
    
    delta_timestamps["action"] = [i / dataset_metadata.fps for i in range(policy.config.chunk_size)]
    
    # Load dataset WITH delta_timestamps (this is crucial!)
    print("\nLoading dataset...")
    dataset = LeRobotDataset(
        args.repo_id, 
        delta_timestamps=delta_timestamps, 
        root=args.root
    )
    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
    
    # Optionally limit dataset size for testing
    if args.max_samples is not None:
        max_samples = min(args.max_samples, len(dataset))
        print(f"⚠️  Limiting to first {max_samples} samples for testing")
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0 if device.type == "mps" else 4,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    warmup_steps = 500 if starting_step == 0 else 0
    
    def lr_lambda(current_step):
        actual_step = starting_step + current_step
        if warmup_steps > 0 and actual_step < warmup_steps:
            return float(actual_step) / float(max(1, warmup_steps))
        return max(0.1, 1.0 - (actual_step - warmup_steps) / (args.training_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Load optimizer and scheduler states if resuming
    if optimizer_state_path and optimizer_state_path.exists():
        print(f"Loading optimizer state from {optimizer_state_path}")
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
    
    if scheduler_state_path and scheduler_state_path.exists():
        print(f"Loading scheduler state from {scheduler_state_path}")
        scheduler.load_state_dict(torch.load(scheduler_state_path, map_location=device))
    
    # Print training config
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Training steps: {args.training_steps}")
    print(f"Starting step: {starting_step}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Chunk size: {policy.config.chunk_size}")
    print(f"Checkpoint frequency: every {args.checkpoint_freq} steps")
    print("="*60 + "\n")
    
    # Initialize Weights & Biases
    wandb.init(
        project="smolvla-ur5",
        name=f"run-{output_dir.name}",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "training_steps": args.training_steps,
            "chunk_size": policy.config.chunk_size,
            "dataset": args.repo_id,
            "pretrained_from": pretrained_path,
            "device": str(device),
        },
        resume="allow",
    )
    print("📊 Weights & Biases initialized!\n")
    
    # Training loop
    print("Starting training...\n")
    step = starting_step
    done = False
    running_loss = 0.0
    
    while not done:
        for batch in dataloader:
            # Use the preprocessor (handles all image/text processing efficiently!)
            batch = preprocessor(batch)
            
            # Rename camera keys to match what the pretrained policy expects
            # Dataset has: observation.images.top, observation.images.wrist
            # Policy expects: observation.images.camera1, observation.images.camera2
            if "observation.images.top" in batch:
                batch["observation.images.camera1"] = batch.pop("observation.images.top")
            if "observation.images.wrist" in batch:
                batch["observation.images.camera2"] = batch.pop("observation.images.wrist")
            if "observation.images.top_is_pad" in batch:
                batch["observation.images.camera1_is_pad"] = batch.pop("observation.images.top_is_pad")
            if "observation.images.wrist_is_pad" in batch:
                batch["observation.images.camera2_is_pad"] = batch.pop("observation.images.wrist_is_pad")
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            running_loss += loss.item()
            
            # Log progress
            if step % args.log_freq == 0:
                avg_loss = running_loss / args.log_freq if step > 0 else loss.item()
                current_lr = scheduler.get_last_lr()[0]
                
                # Log to wandb
                wandb.log({
                    "loss": avg_loss,
                    "learning_rate": current_lr,
                    "step": step,
                })
                
                print(f"Step {step:5d}/{args.training_steps} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
                running_loss = 0.0
            
            step += 1
            
            # Save checkpoint
            if step % args.checkpoint_freq == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n💾 Saving checkpoint at step {step}...")
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                
                # Save optimizer and scheduler states
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
                
                print(f"✅ Checkpoint saved to {checkpoint_dir}\n")
            
            # Clear MPS cache periodically
            if device.type == "mps" and step % 100 == 0:
                torch.mps.empty_cache()
            
            if step >= args.training_steps:
                done = True
                break
    
    # Save final model
    print(f"\n{'='*60}")
    print(f"Training completed! Final loss: {loss.item():.6f}")
    print(f"{'='*60}\n")
    
    final_path = output_dir / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_path)
    preprocessor.save_pretrained(final_path)
    postprocessor.save_pretrained(final_path)
    
    # Finish wandb
    wandb.log({"final_loss": loss.item(), "final_step": step})
    wandb.finish()
    
    print(f"✅ Model saved to: {final_path.absolute()}")
    print("📊 Training metrics saved to Weights & Biases!")


if __name__ == "__main__":
    main()
