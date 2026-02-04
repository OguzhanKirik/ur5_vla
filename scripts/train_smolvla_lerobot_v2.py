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
import time

import logging
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("lerobot").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import wandb

# Import augmentation utilities
try:
    from augmentation_utils import get_augmentation_pipeline, apply_augmentations
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("⚠️  augmentation_utils.py not found - augmentation disabled")

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA on LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="local/ur5_smolvla_grasp", help="Dataset repo ID")
    parser.add_argument("--root", type=str, default="./datasets/lerobot", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/smolvla_full", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--training-steps", type=int, default=60000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--checkpoint-freq", type=int, default=5000, help="Checkpoint frequency")
    parser.add_argument("--log-freq", type=int, default=10, help="Log frequency")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset to N samples for testing")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming from latest checkpoint")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate dataset shapes and camera names, then exit")
    parser.add_argument("--pretrained-model", type=str, default="lerobot/smolvla_base",
                        help="Pretrained model to start from (default: lerobot/smolvla_base)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating (effective batch size = batch_size * this)")
    parser.add_argument("--use-augmentation", action="store_true",
                        help="Enable data augmentation (color jitter, crop, noise)")
    parser.add_argument("--augmentation-strength", type=str, default="conservative", 
                        choices=["minimal", "conservative", "aggressive"],
                        help="Augmentation strength: minimal (sim-to-sim), conservative (sim-to-real), or aggressive")
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Set memory management for MPS to avoid crashes
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If root is wrong relative to cwd, fall back to the script-local dataset.
    root_path = Path(args.root)
    if not (root_path / "meta" / "info.json").exists():
        alt_root = Path(__file__).resolve().parent / "datasets" / "lerobot"
        if (alt_root / "meta" / "info.json").exists():
            print(f"⚠️  Dataset root not found at {root_path}; using {alt_root}")
            args.root = str(alt_root)
        else:
            print(f"⚠️  Dataset root not found at {root_path}; expected meta/info.json")
    
    # Find latest checkpoint if resuming
    starting_step = 0
    pretrained_path = args.pretrained_model  # Use specified pretrained model
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
    
    # Rename camera keys for policy config (but keep original for dataset loading)
    policy_input_features = dict(input_features)
    if "observation.images.top" in policy_input_features:
        policy_input_features["observation.images.camera1"] = policy_input_features.pop("observation.images.top")
    if "observation.images.wrist" in policy_input_features:
        policy_input_features["observation.images.camera2"] = policy_input_features.pop("observation.images.wrist")
    
    # Load SmolVLA policy
    print(f"\nLoading SmolVLA policy from: {pretrained_path}")
    load_start = time.time()
    
    # Load pretrained model for ALL cases (fresh or resume).
    # CRITICAL: Using SmolVLAPolicy(config) from scratch leaves VLM with random weights
    # when load_vlm_weights=False (the default). from_pretrained loads real weights.
    print(f"Loading SmolVLA policy via from_pretrained: {pretrained_path}")
    policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    print(f"Policy loaded from: {pretrained_path}")

    # Override chunk_size and n_action_steps from args
    policy.config.chunk_size = args.chunk_size
    policy.config.n_action_steps = args.chunk_size  # match chunk_size

    load_time = time.time() - load_start
    print(f"Policy ready in {load_time:.2f}s")

    # Remove camera3 from pretrained config if present (our dataset only has 2 cameras)
    if "observation.images.camera3" in policy.config.input_features:
        del policy.config.input_features["observation.images.camera3"]
        print("Removed observation.images.camera3 from config (dataset has 2 cameras)")

    # UPDATE POLICY CONFIG TO ACCEPT 7D ACTIONS AND 8D STATE
    # Dataset has 7D actions (6 joints + 1 gripper) but pretrained model expects 6D
    # Dataset has 8D state (6 joints + 2 gripper values) but pretrained model expects 6D
    # We need to extend both to match our dataset
    dataset_action_dim = output_features['action'].shape[0]  # Should be 7
    dataset_state_dim = input_features['observation.state'].shape[0]  # Should be 8

    print(f"\nDataset action dim: {dataset_action_dim}, Policy action dim: {policy.config.action_feature.shape[0]}")
    print(f"Dataset state dim: {dataset_state_dim}, Policy state dim: {policy.config.input_features['observation.state'].shape[0]}")

    if dataset_action_dim != policy.config.action_feature.shape[0]:
        print(f"Action dimension mismatch - updating policy config...")
        new_action_feature = output_features['action']
        policy.config.output_features['action'] = new_action_feature
        policy.config.max_action_dim = max(policy.config.max_action_dim, dataset_action_dim)
        print(f"Updated action feature to shape {new_action_feature.shape}")

    if dataset_state_dim != policy.config.input_features['observation.state'].shape[0]:
        print(f"State dimension mismatch - updating policy config...")
        new_state_feature = input_features['observation.state']
        policy.config.input_features['observation.state'] = new_state_feature
        policy.config.max_state_dim = max(policy.config.max_state_dim, dataset_state_dim)
        print(f"Updated state feature to shape {new_state_feature.shape}")
    
    # Ensure model is in training mode
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
    
    # Create dataloader with optimized settings
    drop_last = len(dataset) >= args.batch_size
    # MPS has issues with multiprocessing - use 0 workers to avoid crashes
    # CUDA/CPU can use multiple workers for better performance
    if device.type == "mps":
        num_workers = 0
        print("⚠️  Using num_workers=0 on MPS to avoid multiprocessing crashes")
    else:
        num_workers = 4
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep workers alive
    )
    print(f"DataLoader: {num_workers} workers, batch_size={args.batch_size}")
    
    if args.validate_only:
        batch = next(iter(dataloader))
        errors = []
        
        # Raw dataset checks
        expected_cameras = ["observation.images.top", "observation.images.wrist"]
        for cam_key in expected_cameras:
            if cam_key not in batch:
                errors.append(f"Missing camera key: {cam_key}")
            else:
                img = batch[cam_key]
                if not torch.is_floating_point(img):
                    errors.append(f"{cam_key} is not float (dtype={img.dtype})")
                else:
                    img_min = img.min().item()
                    img_max = img.max().item()
                    if img_min < -1e-3 or img_max > 1.0 + 1e-3:
                        errors.append(f"{cam_key} out of range [0,1]: min={img_min:.3f}, max={img_max:.3f}")
                if img.shape[-3] != 3:
                    errors.append(f"{cam_key} channel dim != 3 (shape={tuple(img.shape)})")
        
        if "observation.state" not in batch:
            errors.append("Missing observation.state")
        elif batch["observation.state"].shape[-1] != dataset_state_dim:
            errors.append(
                f"observation.state dim mismatch: {batch['observation.state'].shape[-1]} vs {dataset_state_dim}"
            )
        
        if "action" not in batch:
            errors.append("Missing action")
        elif batch["action"].shape[-1] != dataset_action_dim:
            errors.append(f"action dim mismatch: {batch['action'].shape[-1]} vs {dataset_action_dim}")
        
        # Rename to policy camera keys (matches training)
        batch = preprocessor(batch)
        if "observation.images.top" in batch:
            batch["observation.images.camera1"] = batch.pop("observation.images.top")
        if "observation.images.wrist" in batch:
            batch["observation.images.camera2"] = batch.pop("observation.images.wrist")
        if "observation.images.top_is_pad" in batch:
            batch["observation.images.camera1_is_pad"] = batch.pop("observation.images.top_is_pad")
        if "observation.images.wrist_is_pad" in batch:
            batch["observation.images.camera2_is_pad"] = batch.pop("observation.images.wrist_is_pad")
        
        if "observation.images.camera1" not in batch:
            errors.append("Missing observation.images.camera1 after rename")
        if "observation.images.camera2" not in batch:
            errors.append("Missing observation.images.camera2 after rename")
        
        print("\nValidation summary")
        print(f"  action dim: {dataset_action_dim}")
        print(f"  state dim: {dataset_state_dim}")
        if "observation.images.camera1" in batch:
            img = batch["observation.images.camera1"]
            print(f"  camera1: dtype={img.dtype}, shape={tuple(img.shape)}")
        if "observation.images.camera2" in batch:
            img = batch["observation.images.camera2"]
            print(f"  camera2: dtype={img.dtype}, shape={tuple(img.shape)}")
        
        if errors:
            print("\nValidation failed:")
            for err in errors:
                print(f"  - {err}")
            raise SystemExit(1)
        
        print("\nValidation OK")
        return
    
    # If LR is explicitly overridden, reset LR schedule offset to avoid immediate decay.
    lr_overridden = "--lr" in sys.argv
    if lr_overridden:
        print("⚠️  --lr provided; resetting LR schedule offset to 0")

    # Setup optimizer — match pretrained SmolVLA config:
    # weight_decay=1e-10, betas=(0.9, 0.95), grad_clip=10.0
    # Separate into decay / no-decay groups (exclude bias & norm layers)
    decay_params = []
    no_decay_params = []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-10},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    
    # Learning rate scheduler with warmup
    # If resuming past warmup, skip warmup entirely.
    base_warmup_steps = 500
    warmup_steps = 0 if lr_overridden or starting_step >= base_warmup_steps else base_warmup_steps

    lr_schedule_offset = 0 if lr_overridden else starting_step

    def lr_lambda(current_step):
        actual_step = lr_schedule_offset + current_step
        if warmup_steps > 0 and actual_step < warmup_steps:
            return float(actual_step) / float(max(1, warmup_steps))
        return max(0.1, 1.0 - (actual_step - warmup_steps) / (args.training_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # IMPORTANT: Print the learning rate to verify it's correct when resuming
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Initial learning rate: {current_lr:.2e}")
    
    # If LR is explicitly overridden, skip optimizer/scheduler state to avoid restoring old LR.
    if lr_overridden:
        print("⚠️  --lr provided; skipping optimizer/scheduler state load to respect new LR")
        optimizer_state_path = None
        scheduler_state_path = None

    # Load optimizer and scheduler states if resuming
    if optimizer_state_path and optimizer_state_path.exists():
        print(f"Loading optimizer state from {optimizer_state_path}")
        try:
            optimizer_state = torch.load(optimizer_state_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            print(f"⚠️  Skipping optimizer state (mismatch): {exc}")
            try:
                loaded_groups = optimizer_state.get("param_groups", [])
                current_groups = optimizer.param_groups
                print(f"    loaded param_groups: {len(loaded_groups)} | current: {len(current_groups)}")
                if loaded_groups:
                    print(f"    loaded group0 params: {len(loaded_groups[0].get('params', []))}")
                if current_groups:
                    print(f"    current group0 params: {len(current_groups[0].get('params', []))}")
                trainable_tensors = [p for p in policy.parameters() if p.requires_grad]
                trainable_elems = sum(p.numel() for p in trainable_tensors)
                print(f"    trainable param tensors: {len(trainable_tensors)} | elements: {trainable_elems}")
                for name, module in policy.named_children():
                    mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    if mod_trainable > 0:
                        print(f"    trainable elements in policy.{name}: {mod_trainable}")
                if hasattr(policy, "model"):
                    model = policy.model
                    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"    trainable elements in policy.model: {model_trainable}")
                    if hasattr(model, "vlm_with_expert"):
                        vlm = model.vlm_with_expert
                        vlm_trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
                        print(f"    trainable elements in policy.model.vlm_with_expert: {vlm_trainable}")
            except Exception as debug_exc:
                print(f"    (debug) Could not inspect optimizer state: {debug_exc}")
            optimizer_state_path = None
        except Exception as exc:
            print(f"⚠️  Skipping optimizer state (load failed): {exc}")
            optimizer_state_path = None
    
    if scheduler_state_path and scheduler_state_path.exists():
        print(f"Loading scheduler state from {scheduler_state_path}")
        try:
            scheduler.load_state_dict(torch.load(scheduler_state_path, map_location=device))
        except ValueError as exc:
            print(f"⚠️  Skipping scheduler state (mismatch): {exc}")
            scheduler_state_path = None
    
    # Print training config
    # Setup augmentation pipeline
    augmentation_transforms = None
    if args.use_augmentation and AUGMENTATION_AVAILABLE:
        if args.augmentation_strength == "conservative":
            conservative = True
        elif args.augmentation_strength == "minimal":
            conservative = "minimal"
        else:  # aggressive
            conservative = False
        
        augmentation_transforms = get_augmentation_pipeline(
            image_aug=True,
            action_noise=False,  # Conservative: don't add noise to actions
            temporal_crop=False,
            conservative=conservative,
        )
        print(f"✅ Data augmentation enabled ({args.augmentation_strength})")
    elif args.use_augmentation and not AUGMENTATION_AVAILABLE:
        print("⚠️  Augmentation requested but augmentation_utils.py not found")
    
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Training steps: {args.training_steps}")
    print(f"Starting step: {starting_step}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Chunk size: {policy.config.chunk_size}")
    print(f"Data augmentation: {'enabled' if augmentation_transforms else 'disabled'}")
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
    accumulation_counter = 0
    
    # Track timing for diagnostics
    batch_times = []
    first_batches = 10
    
    while not done:
        for batch in dataloader:
            batch_start = time.time()
            
            # Apply augmentation before preprocessing
            if augmentation_transforms:
                batch = apply_augmentations(batch, augmentation_transforms)
            
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
            
            # Scale loss by accumulation steps for correct gradient magnitude
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            
            running_loss += loss.item()
            accumulation_counter += 1
            
            # Only update weights every N accumulation steps
            should_update = accumulation_counter >= args.gradient_accumulation_steps
            if should_update:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                scheduler.step()
                
                accumulation_counter = 0
                step += 1
            
            # Track batch timing for first batches
            if len(batch_times) < first_batches:
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                if len(batch_times) == first_batches:
                    avg_time = sum(batch_times) / len(batch_times)
                    print(f"\n⏱️  Avg time/batch (first {first_batches}): {avg_time:.3f}s")
                    print(f"   Est. time per {args.checkpoint_freq} steps: {avg_time * args.checkpoint_freq * args.gradient_accumulation_steps / 60:.1f}min\n")
            
            # Only log/checkpoint after actual optimizer updates
            if should_update:
                # Log progress
                if step % args.log_freq == 0:
                    avg_loss = running_loss / (args.log_freq * args.gradient_accumulation_steps) if step > 0 else loss.item()
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Log to wandb
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": current_lr,
                        "step": step,
                    })
                    
                    print(f"Step {step:5d}/{args.training_steps} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
                    running_loss = 0.0
                
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
                
                # Clear MPS cache periodically (not too often to avoid slowdown)
                if device.type == "mps" and step % 500 == 0:
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
