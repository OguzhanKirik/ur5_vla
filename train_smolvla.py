#!/usr/bin/env python

"""
Train SmolVLA policy on UR5 grasping dataset with language annotations.

Uses LeRobot's SmolVLAPolicy for vision-language-action learning.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from PIL import Image
import wandb

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoProcessor


class UR5GraspDataset(Dataset):
    """Dataset for UR5 grasping episodes compatible with LeRobot SmolVLA"""
    
    def __init__(self, dataset_dir, processor, chunk_size=10):
        """
        Args:
            dataset_dir: Path to dataset directory
            processor: SmolVLA processor for tokenizing language
            chunk_size: Number of future actions to predict (default 10)
        """
        self.dataset_dir = Path(dataset_dir)
        self.processor = processor
        self.chunk_size = chunk_size
        self.max_text_length = 77  # Standard for vision-language models
        
        # Load metadata
        metadata_path = self.dataset_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load all episodes and create samples
        # Each sample is: (observation_frame, next_chunk_size_actions, instruction)
        self.samples = []
        
        for ep_info in self.metadata['episodes']:
            ep_id = ep_info['episode_id']
            ep_path = self.dataset_dir / f"episode_{ep_id:04d}.json"
            with open(ep_path, 'r') as f:
                episode = json.load(f)
                
            instruction = episode['instruction']
            frames = episode['frames']
            video_paths = episode['videos']
            
            # Create samples from each frame that has at least chunk_size future actions
            for i in range(len(frames) - chunk_size + 1):
                self.samples.append({
                    'episode_id': ep_id,
                    'frame_idx': i,
                    'instruction': instruction,
                    'video_paths': video_paths,
                    'state': frames[i]['state'],
                    'actions': [frames[j]['action'] for j in range(i, min(i + chunk_size, len(frames)))]
                })
        
        print(f"Loaded {len(self.samples)} training samples from {len(self.metadata['episodes'])} episodes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get instruction
        instruction = sample['instruction']
        
        # Load the specific frame from videos
        frame_idx = sample['frame_idx']
        top_rgb_path = self.dataset_dir / sample['video_paths']['top_rgb']
        wrist_rgb_path = self.dataset_dir / sample['video_paths']['wrist_rgb']
        
        # Load single frame at frame_idx
        top_image = self.load_video_frame(top_rgb_path, frame_idx)
        wrist_image = self.load_video_frame(wrist_rgb_path, frame_idx)
        
        # Get state (single state vector)
        state = np.array(sample['state'], dtype=np.float32)
        
        # Get action chunk (sequence of future actions)
        actions = np.array(sample['actions'], dtype=np.float32)
        
        # Pad actions to chunk_size if needed
        if len(actions) < self.chunk_size:
            pad_length = self.chunk_size - len(actions)
            actions = np.vstack([actions, np.tile(actions[-1:], (pad_length, 1))])
        
        # Tokenize language instruction using processor
        text_inputs = self.processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length
        )
        
        # Return single observation (image + state) with action chunk
        # Use camera1/camera2 naming convention expected by SmolVLA policy
        return {
            'observation.images.camera1': self.preprocess_image(top_image),
            'observation.images.camera2': self.preprocess_image(wrist_image),
            'observation.state': torch.from_numpy(state),
            'observation.language.tokens': text_inputs['input_ids'].squeeze(0),
            'observation.language.attention_mask': text_inputs['attention_mask'].squeeze(0).bool(),  # Convert to boolean
            'action': torch.from_numpy(actions),
        }
    
    def load_video_frame(self, video_path, frame_idx):
        """Load a specific frame from video file"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        else:
            # Return black image if frame not found
            return Image.new('RGB', (640, 480))
    
    def preprocess_image(self, image):
        """Preprocess image to tensor - ensures consistent size"""
        # Ensure image is always the same size (640x480 from our videos)
        if image.size != (640, 480):
            image = image.resize((640, 480))
        
        # Convert PIL image to numpy array
        img_array = np.array(image).astype(np.float32)
        # Convert to tensor HWC -> CHW
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        # Normalize to [0, 1] range
        tensor = tensor / 255.0
        return tensor


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA for UR5 grasping")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/smolvla_ur5", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk size (number of future actions to predict)")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--checkpoint-freq", type=int, default=1000, help="Checkpoint frequency")
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="smolvla-ur5-grasping",
            name=f"run-{output_dir.name}",
            config=vars(args)
        )
        print("📊 Weights & Biases initialized!\n")
    
    # Load SmolVLA processor for tokenization
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    
    # Load dataset
    print("Loading dataset...")
    dataset = UR5GraspDataset(args.dataset_dir, processor=processor, chunk_size=args.chunk_size)
    
    # Split train/val
    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if device.type == "mps" else 4,
        pin_memory=device.type == "cuda",
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type == "mps" else 4,
        pin_memory=device.type == "cuda",
        drop_last=False
    )
    
    # Load SmolVLA policy from pretrained
    print("\nLoading SmolVLA policy from lerobot/smolvla_base...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.train()
    policy.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy: {total_params:,} total params, {trainable_params:,} trainable\n")
    
    # Optimizer with AdamW
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    warmup_steps = 500
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.1, 1.0 - (current_step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Action chunk size: {args.chunk_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Checkpoint frequency: every {args.checkpoint_freq} steps")
    print(f"Device: {device}")
    print("=" * 70 + "\n")
    
    # Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        # Training
        policy.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            loss, loss_dict = policy.forward(batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            total_train_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log to wandb
            if args.use_wandb and global_step % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "step": global_step,
                    "epoch": epoch
                })
            
            # Save checkpoint
            if global_step % args.checkpoint_freq == 0 and global_step > 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n💾 Saving checkpoint at step {global_step}...")
                policy.save_pretrained(checkpoint_dir)
                print(f"✅ Checkpoint saved to {checkpoint_dir}\n")
            
            global_step += 1
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        policy.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, _ = policy.forward(batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "epoch_train_loss": avg_train_loss,
                "epoch_val_loss": avg_val_loss,
                "epoch": epoch+1
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = output_dir / "best_model"
            best_model_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ New best model! Saving to {best_model_path}")
            policy.save_pretrained(best_model_path)
    
    # Save final model
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")
    
    final_model_path = output_dir / "final_model"
    final_model_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving final model to {final_model_path}...")
    policy.save_pretrained(final_model_path)
    
    if args.use_wandb:
        wandb.log({"final_val_loss": best_val_loss})
        wandb.finish()
        print("📊 Training metrics saved to Weights & Biases!")
    
    print(f"\n✅ Training complete!")
    print(f"Best model saved to: {(output_dir / 'best_model').absolute()}")
    print(f"Final model saved to: {final_model_path.absolute()}")


if __name__ == "__main__":
    main()
