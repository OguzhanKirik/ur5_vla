"""
Data augmentation utilities for robotic manipulation training.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict
import random


class RoboticImageAugmentation:
    """
    Image augmentation designed for robotic manipulation tasks.
    Conservative augmentations that preserve spatial relationships.
    """
    
    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        random_crop_scale=(0.9, 1.0),
        gaussian_blur_prob=0.1,
        gaussian_noise_std=0.02,
        enabled=True,
    ):
        self.enabled = enabled
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_crop_scale = random_crop_scale
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_noise_std = gaussian_noise_std
        
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to image observations in batch.
        
        Args:
            batch: Dictionary with keys like "observation.images.camera1", etc.
        
        Returns:
            Augmented batch
        """
        if not self.enabled:
            return batch
        
        # Find all image keys
        image_keys = [k for k in batch.keys() if "observation.images" in k and not k.endswith("_is_pad")]
        
        for key in image_keys:
            imgs = batch[key]  # Shape: [B, T, C, H, W] or [B, C, H, W]
            
            # Flatten temporal dimension if present
            original_shape = imgs.shape
            if len(imgs.shape) == 5:  # [B, T, C, H, W]
                B, T, C, H, W = imgs.shape
                imgs = imgs.reshape(B * T, C, H, W)
            
            # Apply augmentations
            imgs = self._augment_batch(imgs)
            
            # Reshape back
            if len(original_shape) == 5:
                imgs = imgs.reshape(B, T, C, H, W)
            
            batch[key] = imgs
        
        return batch
    
    def _augment_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a batch of images."""
        B, C, H, W = imgs.shape
        augmented = []
        
        for i in range(B):
            img = imgs[i]
            
            # Color jitter (most important for robustness)
            if random.random() > 0.3:  # Apply 70% of the time
                img = F.adjust_brightness(img, 1 + random.uniform(-self.brightness, self.brightness))
                img = F.adjust_contrast(img, 1 + random.uniform(-self.contrast, self.contrast))
                img = F.adjust_saturation(img, 1 + random.uniform(-self.saturation, self.saturation))
                img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))
            
            # Random crop (very conservative to preserve spatial info)
            if random.random() > 0.5:
                scale = random.uniform(*self.random_crop_scale)
                new_h, new_w = int(H * scale), int(W * scale)
                if new_h < H and new_w < W:
                    top = random.randint(0, H - new_h)
                    left = random.randint(0, W - new_w)
                    img = F.crop(img, top, left, new_h, new_w)
                    img = F.resize(img, [H, W], antialias=True)
            
            # Gaussian blur (occasionally)
            if random.random() < self.gaussian_blur_prob:
                kernel_size = random.choice([3, 5])
                img = F.gaussian_blur(img, kernel_size)
            
            # Gaussian noise (very light)
            if random.random() < 0.2:
                noise = torch.randn_like(img) * self.gaussian_noise_std
                img = torch.clamp(img + noise, 0, 1)
            
            augmented.append(img)
        
        return torch.stack(augmented)


class ActionNoise:
    """
    Add small noise to action labels for regularization.
    Helps model be more robust to small action variations.
    """
    
    def __init__(self, noise_std=0.01, enabled=True):
        self.noise_std = noise_std
        self.enabled = enabled
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add noise to action."""
        if not self.enabled or "action" not in batch:
            return batch
        
        actions = batch["action"]
        noise = torch.randn_like(actions) * self.noise_std
        batch["action"] = actions + noise
        
        return batch


class TemporalCrop:
    """
    Randomly crop temporal sequences to encourage temporal invariance.
    Only use if n_obs_steps > 1.
    """
    
    def __init__(self, min_length_ratio=0.8, enabled=True):
        self.min_length_ratio = min_length_ratio
        self.enabled = enabled
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly crop temporal dimension."""
        if not self.enabled:
            return batch
        
        # Find temporal keys
        for key in list(batch.keys()):
            if "observation" in key and len(batch[key].shape) == 5:  # [B, T, ...]
                # Random temporal crop implementation here
                pass
        
        return batch


def get_augmentation_pipeline(
    image_aug=True,
    action_noise=False,
    temporal_crop=False,
    conservative=True,
):
    """
    Get augmentation pipeline for robotic training.
    
    Args:
        image_aug: Enable image augmentation
        action_noise: Add noise to action labels
        temporal_crop: Randomly crop temporal sequences
        conservative: Use conservative augmentation params (recommended for robotics)
    
    Returns:
        List of augmentation transforms
    """
    transforms = []
    
    if image_aug:
        if conservative == "minimal":  # NEW: For sim-to-sim scenarios (minimal augmentation)
            aug = RoboticImageAugmentation(
                brightness=0.05,  # Very minimal
                contrast=0.05,
                saturation=0.05,
                hue=0.02,
                random_crop_scale=(0.98, 1.0),  # Almost no crop
                gaussian_blur_prob=0.02,
                gaussian_noise_std=0.005,
                enabled=True,
            )
        elif conservative == True or conservative == "conservative":  # Default conservative
            aug = RoboticImageAugmentation(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                random_crop_scale=(0.95, 1.0),
                gaussian_blur_prob=0.05,
                gaussian_noise_std=0.01,
            )
        else:  # Aggressive augmentation
            aug = RoboticImageAugmentation(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15,
                random_crop_scale=(0.85, 1.0),
                gaussian_blur_prob=0.15,
                gaussian_noise_std=0.03,
            )
        transforms.append(aug)
    
    if action_noise:
        transforms.append(ActionNoise(noise_std=0.01 if conservative else 0.02))
    
    if temporal_crop:
        transforms.append(TemporalCrop(min_length_ratio=0.85))
    
    return transforms


def apply_augmentations(batch: Dict[str, torch.Tensor], transforms) -> Dict[str, torch.Tensor]:
    """Apply list of augmentation transforms to batch."""
    for transform in transforms:
        batch = transform(batch)
    return batch
