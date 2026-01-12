#!/usr/bin/env python
"""
Quick training test with subset of data to verify everything works
"""

import json
import shutil
from pathlib import Path

# Create a test subset with only 5 episodes
dataset_dir = Path("./datasets/smolvla_ur5_grasp")
test_dataset_dir = Path("./datasets/smolvla_ur5_grasp_test")

# Create test dataset directory
test_dataset_dir.mkdir(parents=True, exist_ok=True)

# Copy metadata and modify it
with open(dataset_dir / "metadata.json", 'r') as f:
    metadata = json.load(f)

# Keep only first 5 episodes
test_episodes = metadata['episodes'][:5]
metadata['episodes'] = test_episodes
metadata['num_episodes'] = len(test_episodes)
metadata['dataset_name'] = 'smolvla_ur5_grasp_test'

with open(test_dataset_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# Copy episode files and videos
(test_dataset_dir / "videos").mkdir(exist_ok=True)
(test_dataset_dir / "images").mkdir(exist_ok=True)

for ep in test_episodes:
    ep_id = ep['episode_id']
    
    # Copy episode JSON
    shutil.copy(
        dataset_dir / f"episode_{ep_id:04d}.json",
        test_dataset_dir / f"episode_{ep_id:04d}.json"
    )
    
    # Copy video files
    with open(dataset_dir / f"episode_{ep_id:04d}.json", 'r') as f:
        ep_data = json.load(f)
    
    for video_type, video_path in ep_data['videos'].items():
        src = dataset_dir / video_path
        dst = test_dataset_dir / video_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, dst)

print(f"✅ Created test dataset with 5 episodes at: {test_dataset_dir}")
print("\nNow run:")
print("python3 train_smolvla.py --dataset-dir ./datasets/smolvla_ur5_grasp_test --output-dir ./checkpoints/smolvla_ur5_test --batch-size 2 --epochs 2 --lr 1e-5 --chunk-size 10")
