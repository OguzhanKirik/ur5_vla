"""
Dataset Collection Script for UR5 Grasping Environment

This script collects demonstration data and saves it in LeRobot v3.0 format,
compatible with the wuc1/pybullet_ur5 dataset structure.
"""

import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import cv2

from pybullet_env import UR5GraspEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def collect_random_episode(env, num_steps=200):
    """
    Collect one episode with random actions.
    In practice, you'd replace this with:
    - Teleoperation
    - Scripted demonstrations
    - Human demonstrations via keyboard/joystick
    
    Returns:
        frames: list of dicts containing observations and actions
    """
    frames = []
    obs = env.reset()
    
    for step in range(num_steps):
        # Random action (replace with your demonstration method)
        action = np.random.randn(7) * 0.3
        action = np.clip(action, -1, 1)
        
        # Store current observation and action
        frame = {
            'observation.state': obs['state'].copy(),
            'observation.images.image_with_depth': obs['images']['observation.images.image_with_depth'].copy(),
            'observation.images.image': obs['images']['observation.images.image'].copy(),
            'observation.images.hand_image': obs['images']['observation.images.hand_image'].copy(),
            'action': action.copy(),
            'timestamp': step / 30.0,  # 30 FPS
            'frame_index': step,
        }
        frames.append(frame)
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        if done:
            break
    
    return frames


def collect_teleoperated_episode(env):
    """
    Placeholder for teleoperation-based data collection.
    You would implement keyboard/joystick control here.
    """
    print("Teleoperation mode not implemented yet.")
    print("Please use random mode or implement your own control method.")
    return []


def save_dataset_lerobot_format(episodes_data, output_dir, dataset_name, fps=30):
    """
    Save collected episodes in LeRobot v3.0 format.
    
    This creates the dataset structure but you'll still need to use
    LeRobotDataset API to properly format and save everything.
    """
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating LeRobot dataset at: {output_path}")
    
    # Create dataset using LeRobot's create method
    # Note: You'll need to adapt this to the actual LeRobot v3.0 API
    try:
        # Define features based on our environment
        features = {
            'observation.state': {'dtype': 'float32', 'shape': (8,)},
            'observation.images.image_with_depth': {'dtype': 'video', 'shape': (480, 640, 3)},
            'observation.images.image': {'dtype': 'video', 'shape': (480, 640, 3)},
            'observation.images.hand_image': {'dtype': 'video', 'shape': (480, 640, 3)},
            'action': {'dtype': 'float32', 'shape': (7,)},
        }
        
        # Create empty dataset
        dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            root=output_dir,
            robot_type="ur5_robotiq_85",
            fps=fps,
            features=features,
        )
        
        # Add episodes
        for ep_idx, episode_frames in enumerate(tqdm(episodes_data, desc="Saving episodes")):
            # Add episode to dataset
            for frame in episode_frames:
                dataset.add_frame(frame)
            dataset.save_episode(episode_index=ep_idx)
        
        # Consolidate and save metadata
        dataset.consolidate()
        print(f"✓ Dataset saved successfully!")
        print(f"  Episodes: {len(episodes_data)}")
        print(f"  Total frames: {sum(len(ep) for ep in episodes_data)}")
        
    except Exception as e:
        print(f"✗ Error creating LeRobot dataset: {e}")
        print("\nNote: For proper dataset creation, you may need to use:")
        print("  python -m lerobot.scripts.push_dataset_to_hub \\")
        print(f"    --local-dir {output_path} \\")
        print(f"    --repo-id <your-hf-username>/{dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Collect UR5 grasping dataset")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to collect")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--output-dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="ur5_grasp", help="Dataset name")
    parser.add_argument("--mode", type=str, choices=["random", "teleop"], default="random",
                       help="Collection mode: random actions or teleoperation")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    
    print("="*70)
    print("UR5 Grasp Dataset Collection")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {args.steps}")
    print(f"Output: {args.output_dir}/{args.dataset_name}")
    print(f"GUI: {'Enabled' if args.gui else 'Disabled'}")
    print("="*70)
    
    # Create environment
    env = UR5GraspEnv(gui=args.gui, fps=args.fps)
    
    # Collect episodes
    all_episodes = []
    try:
        for ep in range(args.episodes):
            print(f"\n[Episode {ep+1}/{args.episodes}]")
            
            if args.mode == "random":
                frames = collect_random_episode(env, num_steps=args.steps)
            elif args.mode == "teleop":
                frames = collect_teleoperated_episode(env)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
            
            if len(frames) > 0:
                all_episodes.append(frames)
                print(f"  ✓ Collected {len(frames)} frames")
            else:
                print(f"  ✗ No frames collected")
        
        # Save dataset
        if len(all_episodes) > 0:
            save_dataset_lerobot_format(
                all_episodes,
                args.output_dir,
                args.dataset_name,
                fps=args.fps
            )
        else:
            print("\n✗ No episodes collected!")
    
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
        if len(all_episodes) > 0:
            print(f"Saving {len(all_episodes)} collected episodes...")
            save_dataset_lerobot_format(
                all_episodes,
                args.output_dir,
                args.dataset_name,
                fps=args.fps
            )
    
    finally:
        env.close()
        print("\nDataset collection complete!")


if __name__ == "__main__":
    main()
