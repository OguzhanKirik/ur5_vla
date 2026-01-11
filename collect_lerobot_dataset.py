"""
LeRobot-compatible Dataset Collection Script for UR5 Grasping Environment

This script collects demonstration data and saves it in LeRobot v3.0 format.
Compatible with LeRobot training pipelines.

Usage:
    python collect_lerobot_dataset.py --episodes 50 --output-dir ./datasets --dataset-name ur5_grasp_sim
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Import LeRobot components
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_initial_features

# Import your environment
from pybullet_env import UR5GraspEnv


def collect_random_episode(env, num_steps=200):
    """
    Collect one episode with random actions.
    
    In practice, replace this with:
    - Teleoperation using robot_controller_rrt.py
    - Scripted demonstrations
    - Human demonstrations via keyboard/joystick
    
    Returns:
        frames: list of dicts containing observations and actions
        success: whether episode was successful
    """
    frames = []
    obs = env.reset()
    success = False
    
    for step in range(num_steps):
        # Random action: 6 joint velocities + 1 gripper
        # Replace this with actual demonstrations
        action = np.random.randn(7) * 0.3
        action = np.clip(action, -1, 1)
        
        # Get current state
        state = obs['state']  # [joint_positions (6) + gripper_position (1) + gripper_state (1)]
        
        # Get camera images (convert to PIL Images for LeRobot)
        image_with_depth = Image.fromarray(obs['images']['observation.images.image_with_depth'])
        image_front = Image.fromarray(obs['images']['observation.images.image'])
        hand_image = Image.fromarray(obs['images']['observation.images.hand_image'])
        
        # Create frame dict following LeRobot convention
        frame = {
            'observation.state': state.astype(np.float32),
            'observation.images.top': image_with_depth,  # Top-down camera
            'observation.images.front': image_front,      # Front camera
            'observation.images.wrist': hand_image,       # Wrist camera
            'action': action.astype(np.float32),
            'timestamp': step / 30.0,  # Assuming 30 FPS
            'frame_index': step,
        }
        frames.append(frame)
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        # Check success condition (customize based on your task)
        if reward > 0.9:  # Example success condition
            success = True
        
        if done:
            break
    
    return frames, success


def define_dataset_features():
    """
    Define the features structure for the dataset.
    This tells LeRobot what data types and shapes to expect.
    """
    features = {
        # Robot state: 6 joint positions + 1 gripper position + 1 gripper state
        'observation.state': {
            'dtype': 'float32',
            'shape': (8,),
            'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'gripper_position', 'gripper_state']
        },
        
        # Camera observations (images will be stored as videos)
        'observation.images.top': {
            'dtype': 'video',
            'shape': (480, 640, 3),
            'info': {'video.fps': 30, 'video.codec': 'av1'}
        },
        'observation.images.front': {
            'dtype': 'video',
            'shape': (480, 640, 3),
            'info': {'video.fps': 30, 'video.codec': 'av1'}
        },
        'observation.images.wrist': {
            'dtype': 'video',
            'shape': (480, 640, 3),
            'info': {'video.fps': 30, 'video.codec': 'av1'}
        },
        
        # Action: 6 joint velocities/positions + 1 gripper action
        'action': {
            'dtype': 'float32',
            'shape': (7,),
            'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'gripper']
        },
    }
    return features


def main():
    parser = argparse.ArgumentParser(description="Collect UR5 grasping dataset in LeRobot format")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to collect")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="ur5_grasp_sim", help="Dataset name")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument("--robot-type", type=str, default="ur5", help="Robot type identifier")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--task", type=str, default="Pick and place object", help="Task description")
    parser.add_argument("--use-videos", action="store_true", default=True, help="Store images as videos (recommended)")
    
    args = parser.parse_args()
    
    # Use dataset name as repo_id if not specified
    if args.repo_id is None:
        args.repo_id = args.dataset_name
    
    print("="*70)
    print("UR5 Grasp Dataset Collection (LeRobot Format)")
    print("="*70)
    print(f"Repo ID: {args.repo_id}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.steps}")
    print(f"Output: {args.output_dir}")
    print(f"Robot type: {args.robot_type}")
    print(f"Task: {args.task}")
    print(f"FPS: {args.fps}")
    print(f"Use videos: {args.use_videos}")
    print(f"GUI: {'Enabled' if args.gui else 'Disabled'}")
    print("="*70)
    
    # Create environment
    print("\n[1/4] Initializing environment...")
    env = UR5GraspEnv(gui=args.gui, fps=args.fps)
    
    # Define dataset features
    print("[2/4] Defining dataset features...")
    features = define_dataset_features()
    
    # Create LeRobot dataset
    print(f"[3/4] Creating LeRobot dataset at {args.output_dir}/{args.repo_id}...")
    try:
        lerobot_dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output_dir,
            fps=args.fps,
            robot_type=args.robot_type,
            features=features,
            use_videos=args.use_videos,
        )
        print(f"✓ Dataset created successfully!")
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        print("\nMake sure:")
        print("  1. Output directory is writable")
        print("  2. Dataset doesn't already exist at this location")
        print(f"  3. Remove existing directory: rm -rf {args.output_dir}/{args.repo_id}")
        return
    
    # Collect episodes
    print(f"\n[4/4] Collecting {args.episodes} episodes...")
    successful_episodes = 0
    
    for episode_idx in tqdm(range(args.episodes), desc="Episodes"):
        try:
            # Collect episode data
            frames, success = collect_random_episode(env, num_steps=args.steps)
            
            if success:
                successful_episodes += 1
            
            # Add frames to dataset
            for frame_idx, frame in enumerate(frames):
                lerobot_dataset.add_frame(frame)
            
            # Save episode
            lerobot_dataset.save_episode(
                task=args.task,
                encode_videos=args.use_videos
            )
            
            print(f"  Episode {episode_idx}: {len(frames)} frames {'[SUCCESS]' if success else ''}")
            
        except Exception as e:
            print(f"  ✗ Error in episode {episode_idx}: {e}")
            continue
    
    # Consolidate dataset
    print("\nConsolidating dataset...")
    try:
        lerobot_dataset.consolidate()
        print("✓ Dataset consolidated!")
    except Exception as e:
        print(f"✗ Error consolidating: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("Collection Summary")
    print("="*70)
    print(f"Total episodes collected: {args.episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Total frames: {lerobot_dataset.num_frames if hasattr(lerobot_dataset, 'num_frames') else 'N/A'}")
    print(f"Dataset location: {args.output_dir}/{args.repo_id}")
    print("\nNext steps:")
    print("  1. Train a policy:")
    print(f"     python -m lerobot.scripts.train policy=act env.task={args.task} dataset_repo_id={args.repo_id}")
    print("\n  2. Push to HuggingFace Hub:")
    print(f"     huggingface-cli upload {args.repo_id} {args.output_dir}/{args.repo_id}")
    print("="*70)


if __name__ == "__main__":
    main()
