"""
Teleoperated Dataset Collection using Robot Controller

This script integrates robot_controller_rrt.py for manual teleoperation
with LeRobot dataset recording capabilities.

Usage:
    python collect_with_teleop.py --episodes 50 --dataset-name my_ur5_demos

Controls (during recording):
    - Use sliders to control robot joints
    - Type commands: goto x y z, open, close, etc.
    - Press 's' to start recording episode
    - Press 'e' to end episode
    - Press 'q' to quit
"""

import numpy as np
import argparse
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pybullet_env import UR5GraspEnv
import pybullet as p


class TeleopDataCollector:
    """Collects dataset through teleoperation with real-time feedback"""
    
    def __init__(self, env, lerobot_dataset, max_steps=500):
        self.env = env
        self.dataset = lerobot_dataset
        self.max_steps = max_steps
        
        # Episode state
        self.recording = False
        self.episode_frames = []
        self.current_step = 0
        self.last_action = np.zeros(7)
        
        print("\n" + "="*70)
        print("Teleoperation Controls:")
        print("="*70)
        print("  s - Start recording episode")
        print("  e - End current episode")
        print("  r - Reset environment (without saving)")
        print("  q - Quit and save dataset")
        print("\nUse sliders to control robot joints")
        print("="*70 + "\n")
    
    def get_current_observation_and_action(self):
        """Get current state and compute action from joint positions"""
        # Get observation
        obs = {
            'state': self.env.get_state(),
            'images': self.env.get_camera_images()
        }
        
        # Compute action as difference from last observation
        current_joints = obs['state'][:7]  # First 7 are joint positions
        action = current_joints - self.last_action[:7] if hasattr(self, 'last_action') else np.zeros(7)
        self.last_action = current_joints.copy()
        
        return obs, action
    
    def create_frame(self, obs, action, step):
        """Convert observation and action to LeRobot frame format"""
        # Convert images to PIL
        image_top = Image.fromarray(obs['images']['observation.images.image_with_depth'])
        image_front = Image.fromarray(obs['images']['observation.images.image'])
        image_wrist = Image.fromarray(obs['images']['observation.images.hand_image'])
        
        frame = {
            'observation.state': obs['state'].astype(np.float32),
            'observation.images.top': image_top,
            'observation.images.front': image_front,
            'observation.images.wrist': image_wrist,
            'action': action.astype(np.float32),
            'timestamp': step / 30.0,
            'frame_index': step,
        }
        return frame
    
    def start_recording(self):
        """Start recording a new episode"""
        if self.recording:
            print("⚠ Already recording!")
            return
        
        self.recording = True
        self.episode_frames = []
        self.current_step = 0
        self.env.reset()
        print(f"🔴 Recording started (Episode {self.dataset.num_episodes if hasattr(self.dataset, 'num_episodes') else 0})")
    
    def stop_recording(self, task="Pick and place"):
        """Stop recording and save episode"""
        if not self.recording:
            print("⚠ Not currently recording!")
            return
        
        self.recording = False
        
        if len(self.episode_frames) == 0:
            print("⚠ No frames recorded, episode discarded")
            return
        
        print(f"⏹ Recording stopped ({len(self.episode_frames)} frames)")
        print("💾 Saving episode...")
        
        try:
            # Add all frames to dataset
            for frame in self.episode_frames:
                self.dataset.add_frame(frame)
            
            # Save episode
            self.dataset.save_episode(task=task, encode_videos=True)
            print(f"✓ Episode saved successfully!")
            
        except Exception as e:
            print(f"✗ Error saving episode: {e}")
        
        # Reset for next episode
        self.episode_frames = []
        self.current_step = 0
    
    def update(self):
        """Update loop - call this continuously during teleoperation"""
        if not self.recording:
            return
        
        if self.current_step >= self.max_steps:
            print(f"⚠ Max steps ({self.max_steps}) reached, stopping recording")
            self.stop_recording()
            return
        
        # Get current observation and action
        obs, action = self.get_current_observation_and_action()
        
        # Create and store frame
        frame = self.create_frame(obs, action, self.current_step)
        self.episode_frames.append(frame)
        self.current_step += 1
        
        # Visual feedback
        if self.current_step % 10 == 0:
            print(f"  Recording: {self.current_step}/{self.max_steps} frames", end='\r')


def run_teleoperation(args):
    """Main teleoperation loop with dataset recording"""
    
    # Create environment
    print("[1/3] Initializing environment...")
    env = UR5GraspEnv(gui=True, fps=args.fps)  # GUI always enabled for teleop
    
    # Define features
    print("[2/3] Creating dataset...")
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (8,)},
        'observation.images.top': {'dtype': 'video', 'shape': (480, 640, 3)},
        'observation.images.front': {'dtype': 'video', 'shape': (480, 640, 3)},
        'observation.images.wrist': {'dtype': 'video', 'shape': (480, 640, 3)},
        'action': {'dtype': 'float32', 'shape': (7,)},
    }
    
    # Create LeRobot dataset
    repo_id = args.repo_id if args.repo_id else args.dataset_name
    
    try:
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=args.output_dir,
            fps=args.fps,
            robot_type=args.robot_type,
            features=features,
            use_videos=True,
        )
        print(f"✓ Dataset created at {args.output_dir}/{repo_id}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Create collector
    print("[3/3] Starting teleoperation interface...")
    collector = TeleopDataCollector(env, lerobot_dataset, max_steps=args.max_steps)
    
    # Create control window for keyboard input
    cv2.namedWindow("Teleop Controls")
    cv2.createTrackbar("Status", "Teleop Controls", 0, 1, lambda x: None)
    
    episodes_collected = 0
    target_episodes = args.episodes
    
    print(f"\nTarget: {target_episodes} episodes")
    print("Press keys in the 'Teleop Controls' window")
    
    try:
        while episodes_collected < target_episodes:
            # Get keyboard input
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('s'):
                collector.start_recording()
            elif key == ord('e'):
                collector.stop_recording(task=args.task)
                episodes_collected += 1
                print(f"Progress: {episodes_collected}/{target_episodes} episodes")
            elif key == ord('r'):
                env.reset()
                print("🔄 Environment reset")
            elif key == ord('q'):
                print("\n⚠ Quit requested")
                break
            
            # Update collector (records frame if recording)
            collector.update()
            
            # Step simulation
            p.stepSimulation()
            
            # Update display
            images = env.get_camera_images()
            combined = np.hstack([
                cv2.resize(images['observation.images.image_with_depth'], (320, 240)),
                cv2.resize(images['observation.images.image'], (320, 240))
            ])
            cv2.imshow("Teleop Controls", combined)
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    finally:
        # Save any recording in progress
        if collector.recording:
            print("\n⚠ Recording in progress, saving...")
            collector.stop_recording()
        
        # Consolidate dataset
        print("\n💾 Consolidating dataset...")
        try:
            lerobot_dataset.consolidate()
            print("✓ Dataset consolidated!")
        except Exception as e:
            print(f"⚠ Error consolidating: {e}")
        
        # Cleanup
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(f"Collection complete!")
        print(f"Episodes collected: {episodes_collected}")
        print(f"Dataset location: {args.output_dir}/{repo_id}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Collect dataset through teleoperation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="ur5_teleop", help="Dataset name")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repo ID")
    parser.add_argument("--robot-type", type=str, default="ur5", help="Robot type")
    parser.add_argument("--task", type=str, default="Pick and place", help="Task description")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Teleoperated Dataset Collection")
    print("="*70)
    
    run_teleoperation(args)


if __name__ == "__main__":
    main()
