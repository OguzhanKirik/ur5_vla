#!/usr/bin/env python

"""
Evaluation script for SmolVLA policy on UR5 grasping task.

This script loads a trained SmolVLA model and uses it to control the robot
in the PyBullet simulation environment (same as used for data collection).

Usage:
    python eval_smolvla_lerobot.py --checkpoint-dir ./checkpoints/smolvla_lerobot/final_model
    python eval_smolvla_lerobot.py --checkpoint-dir ./checkpoints/smolvla_lerobot/checkpoint-5000 --num-episodes 5
"""

import os
import sys

# Suppress warnings and set environment variables BEFORE any imports
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("lerobot").setLevel(logging.WARNING)

import numpy as np
import torch
from collections import deque
from PIL import Image
import pybullet as p
import pybullet_data
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from components import UR5RobotComponent, CameraComponent, ObjectsComponent, RobotController
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# Training objects and instructions (same as data collection)
OBJECT_INSTRUCTIONS = {
    0: "Pick up the red sphere and place it in the container",
    1: "Grasp the green sphere and put it in the box",
    2: "Pick up the blue cylinder and drop it in the container",
    3: "Pick up the red cylinder and place it in the container",
}


def sim_step(n: int = 1) -> None:
    """Perform `n` physics steps."""
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def get_robot_state(robot) -> np.ndarray:
    """Get current robot state as numpy array (8,): 6 joints + gripper_pos + gripper_state"""
    joint_positions = []
    for joint_idx in robot.arm_joint_indices:
        joint_state = p.getJointState(robot.robot_id, joint_idx)
        joint_positions.append(joint_state[0])
    
    if robot.gripper_joint_indices:
        gripper_pos = p.getJointState(robot.robot_id, robot.gripper_joint_indices[0])[0]
        gripper_state = 1.0 if gripper_pos > 0.4 else -1.0
    else:
        gripper_pos = 0.0
        gripper_state = 0.0
    
    return np.array(joint_positions + [gripper_pos, gripper_state], dtype=np.float32)


def capture_images(camera) -> dict:
    """Capture RGB images from top and wrist cameras as numpy arrays"""
    captured_data = camera.capture_all()
    images = {}
    
    for cam_name, rgb_array in captured_data.items():
        if rgb_array is not None:
            target_name = None
            if 'main' in cam_name.lower() or 'top' in cam_name.lower():
                target_name = 'top'
            elif 'wrist' in cam_name.lower():
                target_name = 'wrist'
            
            if target_name:
                images[target_name] = rgb_array.astype(np.uint8)
    
    # Ensure both cameras present
    if images:
        fallback = list(images.values())[0]
        if 'top' not in images:
            images['top'] = fallback
        if 'wrist' not in images:
            images['wrist'] = fallback
    else:
        dummy = np.ones((480, 640, 3), dtype=np.uint8) * 128
        images['top'] = dummy
        images['wrist'] = dummy
    
    return images


def load_model(checkpoint_dir, device):
    """Load the trained SmolVLA model."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"Loading model from: {checkpoint_dir}")
    
    # Load the policy
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    policy.eval()
    policy.to(device)
    
    print(f"Policy loaded: {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    return policy


def preprocess_obs(obs, policy, device, observation_history):
    """
    Preprocess observation for the policy.
    
    obs: dict with keys 'images' (dict of numpy arrays), 'state' (np.array)
    observation_history: deque storing past observations
    """
    import torch.nn.functional as F
    
    # Normalize images and convert to tensors
    normalized_images = {}
    for k, v in obs['images'].items():
        # Normalize to [0, 1] and convert to tensor
        img = torch.from_numpy(v.astype(np.float32) / 255.0)
        # Ensure channel-first format (C, H, W)
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)
        normalized_images[k] = img
    
    observation_history.append({
        'images': normalized_images,
        'state': torch.from_numpy(obs['state']).float(),
    })
    
    # Build batch with observation history
    n_obs_steps = policy.config.n_obs_steps
    batch = {}
    
    # Map environment image names to policy camera format
    # Dataset has 'top' and 'wrist', policy expects camera1, camera2
    camera_mapping = {
        'top': 'observation.images.camera1',
        'wrist': 'observation.images.camera2',
    }
    
    # Handle images - stack from history
    for img_key, policy_key in camera_mapping.items():
        img_stack = []
        for hist_obs in list(observation_history)[-n_obs_steps:]:
            if img_key in hist_obs['images']:
                img_stack.append(hist_obs['images'][img_key])
        
        # Pad with first frame if not enough history
        while len(img_stack) < n_obs_steps:
            img_stack.insert(0, img_stack[0])
        
        # Stack: (n_obs_steps, C, H, W)
        img_tensor = torch.stack(img_stack)
        
        # Resize to policy expected size (256x256)
        if img_tensor.shape[2:] != (256, 256):
            img_tensor = F.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        
        batch[policy_key] = img_tensor.unsqueeze(0).to(device)  # Add batch dim
    
    # Handle state
    state_stack = []
    for hist_obs in list(observation_history)[-n_obs_steps:]:
        state_stack.append(hist_obs['state'])
    
    while len(state_stack) < n_obs_steps:
        state_stack.insert(0, state_stack[0])
    
    state_tensor = torch.stack(state_stack)
    batch['observation.state'] = state_tensor.unsqueeze(0).to(device)  # Add batch dim
    
    return batch


@torch.no_grad()
def get_action(policy, batch, task_instruction, device):
    """Get action from policy given preprocessed observation."""
    # Tokenize the task instruction using the policy's tokenizer
    # The tokenizer is accessed via processor.tokenizer
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokens = tokenizer(
        task_instruction,
        return_tensors="pt",
        padding=True,
        max_length=512,
        truncation=True
    )
    
    batch['observation.language.tokens'] = tokens['input_ids'].to(device)
    batch['observation.language.attention_mask'] = tokens['attention_mask'].bool().to(device)
    
    # Forward pass
    action = policy.select_action(batch)
    
    # Handle different output shapes
    # Could be (batch, chunk_size, action_dim) or (batch, action_dim) or (action_dim,)
    if action.dim() == 3:
        action = action[0, 0, :].cpu().numpy()
    elif action.dim() == 2:
        action = action[0, :].cpu().numpy()
    else:
        action = action.cpu().numpy()
    
    return action


def run_episode(robot, controller, camera, objects, table_height, policy, device, 
                episode_id, max_steps=500):
    """Run a single evaluation episode."""
    
    # Reset robot to home position
    robot.reset_to_home()
    sim_step(50)
    
    # Clear and spawn objects
    objects.clear_objects()
    spawned_ids = objects.spawn_graspable_objects(
        table_height=table_height,
        workspace_bounds=[[0.3, 0.7], [0.15, 0.5]],
        randomize=True
    )
    
    # Create container
    container_pos = [0.15, 0.15, table_height]
    objects.create_container_box(position=container_pos, size=[0.15, 0.15, 0.08])
    sim_step(100)
    
    # Select random object and instruction
    target_idx = np.random.randint(0, len(spawned_ids))
    task_instruction = OBJECT_INSTRUCTIONS[target_idx]
    target_object_id = spawned_ids[target_idx]
    
    object_names = ["red sphere", "green sphere", "blue cylinder", "red cylinder"]
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_id + 1}")
    print(f"Target: {object_names[target_idx]}")
    print(f"Task: {task_instruction}")
    print(f"{'='*60}")
    
    # Initialize observation history
    observation_history = deque(maxlen=policy.config.n_obs_steps)
    
    episode_success = False
    
    for step in range(max_steps):
        # Get current observation
        state = get_robot_state(robot)
        images = capture_images(camera)
        obs = {'state': state, 'images': images}
        
        # Preprocess observation
        batch = preprocess_obs(obs, policy, device, observation_history)
        
        # Get action from policy
        action = get_action(policy, batch, task_instruction, device)
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Handle action dimensions - model may output 6 or 7 dims
        if len(action) == 6:
            # Only joint commands, no gripper - add default gripper command
            # Use open gripper by default
            action = np.concatenate([action, [1.0]])
        elif len(action) < 6:
            # Pad to 7 if needed
            action = np.concatenate([action, np.zeros(7 - len(action))])
        
        # Execute action
        controller.process_action(action[:7])  # Use first 7 elements
        sim_step(4)
        
        # Check gripper command
        if action[6] < 0:
            robot.set_gripper(-1)
        else:
            robot.set_gripper(1)
        
        if step % 50 == 0:
            # Get object position for distance check
            obj_pos, _ = p.getBasePositionAndOrientation(target_object_id)
            ee_pos, _ = robot.get_end_effector_pose()
            distance = np.linalg.norm(np.array(obj_pos) - np.array(ee_pos))
            print(f"  Step {step:3d}/{max_steps} | Distance to object: {distance:.4f}")
    
    # Check success - object in container
    obj_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    distance_to_container = np.linalg.norm(np.array(obj_pos[:2]) - np.array(container_pos[:2]))
    episode_success = distance_to_container < 0.2
    
    print(f"\nEpisode Result: {'SUCCESS ✅' if episode_success else 'FAILED ❌'}")
    print(f"Object distance to container: {distance_to_container:.4f}")
    
    return {
        'episode_id': episode_id,
        'success': episode_success,
        'steps': step,
        'task': task_instruction,
        'target_object': object_names[target_idx],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy on UR5 grasping")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/smolvla_lerobot/final_model",
                        help="Path to checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable GUI visualization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load model
    policy = load_model(args.checkpoint_dir, device)
    
    # Initialize PyBullet
    print("\nInitializing simulation environment...")
    if args.no_gui:
        physics_client = p.connect(p.DIRECT)
    else:
        physics_client = p.connect(p.GUI)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    p.loadURDF("plane.urdf")
    
    # Create environment components (same as collect script)
    objects = ObjectsComponent()
    table_pos = [0.5, 0.5, 0.18]
    table_size = [0.5, 0.5, 0.18]
    table_id = objects.create_table(position=table_pos, size=table_size)
    table_aabb = p.getAABB(table_id)
    table_height = table_aabb[1][2]
    
    # Create robot
    robot_pos = [0.5, 0, table_height]
    robot_orn = p.getQuaternionFromEuler([0, 0, -np.pi/2])
    robot = UR5RobotComponent(position=robot_pos, orientation=robot_orn, use_fixed_base=True)
    robot.load()
    
    # Create controller
    controller = RobotController(
        robot_component=robot,
        control_mode="inverse_kinematics",
        ik_xyz_delta=0.05,
        ik_rpy_delta=0.05
    )
    
    # Create cameras
    camera = CameraComponent()
    camera.add_fixed_camera("top_view", eye_position=[0.5, 1.0, 1.0], target_position=[0.5, 0.5, 0.36])
    wrist_link_idx = robot.arm_joint_indices[-1] if robot.arm_joint_indices else 0
    camera.add_wrist_camera("wrist_view", robot_component=robot, link_idx=wrist_link_idx)
    
    print("✓ Environment initialized")
    
    # Run evaluation episodes
    print(f"\n{'='*60}")
    print(f"Starting evaluation with {args.num_episodes} episodes")
    print(f"{'='*60}")
    
    results = []
    
    try:
        for episode_id in range(args.num_episodes):
            result = run_episode(
                robot, controller, camera, objects, table_height,
                policy, device, episode_id,
                max_steps=args.max_steps
            )
            results.append(result)
    
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    
    finally:
        p.disconnect()
        print("\nEnvironment closed")
    
    # Print summary statistics
    if results:
        print(f"\n{'='*60}")
        print(f"Evaluation Summary ({len(results)} episodes)")
        print(f"{'='*60}")
        
        successes = [r['success'] for r in results]
        
        print(f"\nSuccess rate: {sum(successes)}/{len(successes)} ({100*sum(successes)/len(successes):.1f}%)")
        
        print(f"\nPer-episode results:")
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"  Episode {r['episode_id']+1}: {status} | {r['target_object']} | {r['task'][:50]}...")
        
        print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
