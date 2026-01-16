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
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType


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
    
    # Extend policy config to match training (7D actions, 8D state)
    # Training extended the policy but checkpoint only saved weights, not modified config
    from lerobot.configs.types import PolicyFeature, FeatureType
    
    current_action_dim = policy.config.action_feature.shape[0]
    current_state_dim = policy.config.input_features['observation.state'].shape[0]
    
    if current_action_dim != 7:
        print(f"Extending action feature from {current_action_dim}D to 7D...")
        # action_feature is a property, so modify output_features instead
        new_action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        policy.config.output_features['action'] = new_action_feature
        policy.config.max_action_dim = max(policy.config.max_action_dim, 7)
    
    if current_state_dim != 8:
        print(f"Extending state feature from {current_state_dim}D to 8D...")
        new_state_feature = PolicyFeature(type=FeatureType.STATE, shape=(8,))
        policy.config.input_features['observation.state'] = new_state_feature
        policy.config.max_state_dim = max(policy.config.max_state_dim, 8)
    
    policy.eval()
    policy.to(device)
    
    print(f"Policy loaded: {sum(p.numel() for p in policy.parameters()):,} parameters")
    print(f"  Action feature: {policy.config.action_feature.shape}")
    print(f"  State feature: {policy.config.input_features['observation.state'].shape}")
    
    return policy
def images_to_tensors(images: dict) -> dict:
    """Convert uint8 images to torch tensors in [0, 1] and channel-first."""
    tensor_images = {}
    for key, value in images.items():
        img = torch.from_numpy(value.astype(np.float32)) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)
        tensor_images[key] = img
    return tensor_images


def preprocess_obs(obs, policy, observation_history):
    """
    Preprocess observation for the policy.
    Matches exactly what the training preprocessor does.
    
    obs: dict with keys 'images' (dict of numpy arrays), 'state' (np.array)
    observation_history: deque storing past observations
    """
    # Match dataset loading: float in [0, 1], channel-first.
    normalized_images = images_to_tensors(obs["images"])
    state = obs["state"].copy()
    
    observation_history.append({
        'images': normalized_images,
        'state': torch.from_numpy(state).float(),
    })
    
    # Build batch with observation history
    n_obs_steps = policy.config.n_obs_steps
    batch = {}
    
    # Use DATASET feature names (not camera1/2/3 - preprocessor will handle those)
    image_feature_names = ['observation.images.top', 'observation.images.wrist']
    image_keys = ['top', 'wrist']
    
    # Handle images - stack from history using dataset names
    for img_key, feat_name in zip(image_keys, image_feature_names):
        img_stack = []
        for hist_obs in list(observation_history)[-n_obs_steps:]:
            if img_key in hist_obs['images']:
                img_stack.append(hist_obs['images'][img_key])
        
        # Pad with first frame if not enough history
        while len(img_stack) < n_obs_steps:
            img_stack.insert(0, img_stack[0])
        
        # Stack: (n_obs_steps, C, H, W)
        img_tensor = torch.stack(img_stack)
        batch[feat_name] = img_tensor.unsqueeze(0)
    
    # Handle state - keep full 8D and stack from history
    state_stack = []
    for hist_obs in list(observation_history)[-n_obs_steps:]:
        state_stack.append(hist_obs['state'])
    
    while len(state_stack) < n_obs_steps:
        state_stack.insert(0, state_stack[0])
    
    state_tensor = torch.stack(state_stack)
    batch['observation.state'] = state_tensor.unsqueeze(0)
    
    return batch



def run_episode(robot, controller, camera, objects, table_height, policy, device,
                episode_id, preprocessor, postprocessor, max_steps=500):
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
    
    # Select green sphere (index 1) for testing
    target_idx = 1  # 0=red sphere, 1=green sphere, 2=blue cylinder, 3=red cylinder
    task_instruction = OBJECT_INSTRUCTIONS[target_idx]
    target_object_id = spawned_ids[target_idx]
    
    object_names = ["red sphere", "green sphere", "blue cylinder", "red cylinder"]
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_id + 1}")
    print(f"Target: {object_names[target_idx]}")
    print(f"Task: {task_instruction}")
    print(f"{'='*60}")
    
    # DEBUG: Print initial positions
    obj_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    ee_pos, _ = robot.get_end_effector_pose()
    robot_state = get_robot_state(robot)
    print(f"\nInitial state:")
    print(f"  Robot joint positions: {robot_state[:6]}")
    print(f"  End-effector position: {ee_pos}")
    print(f"  Target (green sphere) position: {obj_pos}")
    print(f"  Distance to target: {np.linalg.norm(np.array(obj_pos) - np.array(ee_pos)):.4f}")
    print()
    
    # Initialize observation history
    observation_history = deque(maxlen=policy.config.n_obs_steps)
    
    # Pre-fill observation history with initial state (critical for context!)
    initial_state = get_robot_state(robot)
    initial_images = capture_images(camera)
    initial_obs = {'state': initial_state, 'images': initial_images}
    
    # Fill history with the same initial observation multiple times
    # This gives the model temporal context about the starting state
    for _ in range(policy.config.n_obs_steps):
        normalized_images = images_to_tensors(initial_images)
        
        observation_history.append({
            'images': normalized_images,
            'state': torch.from_numpy(initial_state).float(),
        })
    
    print(f"Observation history pre-filled with {len(observation_history)} frames")
    
    # Print current state for debugging
    print(f"Initial robot state: {initial_state}")
    print(f"  Joints: {initial_state[:6]}")
    print(f"  Gripper: {initial_state[6:]}") 
    
    episode_success = False
    
    # Collect initial observations to build up observation history (2 steps)
    print("\nCollecting initial observations...")
    for obs_step in range(2):
        state = get_robot_state(robot)
        images = capture_images(camera)
        obs = {'state': state, 'images': images}
        
        # Preprocess and add to history
        normalized_images = images_to_tensors(obs["images"])
        
        observation_history.append({
            'images': normalized_images,
            'state': torch.from_numpy(obs['state']).float(),
        })
        
        sim_step(4)
    
    print(f"Observation history built with {len(observation_history)} frames")
    
    # DEBUG: Save first observation images to verify they're correct
    if episode_id == 0:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(initial_images['top'])
        axes[0].set_title('Top Camera View')
        axes[0].axis('off')
        axes[1].imshow(initial_images['wrist'])
        axes[1].set_title('Wrist Camera View')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig('debug_first_observation.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved first observation to debug_first_observation.png\n")
    
    print("Starting robot control...\n")
    
    for step in range(max_steps):
        # Get current observation
        state = get_robot_state(robot)
        images = capture_images(camera)
        obs = {'state': state, 'images': images}
        
        # Preprocess observation and create batch with correct feature names
        batch = preprocess_obs(obs, policy, observation_history)
        
        # Apply preprocessor pipeline to prepare data for model:
        # 1. RenameObservationsProcessorStep - leaves names as-is (rename_map is empty)
        # 2. AddBatchDimensionProcessorStep - ensures batch dimension exists
        # 3. SmolVLANewLineProcessor - formats task instructions for tokenizer
        # 4. TokenizerProcessorStep - tokenizes language instructions
        # 5. DeviceProcessorStep - moves tensors to policy.config.device
        # 6. NormalizerProcessorStep - applies IDENTITY normalization for VISUAL (keeps [0,1]),
        #                              and MEAN_STD normalization for STATE/ACTION
        batch['task'] = [task_instruction]
        batch = preprocessor(batch)
        
        # Rename images from dataset names (top/wrist) to policy names (camera1/2)
        # This matches the training loop behavior.
        if 'observation.images.top' in batch:
            batch['observation.images.camera1'] = batch.pop('observation.images.top')
        if 'observation.images.wrist' in batch:
            batch['observation.images.camera2'] = batch.pop('observation.images.wrist')
        if 'observation.images.top_is_pad' in batch:
            batch['observation.images.camera1_is_pad'] = batch.pop('observation.images.top_is_pad')
        if 'observation.images.wrist_is_pad' in batch:
            batch['observation.images.camera2_is_pad'] = batch.pop('observation.images.wrist_is_pad')
        
        # Move batch to runtime device (mirrors training)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get single action from policy
        # Policy's select_action() manages action queue internally!
        # It calls _get_action_chunk() only when queue is empty,
        # stores the chunk internally, and returns one action at a time
        action = policy.select_action(batch)
        
        # CRITICAL: DO use postprocessor to denormalize actions!
        # Model outputs normalized actions in [-1, 1]
        # Postprocessor denormalizes them to actual joint space using dataset stats
        action = postprocessor(action)
        
        action = action.cpu().numpy() if hasattr(action, 'cpu') else action
        action = action.squeeze()  # Remove singleton dimensions
        
        # DEBUG: Print first action and batch info
        if step == 0:
            print(f"\n🔍 First action (denormalized): {action}")
            print(f"   Action shape: {action.shape}")
            print(f"   Action range: [{action.min():.4f}, {action.max():.4f}]")
            print(f"   Expected: denormalized joint angles/deltas")
            print(f"\n📷 Image inputs:")
            if 'observation.images.camera1' in batch:
                img1 = batch['observation.images.camera1']
                print(f"   camera1 shape: {img1.shape}, range: [{img1.min():.3f}, {img1.max():.3f}]")
            if 'observation.images.camera2' in batch:
                img2 = batch['observation.images.camera2']
                print(f"   camera2 shape: {img2.shape}, range: [{img2.min():.3f}, {img2.max():.3f}]")
            print(f"\n📝 Task: {task_instruction}")
            print(f"   Batch keys: {list(batch.keys())}")
            if 'input_ids' in batch:
                print(f"   Tokenized task length: {batch['input_ids'].shape if hasattr(batch['input_ids'], 'shape') else len(batch['input_ids'])} tokens")
            else:
                print(f"   ⚠️ WARNING: No 'input_ids' in batch! Task not tokenized!")
            print()
        
        # Handle action dimensions - model may output 6 or 7 dims
        if len(action) == 6:
            # Only joint commands, no gripper - add default gripper command
            action = np.concatenate([action, [1.0]])
        elif len(action) < 6:
            # Pad to 7 if needed
            action = np.concatenate([action, np.zeros(7 - len(action))])
        
        # Clip joint angles to safe range (denormalized actions in radians)
        # Most UR5 joints: -2π to 2π
        action[:6] = np.clip(action[:6], -6.28, 6.28)
        action[6] = np.clip(action[6], -1.0, 1.0)  # Gripper: [-1, 1]
        
        # DEBUG: Print action every 50 steps
        if step % 50 == 0:
            print(f"  Step {step:3d}: joints={action[:6]} | gripper={action[6]:.2f}")
        
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
            print(f"  Distance to target: {distance:.4f}")
    
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
    parser.add_argument("--repo-id", type=str, default="local/ur5_smolvla_grasp",
                        help="Dataset repo ID (for metadata/stats)")
    parser.add_argument("--root", type=str, default="../../datasets_/lerobot",
                        help="Dataset root directory")
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
    
    # Load dataset metadata for preprocessing stats (CRITICAL!)
    print(f"\nLoading dataset metadata: {args.repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    
    # Create preprocessor/postprocessor with dataset stats (same as training!)
    print("Creating preprocessor with dataset statistics...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, 
        dataset_stats=dataset_metadata.stats
    )
    print("✓ Preprocessor ready")
    
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
                policy, device, episode_id, preprocessor, postprocessor,
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
