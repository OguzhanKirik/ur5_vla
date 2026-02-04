#!/usr/bin/env python

"""
Metrics evaluation script for SmolVLA training progress.

Runs N episodes on a checkpoint (headless, fixed seed) and outputs quantitative
metrics that can be compared across training steps to verify the model is learning.

Usage:
    python eval_metrics.py --checkpoint-dir ./checkpoints/smolvla_full/checkpoint-10000
    python eval_metrics.py --checkpoint-dir ./checkpoints/smolvla_full/checkpoint-10000 --num-episodes 20
"""

import os
import sys

# Suppress warnings and set environment variables BEFORE any imports
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
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
import pybullet as p
import pybullet_data
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from components import UR5RobotComponent, CameraComponent, ObjectsComponent, RobotController
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors

# Training objects and instructions (same as data collection)
OBJECT_INSTRUCTIONS = {
    0: "Pick up the red sphere and place it in the container",
    1: "Grasp the green sphere and put it in the box",
    2: "Pick up the blue cylinder and drop it in the container",
    3: "Pick up the red cylinder and place it in the container",
}

INSTRUCTION_VARIANTS = {
    0: [
        "Pick up the red sphere and place it in the container",
        "Grasp the red ball and move it to the box",
        "Take the red sphere to the container",
        "Put the red sphere in the container",
    ],
    1: [
        "Grasp the green sphere and put it in the box",
        "Pick up the green ball and place it in the container",
        "Take the green sphere to the box",
        "Move the green sphere to the container",
    ],
    2: [
        "Pick up the blue cylinder and drop it in the container",
        "Grasp the blue cylinder and place it in the box",
        "Take the blue cylinder to the container",
        "Move the blue cylinder into the box",
    ],
    3: [
        "Pick up the red cylinder and place it in the container",
        "Grasp the red cylinder and move it to the box",
        "Take the red cylinder to the container",
        "Put the red cylinder in the container",
    ],
}

OBJECT_NAMES = ["red sphere", "green sphere", "blue cylinder", "red cylinder"]


# ---------------------------------------------------------------------------
# Reused helpers from eval_smolvla_lerobot.py
# ---------------------------------------------------------------------------

def sim_step(n: int = 1) -> None:
    for _ in range(n):
        p.stepSimulation()


def get_robot_state(robot) -> np.ndarray:
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


def images_to_tensors(images: dict) -> dict:
    tensor_images = {}
    for key, value in images.items():
        img = torch.from_numpy(value.astype(np.float32)) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)
        tensor_images[key] = img
    return tensor_images


def preprocess_obs(obs, policy, observation_history):
    normalized_images = images_to_tensors(obs["images"])
    state = obs["state"].copy()
    observation_history.append({
        'images': normalized_images,
        'state': torch.from_numpy(state).float(),
    })
    n_obs_steps = policy.config.n_obs_steps
    batch = {}
    image_feature_names = ['observation.images.top', 'observation.images.wrist']
    image_keys = ['top', 'wrist']
    for img_key, feat_name in zip(image_keys, image_feature_names):
        img_stack = []
        for hist_obs in list(observation_history)[-n_obs_steps:]:
            if img_key in hist_obs['images']:
                img_stack.append(hist_obs['images'][img_key])
        while len(img_stack) < n_obs_steps:
            img_stack.insert(0, img_stack[0])
        img_tensor = torch.stack(img_stack)
        batch[feat_name] = img_tensor.unsqueeze(0)
    state_stack = []
    for hist_obs in list(observation_history)[-n_obs_steps:]:
        state_stack.append(hist_obs['state'])
    while len(state_stack) < n_obs_steps:
        state_stack.insert(0, state_stack[0])
    state_tensor = torch.stack(state_stack)
    batch['observation.state'] = state_tensor.unsqueeze(0)
    return batch


def load_model(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    print(f"Loading model from: {checkpoint_dir}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    from lerobot.configs.types import PolicyFeature, FeatureType
    current_action_dim = policy.config.action_feature.shape[0]
    current_state_dim = policy.config.input_features['observation.state'].shape[0]
    if current_action_dim != 7:
        new_action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        policy.config.output_features['action'] = new_action_feature
        policy.config.max_action_dim = max(policy.config.max_action_dim, 7)
    if current_state_dim != 8:
        new_state_feature = PolicyFeature(type=FeatureType.STATE, shape=(8,))
        policy.config.input_features['observation.state'] = new_state_feature
        policy.config.max_state_dim = max(policy.config.max_state_dim, 8)
    if "observation.images.camera3" in policy.config.input_features:
        del policy.config.input_features["observation.images.camera3"]
    policy.eval()
    policy.to(device)
    print(f"Policy loaded: {sum(pp.numel() for pp in policy.parameters()):,} parameters")
    return policy


# ---------------------------------------------------------------------------
# New: metrics episode runner
# ---------------------------------------------------------------------------

def run_metrics_episode(robot, controller, camera, objects, table_height,
                        policy, device, preprocessor, postprocessor,
                        episode_id, max_steps=500, n_action_steps=10,
                        sim_steps_per_action=32, container_pos=None):
    """Run a single episode and collect per-step metrics."""

    if container_pos is None:
        container_pos = [0.15, 0.15, table_height]

    policy.reset()
    policy.config.n_action_steps = n_action_steps

    # Reset robot
    robot.reset_to_home()
    sim_step(50)

    # Clear and spawn objects
    objects.clear_objects()
    spawned_ids = objects.spawn_graspable_objects(
        table_height=table_height,
        workspace_bounds=[[0.3, 0.7], [0.15, 0.5]],
        randomize=True
    )
    objects.create_container_box(position=container_pos, size=[0.15, 0.15, 0.08])
    sim_step(100)

    # Select random target
    target_idx = np.random.randint(0, len(spawned_ids))
    variants = INSTRUCTION_VARIANTS.get(target_idx, [OBJECT_INSTRUCTIONS[target_idx]])
    task_instruction = np.random.choice(variants)
    target_object_id = spawned_ids[target_idx]
    target_name = OBJECT_NAMES[target_idx]

    # Initialize observation history
    observation_history = deque(maxlen=policy.config.n_obs_steps)
    initial_state = get_robot_state(robot)
    initial_images = capture_images(camera)
    for _ in range(policy.config.n_obs_steps):
        observation_history.append({
            'images': images_to_tensors(initial_images),
            'state': torch.from_numpy(initial_state).float(),
        })

    # Warm-up observations
    for _ in range(2):
        state = get_robot_state(robot)
        imgs = capture_images(camera)
        observation_history.append({
            'images': images_to_tensors(imgs),
            'state': torch.from_numpy(state).float(),
        })
        sim_step(sim_steps_per_action)

    # Per-step tracking
    step_ee_positions = []
    step_distances = []
    step_gripper_values = []
    step_action_magnitudes = []
    step_obj_positions = []

    for step in range(max_steps):
        state = get_robot_state(robot)
        images = capture_images(camera)
        obs = {'state': state, 'images': images}

        batch = preprocess_obs(obs, policy, observation_history)
        batch['task'] = [task_instruction]
        batch = preprocessor(batch)

        # Rename image keys to match policy expectations
        if 'observation.images.top' in batch:
            batch['observation.images.camera1'] = batch.pop('observation.images.top')
        if 'observation.images.wrist' in batch:
            batch['observation.images.camera2'] = batch.pop('observation.images.wrist')
        if 'observation.images.top_is_pad' in batch:
            batch['observation.images.camera1_is_pad'] = batch.pop('observation.images.top_is_pad')
        if 'observation.images.wrist_is_pad' in batch:
            batch['observation.images.camera2_is_pad'] = batch.pop('observation.images.wrist_is_pad')

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        action = policy.select_action(batch)
        action = postprocessor(action)
        action = action.cpu().numpy() if hasattr(action, 'cpu') else action
        action = action.squeeze()

        # Pad / clip action
        if len(action) == 6:
            action = np.concatenate([action, [1.0]])
        elif len(action) < 6:
            action = np.concatenate([action, np.zeros(7 - len(action))])

        action[:3] = np.clip(action[:3], -1.0, 1.0)
        action[3:6] = np.clip(action[3:6], -1.0, 1.0)
        action[6] = np.clip(action[6], -1.0, 1.0)

        # Record metrics BEFORE executing action so gripper value is the command
        ee_pos, _ = robot.get_end_effector_pose()
        obj_pos, _ = p.getBasePositionAndOrientation(target_object_id)
        ee_pos = np.array(ee_pos)
        obj_pos = np.array(obj_pos)

        step_ee_positions.append(ee_pos.copy())
        step_distances.append(float(np.linalg.norm(ee_pos - obj_pos)))
        step_gripper_values.append(float(action[6]))
        step_action_magnitudes.append(float(np.linalg.norm(action[:6])))
        step_obj_positions.append(obj_pos.copy())

        # Execute action
        controller.process_action(action[:7])
        sim_step(sim_steps_per_action)

        if action[6] < 0:
            robot.set_gripper(-1)
        else:
            robot.set_gripper(1)

    # Final object position
    final_obj_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    final_obj_pos = np.array(final_obj_pos)

    return {
        'target_name': target_name,
        'target_idx': target_idx,
        'task_instruction': task_instruction,
        'container_pos': container_pos,
        'table_height': table_height,
        'final_obj_pos': final_obj_pos.tolist(),
        'ee_positions': [pos.tolist() for pos in step_ee_positions],
        'distances': step_distances,
        'gripper_values': step_gripper_values,
        'action_magnitudes': step_action_magnitudes,
        'obj_positions': [pos.tolist() for pos in step_obj_positions],
    }


def compute_episode_metrics(episode_data):
    """Derive per-episode metrics from raw step data."""
    distances = episode_data['distances']
    gripper_values = episode_data['gripper_values']
    action_magnitudes = episode_data['action_magnitudes']
    ee_positions = [np.array(pos) for pos in episode_data['ee_positions']]
    obj_positions = [np.array(pos) for pos in episode_data['obj_positions']]
    table_height = episode_data['table_height']
    container_pos = np.array(episode_data['container_pos'])
    final_obj_pos = np.array(episode_data['final_obj_pos'])

    min_dist = float(np.min(distances))
    approach_success = min_dist < 0.05

    # Gripper attempted: gripper value goes negative at any step
    grasp_attempted = any(g < 0 for g in gripper_values)

    # Gripper at object: gripper < 0 while distance < 0.05
    grasp_at_object = any(
        g < 0 and d < 0.05
        for g, d in zip(gripper_values, distances)
    )

    # Object lifted: object z > table_height + 0.03 at any step
    object_lifted = any(pos[2] > table_height + 0.03 for pos in obj_positions)

    # Object in container: final object XY distance to container < 0.2
    object_in_container = float(np.linalg.norm(final_obj_pos[:2] - container_pos[:2])) < 0.2

    # Gripper switch step: first step where gripper sign flips from + to -
    gripper_switch_step = None
    for i in range(1, len(gripper_values)):
        if gripper_values[i - 1] >= 0 and gripper_values[i] < 0:
            gripper_switch_step = i
            break

    # Mean action magnitude
    mean_action_mag = float(np.mean(action_magnitudes))

    # Path length: sum of EE displacements
    path_length = 0.0
    for i in range(1, len(ee_positions)):
        path_length += float(np.linalg.norm(ee_positions[i] - ee_positions[i - 1]))

    return {
        'target_name': episode_data['target_name'],
        'min_dist_to_target': min_dist,
        'approach_success': approach_success,
        'grasp_attempted': grasp_attempted,
        'grasp_at_object': grasp_at_object,
        'object_lifted': object_lifted,
        'object_in_container': object_in_container,
        'gripper_switch_step': gripper_switch_step,
        'mean_action_magnitude': mean_action_mag,
        'path_length': path_length,
    }


def print_summary(checkpoint_name, episode_metrics, num_episodes, seed):
    """Print formatted summary table to terminal."""
    print(f"\nEval: {checkpoint_name} | {num_episodes} episodes | seed {seed}\n")

    # Per-episode table
    hdr = f"  {'Ep':>3}  {'Target':<16} {'MinDist':>7}  {'Approach':>8}  {'GripTry':>7}  {'GripObj':>7}  {'Lifted':>6}  {'Placed':>6}  {'GripStep':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    yn = lambda b: "yes" if b else "no"
    for i, m in enumerate(episode_metrics):
        gs = str(m['gripper_switch_step']) if m['gripper_switch_step'] is not None else "-"
        row = (
            f"  {i+1:3d}  {m['target_name']:<16} {m['min_dist_to_target']:7.3f}  "
            f"{yn(m['approach_success']):>8}  "
            f"{yn(m['grasp_attempted']):>7}  "
            f"{yn(m['grasp_at_object']):>7}  "
            f"{yn(m['object_lifted']):>6}  "
            f"{yn(m['object_in_container']):>6}  "
            f"{gs:>8}"
        )
        print(row)

    # Aggregate
    n = len(episode_metrics)
    min_dists = [m['min_dist_to_target'] for m in episode_metrics]
    approach_ct = sum(m['approach_success'] for m in episode_metrics)
    grip_try_ct = sum(m['grasp_attempted'] for m in episode_metrics)
    grip_obj_ct = sum(m['grasp_at_object'] for m in episode_metrics)
    lifted_ct = sum(m['object_lifted'] for m in episode_metrics)
    placed_ct = sum(m['object_in_container'] for m in episode_metrics)
    action_mags = [m['mean_action_magnitude'] for m in episode_metrics]
    path_lens = [m['path_length'] for m in episode_metrics]

    print(f"\nAggregate ({n} episodes):")
    print(f"  Min distance to target:    {np.mean(min_dists):.3f} +/- {np.std(min_dists):.3f}")
    print(f"  Approach success (<5cm):   {approach_ct}/{n} ({100*approach_ct/n:.1f}%)")
    print(f"  Gripper attempted:         {grip_try_ct}/{n} ({100*grip_try_ct/n:.1f}%)")
    print(f"  Gripper at object:         {grip_obj_ct}/{n} ({100*grip_obj_ct/n:.1f}%)")
    print(f"  Object lifted:             {lifted_ct}/{n} ({100*lifted_ct/n:.1f}%)")
    print(f"  Place success:             {placed_ct}/{n} ({100*placed_ct/n:.1f}%)")
    print(f"  Mean action magnitude:     {np.mean(action_mags):.2f} +/- {np.std(action_mags):.2f}")
    print(f"  Mean path length:          {np.mean(path_lens):.2f} +/- {np.std(path_lens):.2f}")


def save_metrics(checkpoint_dir, episode_metrics, aggregate, seed, num_episodes):
    """Write all metrics to JSON in the checkpoint directory."""
    out_path = Path(checkpoint_dir) / "eval_metrics.json"
    payload = {
        'checkpoint': str(checkpoint_dir),
        'seed': seed,
        'num_episodes': num_episodes,
        'episodes': episode_metrics,
        'aggregate': aggregate,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Metrics evaluation for SmolVLA training progress")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--repo-id", type=str, default="local/ur5_smolvla_grasp",
                        help="Dataset repo ID (for metadata/stats)")
    parser.add_argument("--root", type=str, default="./datasets/lerobot",
                        help="Dataset root directory")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--action-steps", type=int, default=5,
                        help="Number of actions to execute per policy inference")
    parser.add_argument("--sim-steps-per-action", type=int, default=32,
                        help="Physics steps per control step")

    args = parser.parse_args()

    # Fixed seed for fair comparison across checkpoints
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Resolve dataset root
    root_path = Path(args.root)
    if not (root_path / "meta" / "info.json").exists():
        alt_root = Path(__file__).resolve().parent / "datasets" / "lerobot"
        if (alt_root / "meta" / "info.json").exists():
            args.root = str(alt_root)

    # Load model
    policy = load_model(args.checkpoint_dir, device)

    # Dataset metadata for preprocessor stats
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats,
    )

    # Initialize PyBullet (always headless)
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1. / 240.)
    p.loadURDF("plane.urdf")

    # Environment
    objects_comp = ObjectsComponent()
    table_pos = [0.5, 0.5, 0.18]
    table_size = [0.5, 0.5, 0.18]
    table_id = objects_comp.create_table(position=table_pos, size=table_size)
    table_aabb = p.getAABB(table_id)
    table_height = table_aabb[1][2]

    robot_pos = [0.5, 0, table_height]
    robot_orn = p.getQuaternionFromEuler([0, 0, -np.pi / 2])
    robot = UR5RobotComponent(position=robot_pos, orientation=robot_orn, use_fixed_base=True)
    robot.load()

    controller = RobotController(
        robot_component=robot,
        control_mode="inverse_kinematics",
        ik_xyz_delta=0.08,
        ik_rpy_delta=0.05,
    )

    camera = CameraComponent()
    camera.add_fixed_camera("top_view", eye_position=[0.5, 1.0, 1.0], target_position=[0.5, 0.5, 0.36])
    wrist_link_idx = robot.arm_joint_indices[-1] if robot.arm_joint_indices else 0
    camera.add_wrist_camera("wrist_view", robot_component=robot, link_idx=wrist_link_idx)

    container_pos = [0.15, 0.15, table_height]

    checkpoint_name = Path(args.checkpoint_dir).name

    # Run episodes
    episode_metrics = []
    try:
        for ep in range(args.num_episodes):
            print(f"Running episode {ep + 1}/{args.num_episodes} ...", end=" ", flush=True)
            raw = run_metrics_episode(
                robot, controller, camera, objects_comp, table_height,
                policy, device, preprocessor, postprocessor,
                episode_id=ep,
                max_steps=args.max_steps,
                n_action_steps=args.action_steps,
                sim_steps_per_action=args.sim_steps_per_action,
                container_pos=container_pos,
            )
            metrics = compute_episode_metrics(raw)
            episode_metrics.append(metrics)
            print(f"min_dist={metrics['min_dist_to_target']:.3f}  "
                  f"approach={'Y' if metrics['approach_success'] else 'N'}  "
                  f"grip={'Y' if metrics['grasp_attempted'] else 'N'}  "
                  f"lift={'Y' if metrics['object_lifted'] else 'N'}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        p.disconnect()

    if not episode_metrics:
        print("No episodes completed.")
        return

    # Aggregate statistics
    n = len(episode_metrics)
    min_dists = [m['min_dist_to_target'] for m in episode_metrics]
    action_mags = [m['mean_action_magnitude'] for m in episode_metrics]
    path_lens = [m['path_length'] for m in episode_metrics]

    aggregate = {
        'min_dist_to_target_mean': float(np.mean(min_dists)),
        'min_dist_to_target_std': float(np.std(min_dists)),
        'approach_success_rate': sum(m['approach_success'] for m in episode_metrics) / n,
        'grasp_attempted_rate': sum(m['grasp_attempted'] for m in episode_metrics) / n,
        'grasp_at_object_rate': sum(m['grasp_at_object'] for m in episode_metrics) / n,
        'object_lifted_rate': sum(m['object_lifted'] for m in episode_metrics) / n,
        'place_success_rate': sum(m['object_in_container'] for m in episode_metrics) / n,
        'mean_action_magnitude_mean': float(np.mean(action_mags)),
        'mean_action_magnitude_std': float(np.std(action_mags)),
        'path_length_mean': float(np.mean(path_lens)),
        'path_length_std': float(np.std(path_lens)),
    }

    # Print summary
    print_summary(checkpoint_name, episode_metrics, n, args.seed)

    # Save JSON
    save_metrics(args.checkpoint_dir, episode_metrics, aggregate, args.seed, n)


if __name__ == "__main__":
    main()
