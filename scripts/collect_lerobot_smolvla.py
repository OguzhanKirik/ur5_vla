"""
LeRobot-Compatible SmolVLA Dataset Collection Script for UR5 Grasping

Collects demonstration data using LeRobot API for fast training with SmolVLA.
Each episode includes:
- Camera images (multiple views as video)
- Robot state (joint positions, gripper state)
- Actions (joint positions/velocities, gripper commands)
- Language instruction (natural language task description)

Usage:
    python collect_lerobot_smolvla.py --episodes 100 --repo-id local/ur5_smolvla_grasp
"""

import sys
from pathlib import Path

# Add parent directory to path to import components
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse
from tqdm import tqdm
import pybullet as p
import pybullet_data
import time
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from components import UR5RobotComponent, CameraComponent, ObjectsComponent, RobotController


# Language instructions mapped to object indices
OBJECT_INSTRUCTIONS = {
    0: "Pick up the red sphere and place it in the container",
    1: "Grasp the green sphere and put it in the box",
    2: "Pick up the blue cylinder and drop it in the container",
    3: "Pick up the red cylinder and place it in the container",
}

# Alternative phrasings for data augmentation
INSTRUCTION_VARIANTS = {
    0: [
        "Pick up the red sphere and place it in the container",
        "Grasp the red ball and move it to the box",
        "Take the red sphere to the container",
        "Put the red sphere in the container"
    ],
    1: [
        "Grasp the green sphere and put it in the box",
        "Pick up the green ball and place it in the container",
        "Take the green sphere to the box",
        "Move the green sphere to the container"
    ],
    2: [
        "Pick up the blue cylinder and drop it in the container",
        "Grasp the blue cylinder and place it in the box",
        "Take the blue cylinder to the container",
        "Move the blue cylinder into the box"
    ],
    3: [
        "Pick up the red cylinder and place it in the container",
        "Grasp the red cylinder and move it to the box",
        "Take the red cylinder to the container",
        "Put the red cylinder in the container"
    ]
}

# Global flag for fast simulation
FAST_SIM = False


def sim_step(n: int = 1) -> None:
    """Perform `n` physics steps."""
    for _ in range(n):
        p.stepSimulation()
        if not FAST_SIM:
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
    """Capture RGB images from top and wrist cameras as PIL Images"""
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
                # Convert to PIL Image for LeRobot
                images[target_name] = Image.fromarray(rgb_array.astype(np.uint8))
    
    # Ensure both cameras present
    if images:
        fallback = list(images.values())[0]
        if 'top' not in images:
            images['top'] = fallback
        if 'wrist' not in images:
            images['wrist'] = fallback
    else:
        dummy = Image.new('RGB', (640, 480), color=(128, 128, 128))
        images['top'] = dummy
        images['wrist'] = dummy
    
    return images


def create_lerobot_features() -> dict:
    """Create feature specification for LeRobot dataset compatible with SmolVLA"""
    return {
        # Robot state: 6 joints + gripper_pos + gripper_state
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_pos", "gripper_state"]
        },
        # Top camera (RGB video)
        "observation.images.top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            }
        },
        # Wrist camera (RGB video)
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            }
        },
        # Action: 6 joint velocities + gripper command
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_0_vel", "joint_1_vel", "joint_2_vel", "joint_3_vel", "joint_4_vel", "joint_5_vel", "gripper"]
        },
    }


def collect_grasp_episode(robot, controller, camera, target_object_id, container_pos, 
                          table_height, dataset, instruction, distance_threshold=0.02):
    """
    Collect one grasp-and-place episode using LeRobot API.
    Returns success status.
    """
    target_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    target_pos = np.array(target_pos)
    
    frame_count = 0
    
    def add_frame(action):
        """Add a frame to the dataset"""
        nonlocal frame_count
        state = get_robot_state(robot)
        images = capture_images(camera)
        
        frame = {
            "observation.state": state,
            "observation.images.top": images['top'],
            "observation.images.wrist": images['wrist'],
            "action": action.astype(np.float32),
            "task": instruction,
        }
        dataset.add_frame(frame)
        frame_count += 1
    
    # Phase 1: Move above object (20cm)
    approach_pos = target_pos.copy()
    approach_pos[2] += 0.20
    
    for step in range(200):
        current_pos, _ = robot.get_end_effector_pose()
        delta = approach_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            action_pos = np.clip((delta / distance) * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [1]])
        controller.process_action(action)
        sim_step(4)
        
        if step % 8 == 0:
            add_frame(action)
    
    # Phase 2: Descend to grasp (1cm above)
    grasp_pos = target_pos.copy()
    grasp_pos[2] += 0.01
    
    for step in range(200):
        current_pos, _ = robot.get_end_effector_pose()
        delta = grasp_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            action_pos = np.clip((delta / distance) * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [1]])
        controller.process_action(action)
        sim_step(4)
        
        if step % 8 == 0:
            add_frame(action)
    
    # Phase 3: Close gripper
    robot.set_gripper(-1)
    
    for close_step in range(1000):
        sim_step(1)
        
        if close_step >= 60:
            gripper_torques = []
            for joint_idx in robot.gripper_joint_indices:
                joint_state = p.getJointState(robot.robot_id, joint_idx)
                gripper_torques.append(abs(joint_state[3]))
            
            if max(gripper_torques) > 2.5:
                for joint_idx in robot.gripper_joint_indices:
                    current_pos = p.getJointState(robot.robot_id, joint_idx)[0]
                    p.setJointMotorControl2(
                        robot.robot_id, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=current_pos,
                        force=150, maxVelocity=0
                    )
                sim_step(10)
                break
        
        if close_step % 8 == 0:
            action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            add_frame(action)
    
    # Save gripper positions
    grasp_positions = [p.getJointState(robot.robot_id, j)[0] for j in robot.gripper_joint_indices]
    
    # Phase 4: Lift
    current_pos, _ = robot.get_end_effector_pose()
    lift_pos = current_pos.copy()
    lift_pos[2] += 0.15
    
    for step in range(100):
        current_pos, _ = robot.get_end_effector_pose()
        delta = lift_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            action_pos = np.clip((delta / distance) * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        # Maintain grip
        for i, joint_idx in enumerate(robot.gripper_joint_indices):
            p.setJointMotorControl2(
                robot.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=grasp_positions[i],
                force=150, maxVelocity=0
            )
        
        action = np.concatenate([action_pos, [0, 0, 0], [-1]])
        controller.process_action(action)
        sim_step(4)
        
        if step % 8 == 0:
            add_frame(action)
    
    # Phase 5: Move to container
    place_pos = np.array([container_pos[0], container_pos[1], container_pos[2] + 0.15])
    
    for step in range(200):
        current_pos, _ = robot.get_end_effector_pose()
        delta = place_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            action_pos = np.clip((delta / distance) * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        for i, joint_idx in enumerate(robot.gripper_joint_indices):
            p.setJointMotorControl2(
                robot.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=grasp_positions[i],
                force=150, maxVelocity=0
            )
        
        action = np.concatenate([action_pos, [0, 0, 0], [-1]])
        controller.process_action(action)
        sim_step(4)
        
        if step % 8 == 0:
            add_frame(action)
    
    # Phase 6: Release
    robot.set_gripper(1)
    
    for step in range(50):
        sim_step(4)
        
        if step % 8 == 0:
            action = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            add_frame(action)
    
    # Phase 7: Settle
    for step in range(100):
        sim_step(1)
        
        if step % 8 == 0:
            action = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            add_frame(action)
    
    # Phase 8: Return home
    home_pos = np.array([0.5, 0.4, table_height + 0.3])
    
    for step in range(300):
        current_pos, _ = robot.get_end_effector_pose()
        delta = home_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < 0.02:
            break
        
        if distance > 0.001:
            action_pos = np.clip((delta / distance) * 2.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [1]])
        controller.process_action(action)
        sim_step(4)
        
        if step % 8 == 0:
            add_frame(action)
    
    # Check success
    obj_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    success = np.linalg.norm(np.array(obj_pos[:2]) - np.array(container_pos[:2])) < 0.2
    
    return success, frame_count


def main():
    parser = argparse.ArgumentParser(description="Collect LeRobot-compatible SmolVLA dataset")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--repo-id", type=str, default="local/ur5_smolvla_grasp", help="Dataset repo ID")
    parser.add_argument("--root", type=str, default="./datasets/lerobot", help="Root directory for dataset")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--fast", action="store_true", help="Fast mode (no sleeps)")
    parser.add_argument("--save-only-success", action="store_true", help="Only save successful episodes")
    
    args = parser.parse_args()
    
    print("="*70)
    print("LeRobot SmolVLA Dataset Collection for UR5 Grasping")
    print("="*70)
    print(f"Episodes: {args.episodes}")
    print(f"Repo ID: {args.repo_id}")
    print(f"Root: {args.root}")
    print("="*70)
    
    global FAST_SIM
    FAST_SIM = bool(args.fast and not args.gui)
    
    # Initialize PyBullet
    if args.gui:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    p.loadURDF("plane.urdf")
    
    # Create environment
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
    
    # Container
    container_pos = [0.15, 0.15, table_height]
    objects.create_container_box(position=container_pos, size=[0.15, 0.15, 0.08])
    
    object_names = ["red sphere", "green sphere", "blue cylinder", "red cylinder"]
    
    # Create LeRobot dataset
    # LeRobotDataset will create: root/repo_id directory
    # So we need to remove that specific directory if it exists
    root_path = Path(args.root)
    
    # Remove existing dataset directory (root/repo_id)
    dataset_path = root_path / args.repo_id
    if dataset_path.exists():
        import shutil
        print(f"Removing existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)
    
    features = create_lerobot_features()
    
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=30,
        features=features,
        root=root_path,
        robot_type="ur5_rg2",
        use_videos=True,
        image_writer_threads=4,
    )
    
    # Collect episodes
    successful_episodes = 0
    total_frames = 0
    
    for episode_idx in tqdm(range(args.episodes), desc="Collecting episodes"):
        # Clear previous objects if not first episode
        if episode_idx > 0:
            objects.clear_objects()
        
        # IMPORTANT: Reset robot to home position at the start of each episode
        robot.reset_to_home()
        sim_step(50)  # Let robot settle at home position
        
        # Reset environment
        spawned_ids = objects.spawn_graspable_objects(
            table_height=table_height,
            workspace_bounds=[[0.3, 0.7], [0.15, 0.5]],
            randomize=True
        )
        
        container_pos = [0.15, 0.15, table_height]
        objects.create_container_box(position=container_pos, size=[0.15, 0.15, 0.08])
        
        sim_step(100)
        
        # Select random object
        target_idx = np.random.randint(0, len(spawned_ids))
        instruction = np.random.choice(INSTRUCTION_VARIANTS[target_idx])
        
        # Collect episode
        success, frame_count = collect_grasp_episode(
            robot, controller, camera, spawned_ids[target_idx],
            container_pos, table_height, dataset, instruction
        )
        
        # Save or discard episode
        if args.save_only_success and not success:
            # Clear episode buffer without saving
            dataset.clear_episode_buffer()
            print(f"Episode {episode_idx}: {object_names[target_idx]} - FAILED (discarded)")
        else:
            dataset.save_episode()
            total_frames += frame_count
            if success:
                successful_episodes += 1
            print(f"Episode {episode_idx}: {object_names[target_idx]} - {'SUCCESS' if success else 'FAILED'} ({frame_count} frames)")
    
    # Finalize dataset
    dataset.stop_image_writer()
    dataset.finalize()
    
    # Print summary
    print("\n" + "="*70)
    print("Collection Complete!")
    print("="*70)
    print(f"Total attempted: {args.episodes}")
    print(f"Successful: {successful_episodes}")
    print(f"Saved episodes: {dataset.num_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Dataset location: {dataset.root}")
    print(f"Repo ID: {args.repo_id}")
    print("="*70)
    print("\nTo train with this dataset:")
    print(f"  python train_smolvla.py --dataset-dir {dataset.root}")
    print("="*70)
    
    p.disconnect()


if __name__ == "__main__":
    main()
