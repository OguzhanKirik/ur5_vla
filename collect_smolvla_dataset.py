"""
SmolVLA Dataset Collection Script for UR5 Grasping

Collects demonstration data with language instructions for SmolVLA training.
Each episode includes:
- Camera images (multiple views)
- Robot state (joint positions, gripper state)
- Actions (joint positions/velocities, gripper commands)
- Language instruction (natural language task description)

Usage:
    python collect_smolvla_dataset.py --episodes 100 --output-dir ./datasets/smolvla_ur5_grasp
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
from datetime import datetime
import pybullet as p
import pybullet_data
import time
import cv2

from components import UR5RobotComponent, CameraComponent, ObjectsComponent, RobotController


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Language instructions mapped to object indices
OBJECT_INSTRUCTIONS = {
    0: "Pick up the red box and place it in the container",
    1: "Grasp the green sphere and put it in the box",
    2: "Pick up the blue cylinder and drop it in the container",
    3: "Grab the blue box and place it in the container",
    "all": "Pick up all objects and place them in the container one by one"
}

# Alternative phrasings for data augmentation
INSTRUCTION_VARIANTS = {
    0: [
        "Pick up the red box and place it in the container",
        "Grasp the red cube and move it to the box",
        "Take the red box to the container",
        "Put the red cube in the container"
    ],
    1: [
        "Grasp the green sphere and put it in the box",
        "Pick up the green ball and place it in the container",
        "Take the green sphere to the box",
        "Move the green ball to the container"
    ],
    2: [
        "Pick up the blue cylinder and drop it in the container",
        "Grasp the blue cylinder and place it in the box",
        "Take the blue cylinder to the container",
        "Move the blue cylinder into the box"
    ],
    3: [
        "Grab the blue box and place it in the container",
        "Pick up the blue cube and put it in the box",
        "Take the blue box to the container",
        "Move the blue cube into the container"
    ]
}


class SmolVLADataCollector:
    """Collects data for SmolVLA training with language instructions"""
    
    def __init__(self, output_dir, dataset_name="ur5_grasp_smolvla"):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.dataset_dir = self.output_dir / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.dataset_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.videos_dir = self.dataset_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True)
        
        # Create separate folders for each video type
        self.top_rgb_dir = self.videos_dir / "top_rgb"
        self.top_rgb_dir.mkdir(exist_ok=True)
        
        self.wrist_rgb_dir = self.videos_dir / "wrist_rgb"
        self.wrist_rgb_dir.mkdir(exist_ok=True)
        
        self.top_depth_dir = self.videos_dir / "top_depth"
        self.top_depth_dir.mkdir(exist_ok=True)
        
        self.episodes = []
        self.episode_count = 0
        
        # Video recording
        self.video_writers = {}
        self.video_fps = 30
        
        # Metadata
        self.metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "robot_type": "ur5_rg2",
            "task_type": "pick_and_place",
            "fps": 30,
            "control_frequency": 240,
            "image_resolution": [480, 640],
            "num_cameras": 2,
            "camera_names": ["top", "wrist"],
            "videos_per_episode": 3,
            "video_types": ["top_rgb", "wrist_rgb", "top_depth"],
            "state_dim": 8,  # 6 joints + gripper_pos + gripper_state
            "action_dim": 7,  # 6 joints + gripper
            "objects": ["red box", "green sphere", "blue cylinder", "blue box"]
        }
        
    def start_episode(self, instruction, object_idx):
        """Start recording a new episode"""
        self.current_episode = {
            "episode_id": self.episode_count,
            "instruction": instruction,
            "object_idx": object_idx,
            "frames": [],
            "success": False,
            "start_time": time.time()
        }
        
        # Initialize video writers for this episode (3 videos only)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_filename = f"ep{self.episode_count:04d}.mp4"
        
        self.video_writers = {
            'top_rgb': cv2.VideoWriter(
                str(self.top_rgb_dir / video_filename),
                fourcc, self.video_fps, (640, 480)
            ),
            'wrist_rgb': cv2.VideoWriter(
                str(self.wrist_rgb_dir / video_filename),
                fourcc, self.video_fps, (640, 480)
            ),
            'top_depth': cv2.VideoWriter(
                str(self.top_depth_dir / video_filename),
                fourcc, self.video_fps, (640, 480)
            ),
        }
        
    def add_frame(self, state, action, images):
        """
        Add a frame to current episode
        
        Args:
            state: np.array of shape (8,) - joint positions + gripper
            action: np.array of shape (7,) - joint velocities + gripper command
            images: dict with keys 'top', 'wrist' containing dicts with 'rgb' and 'depth'
        """
        frame_idx = len(self.current_episode['frames'])
        
        # Write top RGB video
        if 'top' in images and 'rgb' in images['top']:
            rgb_img = images['top']['rgb']
            if isinstance(rgb_img, Image.Image):
                rgb_img = np.array(rgb_img)
            rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            self.video_writers['top_rgb'].write(rgb_bgr)
        
        # Write wrist RGB video
        if 'wrist' in images and 'rgb' in images['wrist']:
            rgb_img = images['wrist']['rgb']
            if isinstance(rgb_img, Image.Image):
                rgb_img = np.array(rgb_img)
            rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            self.video_writers['wrist_rgb'].write(rgb_bgr)
        
        # Write top depth video (normalized to 0-255 for visualization)
        if 'top' in images and 'depth' in images['top']:
            depth_img = images['top']['depth']
            if depth_img is not None:
                # Normalize depth to 0-255 range
                depth_norm = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8) * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                self.video_writers['top_depth'].write(depth_color)
        
        # Add frame data (just metadata, videos stored separately)
        frame = {
            "frame_idx": frame_idx,
            "timestamp": time.time() - self.current_episode['start_time'],
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
        }
        
        self.current_episode['frames'].append(frame)
    
    def end_episode(self, success=False):
        """Finish recording current episode"""
        self.current_episode['success'] = bool(success)  # Ensure Python bool, not numpy
        self.current_episode['duration'] = float(time.time() - self.current_episode['start_time'])
        self.current_episode['num_frames'] = int(len(self.current_episode['frames']))
        
        # Close video writers
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}
        
        # Add video filenames to episode data
        video_filename = f"ep{self.episode_count:04d}.mp4"
        self.current_episode['videos'] = {
            'top_rgb': f"videos/top_rgb/{video_filename}",
            'wrist_rgb': f"videos/wrist_rgb/{video_filename}",
            'top_depth': f"videos/top_depth/{video_filename}",
        }
        
        # Remove start_time (not JSON serializable)
        del self.current_episode['start_time']
        
        # Save episode
        episode_file = self.dataset_dir / f"episode_{self.episode_count:04d}.json"
        with open(episode_file, 'w') as f:
            json.dump(self.current_episode, f, indent=2, cls=NumpyEncoder)
        
        self.episodes.append({
            "episode_id": self.episode_count,
            "instruction": self.current_episode['instruction'],
            "object_idx": int(self.current_episode['object_idx']),
            "num_frames": self.current_episode['num_frames'],
            "success": bool(success)
        })
        
        self.episode_count += 1
        return self.episode_count - 1
    
    def save_metadata(self):
        """Save dataset metadata"""
        self.metadata['num_episodes'] = self.episode_count
        self.metadata['episodes'] = self.episodes
        
        metadata_file = self.dataset_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved metadata to {metadata_file}")


def collect_demonstration_episode(robot, controller, camera, objects, spawned_ids, object_names, 
                                   target_idx, container_pos, table_height, collector):
    """
    Collect one demonstration episode for a specific object
    
    Returns:
        success: bool indicating if grasp was successful
    """
    # Get language instruction
    instruction = np.random.choice(INSTRUCTION_VARIANTS[target_idx])
    
    # Start episode recording
    collector.start_episode(instruction, target_idx)
    
    print(f"\nInstruction: '{instruction}'")
    
    # Record initial state
    state = get_robot_state(robot)
    action = np.zeros(7)  # No action yet
    images = capture_images(camera)
    collector.add_frame(state, action, images)
    
    # Execute grasp and place (modified to collect data)
    try:
        success = collect_grasp_and_place(
            robot, controller, camera, spawned_ids[target_idx],
            container_pos, table_height, collector
        )
    except Exception as e:
        print(f"Error during grasp: {e}")
        success = False
    
    # End episode
    episode_id = collector.end_episode(success=success)
    
    return success


def get_robot_state(robot):
    """Get current robot state as numpy array"""
    # Get joint positions
    joint_positions = []
    for joint_idx in robot.arm_joint_indices:
        joint_state = p.getJointState(robot.robot_id, joint_idx)
        joint_positions.append(joint_state[0])
    
    # Get gripper state
    if robot.gripper_joint_indices:
        gripper_pos = p.getJointState(robot.robot_id, robot.gripper_joint_indices[0])[0]
        gripper_state = 1.0 if gripper_pos > 0.4 else -1.0  # Open or closed
    else:
        gripper_pos = 0.0
        gripper_state = 0.0
    
    state = np.array(joint_positions + [gripper_pos, gripper_state], dtype=np.float32)
    return state


def capture_images(camera):
    """Capture RGB and depth images from top and wrist cameras only"""
    captured_data = camera.capture_all()  # Returns dict of camera_name -> RGB array
    images = {}
    
    # Map camera names and get both RGB and depth
    for cam_name, rgb_array in captured_data.items():
        if rgb_array is not None:
            # Get depth data from camera
            cam_data = camera.cameras.get(cam_name, {})
            depth_array = cam_data.get('depth', None)
            
            # Map camera names - only keep top and wrist
            target_name = None
            if 'main' in cam_name.lower() or 'top' in cam_name.lower():
                target_name = 'top'
            elif 'wrist' in cam_name.lower():
                target_name = 'wrist'
            
            if target_name:
                images[target_name] = {
                    'rgb': rgb_array,
                    'depth': depth_array
                }
    
    # Ensure both cameras present - use first available as fallback
    if images:
        fallback_data = list(images.values())[0]
        if 'top' not in images:
            images['top'] = fallback_data
        if 'wrist' not in images:
            images['wrist'] = fallback_data
    else:
        # No cameras captured - create dummy images
        print("Warning: No camera data captured, using dummy images")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.float32)
        images['top'] = {'rgb': dummy_rgb, 'depth': dummy_depth}
        images['wrist'] = {'rgb': dummy_rgb, 'depth': dummy_depth}
    
    return images


def collect_grasp_and_place(robot, controller, camera, target_object_id, container_pos, 
                            table_height, collector, distance_threshold=0.02):
    """
    Modified grasp_and_place_object that collects data at each step.
    Uses exact same logic as grasp_objects.py for reliable grasping.
    """
    # Get target object position
    target_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    target_pos = np.array(target_pos)
    
    # Phase 1: Move to position 20cm above object
    approach_pos = target_pos.copy()
    approach_pos[2] += 0.20  # 20cm above object
    
    num_steps = 200
    for step in range(num_steps):
        current_pos, current_orn = robot.get_end_effector_pose()
        
        delta = approach_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [1]])  # gripper open
        controller.process_action(action)
        
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        if step % 10 == 0:
            camera.capture_all()
        
        # Collect data every 8 steps for 30 FPS
        if step % 8 == 0:
            state = get_robot_state(robot)
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Phase 2: Descend to grasp position (1cm above object)
    grasp_pos = target_pos.copy()
    grasp_pos[2] += 0.01  # 1cm above object
    
    for step in range(num_steps):
        current_pos, current_orn = robot.get_end_effector_pose()
        delta = grasp_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [1]])
        controller.process_action(action)
        
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        if step % 10 == 0:
            camera.capture_all()
        
        if step % 8 == 0:
            state = get_robot_state(robot)
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Close gripper with adaptive torque detection
    robot.set_gripper(-1)
    
    max_close_steps = 1000
    torque_threshold = 2.5
    min_steps = 60
    
    for close_step in range(max_close_steps):
        p.stepSimulation()
        time.sleep(1./240.)
        
        if close_step >= min_steps:
            gripper_torques = []
            for joint_idx in robot.gripper_joint_indices:
                joint_state = p.getJointState(robot.robot_id, joint_idx)
                gripper_torques.append(abs(joint_state[3]))
            
            max_torque = max(gripper_torques)
            
            if max_torque > torque_threshold:
                # Stop gripper
                for joint_idx in robot.gripper_joint_indices:
                    current_pos = p.getJointState(robot.robot_id, joint_idx)[0]
                    p.setJointMotorControl2(
                        robot.robot_id, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=current_pos,
                        force=150,
                        maxVelocity=0
                    )
                
                for _ in range(10):
                    p.stepSimulation()
                    time.sleep(1./240.)
                break
        
        # Record data
        if close_step % 8 == 0:
            state = get_robot_state(robot)
            action = np.concatenate([[0, 0, 0, 0, 0, 0], [-1]])
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Save gripper positions for holding during carry
    grasp_positions = []
    for joint_idx in robot.gripper_joint_indices:
        grasp_positions.append(p.getJointState(robot.robot_id, joint_idx)[0])
    
    # Lift object
    current_pos, _ = robot.get_end_effector_pose()
    lift_pos = current_pos.copy()
    lift_pos[2] += 0.15  # 15cm up
    
    for step in range(100):
        current_pos, _ = robot.get_end_effector_pose()
        delta = lift_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 1.5, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [-1]])
        controller.process_action(action)
        
        # Maintain grip
        for i, joint_idx in enumerate(robot.gripper_joint_indices):
            if i < len(grasp_positions):
                p.setJointMotorControl2(
                    robot.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=grasp_positions[i],
                    force=300,
                    maxVelocity=0
                )
        
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        if step % 8 == 0:
            state = get_robot_state(robot)
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Move to container
    drop_pos = np.array([container_pos[0], container_pos[1], table_height + 0.15])
    
    for step in range(200):
        current_pos, _ = robot.get_end_effector_pose()
        delta = drop_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            break
        
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 1.5, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [-1]])
        controller.process_action(action)
        
        # Maintain grip
        for i, joint_idx in enumerate(robot.gripper_joint_indices):
            if i < len(grasp_positions):
                p.setJointMotorControl2(
                    robot.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=grasp_positions[i],
                    force=300,
                    maxVelocity=0
                )
        
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        if step % 8 == 0:
            state = get_robot_state(robot)
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Wait 1 second
    for wait_step in range(240):
        for i, joint_idx in enumerate(robot.gripper_joint_indices):
            if i < len(grasp_positions):
                p.setJointMotorControl2(
                    robot.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=grasp_positions[i],
                    force=300,
                    maxVelocity=0
                )
        p.stepSimulation()
        time.sleep(1./240.)
        
        if wait_step % 8 == 0:
            state = get_robot_state(robot)
            action = np.concatenate([[0, 0, 0, 0, 0, 0], [-1]])
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Open gripper wider for release
    for joint_idx in robot.gripper_joint_indices:
        p.setJointMotorControl2(
            robot.robot_id, joint_idx,
            p.POSITION_CONTROL,
            targetPosition=1.0,  # Open wider
            force=200,
            maxVelocity=15.0  # Fast opening
        )
    for step in range(30):
        p.stepSimulation()
        time.sleep(1./240.)
        
        if step % 8 == 0:
            state = get_robot_state(robot)
            action = np.concatenate([[0, 0, 0, 0, 0, 0], [1]])
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Let settle
    for step in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
        
        if step % 8 == 0:
            state = get_robot_state(robot)
            action = np.concatenate([[0, 0, 0, 0, 0, 0], [1]])
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Return to home position smoothly
    home_ee_pos = np.array([0.5, 0.4, table_height + 0.3])
    
    for step in range(300):
        current_pos, _ = robot.get_end_effector_pose()
        delta = home_ee_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < 0.02:
            break
        
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 2.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [1]])  # gripper open
        controller.process_action(action)
        
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        if step % 8 == 0:
            state = get_robot_state(robot)
            images = capture_images(camera)
            collector.add_frame(state, action, images)
    
    # Check if object reached container (success condition)
    obj_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    success = np.linalg.norm(np.array(obj_pos[:2]) - np.array(container_pos[:2])) < 0.2
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Collect SmolVLA dataset for UR5 grasping")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--output-dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="smolvla_ur5_grasp", help="Dataset name")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    
    args = parser.parse_args()
    
    print("="*70)
    print("SmolVLA Dataset Collection for UR5 Grasping")
    print("="*70)
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output_dir}/{args.dataset_name}")
    print(f"GUI: {'Enabled' if args.gui else 'Disabled'}")
    print("="*70)
    
    # Initialize data collector
    collector = SmolVLADataCollector(args.output_dir, args.dataset_name)
    
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
    
    # Create cameras (only top view and wrist)
    camera = CameraComponent()
    camera.add_fixed_camera("top_view", eye_position=[0.5, 1.0, 1.0], target_position=[0.5, 0.5, 0.36])
    wrist_link_idx = robot.arm_joint_indices[-1] if robot.arm_joint_indices else 0
    camera.add_wrist_camera("wrist_view", robot_component=robot, link_idx=wrist_link_idx)
    
    # Container
    container_pos = [0.15, 0.15, table_height]
    objects.create_container_box(position=container_pos, size=[0.15, 0.15, 0.08])
    
    object_names = ["red box", "green sphere", "blue cylinder", "blue box"]
    
    # Collect episodes
    successful_episodes = 0
    
    for episode_idx in tqdm(range(args.episodes), desc="Collecting episodes"):
        # Clear previous objects if not first episode
        if episode_idx > 0:
            objects.clear_objects()
        
        # Reset environment with randomized objects
        # Robot is at [0.5, 0, table_height] facing -Y direction
        # Workspace bounds define reachable area: X [0.3, 0.7], Y [0.15, 0.5]
        spawned_ids = objects.spawn_graspable_objects(
            table_height=table_height,
            workspace_bounds=[[0.3, 0.7], [0.15, 0.5]],
            randomize=True
        )
        
        # Recreate container for each episode
        container_pos = [0.15, 0.15, table_height]
        objects.create_container_box(position=container_pos, size=[0.15, 0.15, 0.08])
        
        # Let physics settle
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Select random object to grasp
        target_idx = np.random.randint(0, len(spawned_ids))
        
        # Collect demonstration
        success = collect_demonstration_episode(
            robot, controller, camera, objects, spawned_ids, object_names,
            target_idx, container_pos, table_height, collector
        )
        
        if success:
            successful_episodes += 1
        
        print(f"Episode {episode_idx}: {object_names[target_idx]} - {'SUCCESS' if success else 'FAILED'}")
    
    # Save metadata
    collector.save_metadata()
    
    # Print summary
    print("\n" + "="*70)
    print("Collection Complete!")
    print("="*70)
    print(f"Total episodes: {args.episodes}")
    print(f"Successful: {successful_episodes} ({100*successful_episodes/args.episodes:.1f}%)")
    print(f"Dataset location: {collector.dataset_dir}")
    print("="*70)
    
    p.disconnect()


if __name__ == "__main__":
    main()
