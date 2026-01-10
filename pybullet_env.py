"""
PyBullet UR5 Grasping Environment
Matches the specifications from wuc1/pybullet_ur5 dataset
"""

import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
import time


class UR5GraspEnv:
    """
    UR5 robot with Robotiq 85 gripper in PyBullet for grasping tasks.
    
    Observation Space:
        - 3 RGB camera images (480x640 each)
        - 8-dimensional state: [6 joint positions, 2 gripper positions]
    
    Action Space:
        - 7-dimensional: [6 joint velocities, 1 gripper command]
    """
    
    def __init__(self, gui=True, fps=30):
        self.gui = gui
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Physics client
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Camera parameters
        self.img_width = 640
        self.img_height = 480
        
        # Joint limits for UR5 (6 revolute joints)
        self.joint_lower_limits = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_upper_limits = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # Gripper limits (0 = closed, 0.085 = open)
        self.gripper_closed = 0.0
        self.gripper_open = 0.085
        
        # Robot and object IDs
        self.robot_id = None
        self.object_ids = []
        self.table_id = None
        
        # Joint indices
        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        
        # Wrist camera parameters (can be overridden externally)
        # Position camera to see gripper tips
        self.wrist_cam_offset = [-0.15, 0.05, 0.0]  # Slider values: X=35, Y=55, Z=50
        self.wrist_cam_target = [0.0, 0.35, 0.0]  # Slider values: X=50, Y=85, Z=50
        self.wrist_cam_yaw = 0.0  # Camera yaw angle in radians (rotate left/right)
        self.wrist_cam_pitch = 0.0  # Camera pitch angle in radians (tilt up/down)
        self.wrist_cam_roll = 0.0  # Camera roll angle in radians (rotation around view axis)
        
        self._setup_scene()
    
    def _setup_scene(self):
        """Initialize the simulation scene"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Create custom square table
        # Position table so robot will be at the middle of one edge
        # Robot will be at [0.5, 0, z], so table center should be at [0.5, 0.5, z]
        # This puts robot at the front edge center of the table
        table_pos = [0.5, 0.5, 0.18]  # Center position, half of table height
        table_size = [0.5, 0.5, 0.18]  # Square 1m x 1m, 36cm high
        
        # Create table collision and visual shapes
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_size)
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=table_size,
                                          rgbaColor=[0.6, 0.4, 0.2, 1])  # Brown wood color
        
        # Create table body
        self.table_id = p.createMultiBody(
            baseMass=0,  # Static table
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=table_pos
        )
        
        # Get table dimensions to properly place robot
        table_aabb = p.getAABB(self.table_id)
        table_height = table_aabb[1][2]  # Top surface Z coordinate
        
        print(f"Square table created: 1m x 1m, top at Z = {table_height:.4f}")
        print(f"Robot will be positioned at front edge center of table")
        
        # Load robot (using local UR5 with RG2 gripper)
        # Place robot base at the middle of the front edge
        robot_pos = [0.5, 0, table_height]  # At front edge center
        robot_orn = p.getQuaternionFromEuler([0, 0, -np.pi/2])  # -90 degree rotation around Z-axis
        
        # Load UR5 with RG2 gripper from local URDF
        urdf_path = str(Path(__file__).parent / "assets" / "robots" / "predefined" / "ur5" / "urdf" / "ur5_rg2.urdf")
        self.robot_id = p.loadURDF(urdf_path, robot_pos, robot_orn,
                                    useFixedBase=True,
                                    flags=p.URDF_USE_SELF_COLLISION)
        print("Loaded UR5 with RG2 gripper")
        
        # Get joint information
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(i)
        
        # Note: Default robot URDFs don't have gripper, we'll simulate it
        # For proper gripper, you'd need a robot + gripper URDF (e.g., UR5 + Robotiq)
        
        print(f"Robot loaded with {len(self.arm_joint_indices)} controllable joints")
        
        # Pad or truncate to 6 joints for consistency with dataset format
        if len(self.arm_joint_indices) > 6:
            print(f"  Using first 6 joints (robot has {len(self.arm_joint_indices)})")
            self.arm_joint_indices = self.arm_joint_indices[:6]
        elif len(self.arm_joint_indices) < 6:
            print(f"  Warning: Robot has only {len(self.arm_joint_indices)} joints, padding to 6")
        
        # Set initial robot pose (neutral position)
        self.reset_robot()
        
        # Spawn objects
        self._spawn_objects()
    
    def reset_robot(self):
        """Reset robot to initial pose"""
        # Home position (adjust based on robot type)
        num_joints = min(len(self.arm_joint_indices), 6)
        
        if num_joints >= 6:
            # Standard 6-DOF home position
            # Joint 5 adjusted to -90° (-π/2) from -180° (-π)
            home_positions = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]
        else:
            # Fallback for fewer joints
            home_positions = [0] * num_joints
        
        for i, joint_idx in enumerate(self.arm_joint_indices[:num_joints]):
            p.resetJointState(self.robot_id, joint_idx, home_positions[i])
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=home_positions[i],
                force=500
            )
    
    def _spawn_objects(self):
        """Spawn graspable objects on the table"""
        # Clear existing objects
        for obj_id in self.object_ids:
            p.removeBody(obj_id)
        self.object_ids = []
        
        # Get accurate table height for object placement
        table_aabb = p.getAABB(self.table_id)
        table_height = table_aabb[1][2]  # Top surface Z coordinate
        
        print(f"Table top surface at Z = {table_height:.4f}")
        
        # Place objects in the robot's workspace on the table
        # Table extends from Y=0 (front edge where robot is) to Y=1.0 (back edge)
        # Place objects in the center area of the table, within robot's reach
        object_configs = [
            # (shape, size, position, color)
            ("cube", 0.03, [0.5, 0.4, table_height + 0.015], [1, 0, 0, 1]),  # Red cube - center
            ("sphere", 0.025, [0.3, 0.3, table_height + 0.025], [0, 1, 0, 1]),  # Green sphere - left
            ("cylinder", (0.02, 0.05), [0.7, 0.5, table_height + 0.025], [0, 0, 1, 1]),  # Blue cylinder - right
        ]
        
        for config in object_configs:
            shape_type = config[0]
            
            if shape_type == "cube":
                size = config[1]
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3)
                vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, 
                                               rgbaColor=config[3])
            elif shape_type == "sphere":
                radius = config[1]
                col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                               rgbaColor=config[3])
            elif shape_type == "cylinder":
                radius, length = config[1]
                col_shape = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                  radius=radius, height=length)
                vis_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                               radius=radius, length=length,
                                               rgbaColor=config[3])
            
            obj_id = p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=config[2]
            )
            self.object_ids.append(obj_id)
    
    def get_camera_images(self):
        """
        Capture images from 3 cameras to match the dataset spec.
        Returns dict with keys matching the dataset:
        - observation.images.image_with_depth
        - observation.images.image
        - observation.images.hand_image
        """
        images = {}
        
        # Camera 1: Main view with depth
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 1.0, 1.0],
            cameraTargetPosition=[0.5, 0.5, 0.36],
            cameraUpVector=[0, 1, 0]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=self.img_width/self.img_height,
            nearVal=0.1, farVal=3.0
        )
        
        img_arr = p.getCameraImage(
            self.img_width, self.img_height,
            view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        rgb = np.array(img_arr[2]).reshape(self.img_height, self.img_width, 4)[:, :, :3]
        images['observation.images.image_with_depth'] = rgb
        
        # Camera 2: Top-down view over table
        view_matrix2 = p.computeViewMatrix(
            cameraEyePosition=[0.5, 1.0, 1.0],  # Directly above table center
            cameraTargetPosition=[0.5, 0.5, 0.36],  # Looking at table surface
            cameraUpVector=[0, 1, 0]  # Y-axis points up in the image
        )
        img_arr2 = p.getCameraImage(
            self.img_width, self.img_height,
            view_matrix2, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        rgb2 = np.array(img_arr2[2]).reshape(self.img_height, self.img_width, 4)[:, :, :3]
        images['observation.images.image'] = rgb2
        
        # Camera 3: Hand/wrist camera (attached to end-effector)
        # Use the last link of the arm
        ee_link_idx = self.arm_joint_indices[-1] if self.arm_joint_indices else 0
        ee_state = p.getLinkState(self.robot_id, ee_link_idx)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        
        # Camera offset from end-effector - use configurable parameters
        rot_matrix = p.getMatrixFromQuaternion(ee_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # Apply yaw, pitch rotations to camera offset
        offset = np.array(self.wrist_cam_offset)
        
        # Yaw rotation (around Z-axis)
        cos_yaw = np.cos(self.wrist_cam_yaw)
        sin_yaw = np.sin(self.wrist_cam_yaw)
        yaw_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Pitch rotation (around X-axis)
        cos_pitch = np.cos(self.wrist_cam_pitch)
        sin_pitch = np.sin(self.wrist_cam_pitch)
        pitch_matrix = np.array([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ])
        
        # Apply rotations: first pitch, then yaw
        rotated_offset = yaw_matrix @ pitch_matrix @ offset
        
        cam_pos = ee_pos + rot_matrix @ rotated_offset
        # Look at target position relative to end-effector (also apply yaw and pitch)
        target_offset = yaw_matrix @ pitch_matrix @ np.array(self.wrist_cam_target)
        target_pos = ee_pos + rot_matrix @ target_offset
        
        # Compute camera up vector with roll rotation
        # Start with default up vector in ee frame
        base_up = np.array([0, 0, 1])
        # Apply roll rotation around the view direction
        view_dir = target_pos - cam_pos
        view_dir = view_dir / np.linalg.norm(view_dir)
        # Rotate up vector around view direction by roll angle
        cos_roll = np.cos(self.wrist_cam_roll)
        sin_roll = np.sin(self.wrist_cam_roll)
        # Rodrigues' rotation formula
        up_vec = (base_up * cos_roll + 
                  np.cross(view_dir, base_up) * sin_roll + 
                  view_dir * np.dot(view_dir, base_up) * (1 - cos_roll))
        
        view_matrix3 = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vec.tolist()
        )
        img_arr3 = p.getCameraImage(
            self.img_width, self.img_height,
            view_matrix3, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        rgb3 = np.array(img_arr3[2]).reshape(self.img_height, self.img_width, 4)[:, :, :3]
        images['observation.images.hand_image'] = rgb3
        
        return images
    
    def get_state(self):
        """
        Get robot state (8-dim): [6 joint positions, 2 gripper positions]
        """
        joint_states = []
        num_joints = min(len(self.arm_joint_indices), 6)
        
        for joint_idx in self.arm_joint_indices[:num_joints]:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_states.append(joint_state[0])  # position
        
        # Pad to 6 joints if needed
        while len(joint_states) < 6:
            joint_states.append(0.0)
        
        # Add gripper state (simulated with last 2 values)
        # In real setup, this would be actual gripper joint positions
        gripper_state = [0.0, 0.0]  # Placeholder for gripper joints
        
        state = np.array(joint_states + gripper_state, dtype=np.float32)
        return state
    
    def step(self, action):
        """
        Execute action (7-dim): [6 joint velocities, 1 gripper command]
        
        Args:
            action: numpy array of shape (7,)
        
        Returns:
            observation: dict with 'images' and 'state'
            reward: float
            done: bool
            info: dict
        """
        # Apply joint velocity control
        num_joints = min(len(self.arm_joint_indices), 6)
        for i, joint_idx in enumerate(self.arm_joint_indices[:num_joints]):
            if i < len(action):
                p.setJointMotorControl2(
                    self.robot_id, joint_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=action[i],
                    force=500
                )
        
        # Apply gripper command (if we had a proper gripper)
        # gripper_cmd = action[6]  # -1 to 1, where -1=close, 1=open
        
        # Step simulation
        p.stepSimulation()
        if self.gui:
            time.sleep(self.dt)
        
        # Get new observation
        obs = {
            'images': self.get_camera_images(),
            'state': self.get_state()
        }
        
        # Compute reward (simple distance to nearest object)
        ee_link_idx = self.arm_joint_indices[-1] if self.arm_joint_indices else 0
        ee_state = p.getLinkState(self.robot_id, ee_link_idx)
        ee_pos = np.array(ee_state[0])
        
        min_dist = float('inf')
        for obj_id in self.object_ids:
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = np.linalg.norm(ee_pos - np.array(obj_pos))
            min_dist = min(min_dist, dist)
        
        reward = -min_dist  # Negative distance as reward
        done = False
        info = {'distance_to_object': min_dist}
        
        return obs, reward, done, info
    
    def reset(self):
        """Reset environment to initial state"""
        self.reset_robot()
        self._spawn_objects()
        
        # Let physics settle
        for _ in range(50):
            p.stepSimulation()
        
        obs = {
            'images': self.get_camera_images(),
            'state': self.get_state()
        }
        return obs
    
    def close(self):
        """Clean up"""
        p.disconnect()


if __name__ == "__main__":
    # Test the environment
    print("Testing UR5 Grasp Environment...")
    env = UR5GraspEnv(gui=True, fps=30)
    
    # Reset and get initial observation
    obs = env.reset()
    print(f"State shape: {obs['state'].shape}")
    print(f"Number of camera views: {len(obs['images'])}")
    for key, img in obs['images'].items():
        print(f"  {key}: {img.shape}")
    
    # Run a few random actions
    print("\nRunning random actions...")
    for i in range(100):
        action = np.random.randn(7) * 0.5  # Random small velocities
        obs, reward, done, info = env.step(action)
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.3f}, distance={info['distance_to_object']:.3f}")
    
    env.close()
    print("Test complete!")
