"""
Camera Component - Multiple camera views

Handles camera configuration and image capture.
"""

import numpy as np
import pybullet as p


class CameraComponent:
    """Multi-camera setup for observations"""
    
    def __init__(self, width=640, height=480, renderer='auto'):
        """
        Initialize camera component.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            renderer: 'auto', 'hardware', or 'tiny'
        """
        self.width = width
        self.height = height
        
        # Select renderer
        if renderer == 'auto':
            # Try hardware, fallback to tiny
            try:
                p.getCameraImage(32, 32, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                self.renderer = p.ER_BULLET_HARDWARE_OPENGL
            except:
                self.renderer = p.ER_TINY_RENDERER
        elif renderer == 'hardware':
            self.renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            self.renderer = p.ER_TINY_RENDERER
        
        # Projection matrix (shared by all cameras)
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=3.0
        )
        
        # Camera configurations
        self.cameras = {}
        
        # Wrist camera adjustable parameters
        self.wrist_cam_offset = [-0.15, 0.05, 0.0]
        self.wrist_cam_target = [0.0, 0.35, 0.0]
        self.wrist_cam_yaw = 0.0
        self.wrist_cam_pitch = 0.0
        self.wrist_cam_roll = 0.0
    
    def add_fixed_camera(self, name, eye_position, target_position, up_vector=[0, 0, 1]):
        """
        Add a fixed camera with static position.
        
        Args:
            name: Camera identifier
            eye_position: [x, y, z] camera position
            target_position: [x, y, z] look-at target
            up_vector: [x, y, z] up direction
        """
        self.cameras[name] = {
            'type': 'fixed',
            'eye_position': eye_position,
            'target_position': target_position,
            'up_vector': up_vector
        }
    
    def add_wrist_camera(self, name, robot_component, link_idx=None):
        """
        Add a camera attached to robot's end effector.
        
        Args:
            name: Camera identifier
            robot_component: RobotComponent instance
            link_idx: Link index to attach to (uses end effector if None)
        """
        self.cameras[name] = {
            'type': 'wrist',
            'robot': robot_component,
            'link_idx': link_idx if link_idx is not None else robot_component.end_effector_link_idx
        }
    
    def capture_image(self, name):
        """
        Capture image from specific camera.
        
        Args:
            name: Camera identifier
            
        Returns:
            rgb_array: (H, W, 3) RGB image
        """
        if name not in self.cameras:
            raise ValueError(f"Camera '{name}' not found")
        
        camera = self.cameras[name]
        
        if camera['type'] == 'fixed':
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera['eye_position'],
                cameraTargetPosition=camera['target_position'],
                cameraUpVector=camera['up_vector']
            )
        elif camera['type'] == 'wrist':
            view_matrix = self._compute_wrist_view_matrix(camera)
        else:
            raise ValueError(f"Unknown camera type: {camera['type']}")
        
        # Capture image
        img_arr = p.getCameraImage(
            self.width, self.height,
            view_matrix, self.proj_matrix,
            renderer=self.renderer
        )
        
        # Extract RGB (ignore alpha channel)
        rgb = np.array(img_arr[2]).reshape(self.height, self.width, 4)[:, :, :3]
        return rgb
    
    def _compute_wrist_view_matrix(self, camera_config):
        """Compute view matrix for wrist-mounted camera"""
        robot = camera_config['robot']
        link_idx = camera_config['link_idx']
        
        # Get end effector pose
        ee_state = p.getLinkState(robot.robot_id, link_idx)
        ee_pos = np.array(ee_state[0])
        ee_orn = ee_state[1]
        
        # Get rotation matrix from end effector orientation
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
        
        # Apply rotations to offset
        rotated_offset = yaw_matrix @ pitch_matrix @ offset
        cam_pos = ee_pos + rot_matrix @ rotated_offset
        
        # Compute target position
        target_offset = yaw_matrix @ pitch_matrix @ np.array(self.wrist_cam_target)
        target_pos = ee_pos + rot_matrix @ target_offset
        
        # Compute up vector with roll
        base_up = np.array([0, 0, 1])
        view_dir = target_pos - cam_pos
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
        
        # Apply roll rotation using Rodrigues' formula
        cos_roll = np.cos(self.wrist_cam_roll)
        sin_roll = np.sin(self.wrist_cam_roll)
        up_vec = (base_up * cos_roll + 
                  np.cross(view_dir, base_up) * sin_roll + 
                  view_dir * np.dot(view_dir, base_up) * (1 - cos_roll))
        
        return p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=up_vec.tolist()
        )
    
    def capture_all(self):
        """
        Capture images from all cameras.
        
        Returns:
            images: dict mapping camera names to RGB arrays
        """
        images = {}
        for name in self.cameras:
            images[name] = self.capture_image(name)
        return images
    
    def set_wrist_camera_params(self, offset=None, target=None, yaw=None, pitch=None, roll=None):
        """
        Update wrist camera parameters.
        
        Args:
            offset: [x, y, z] camera offset from end effector
            target: [x, y, z] target offset from end effector
            yaw: rotation around Z-axis (radians)
            pitch: rotation around X-axis (radians)
            roll: rotation around view axis (radians)
        """
        if offset is not None:
            self.wrist_cam_offset = offset
        if target is not None:
            self.wrist_cam_target = target
        if yaw is not None:
            self.wrist_cam_yaw = yaw
        if pitch is not None:
            self.wrist_cam_pitch = pitch
        if roll is not None:
            self.wrist_cam_roll = roll
