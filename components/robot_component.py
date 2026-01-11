"""
Robot Component - UR5 with RG2 Gripper

Handles robot loading, joint control, and state management.
Low-level robot interface - does not include control logic.
Use RobotController for control modes.
"""

import numpy as np
import pybullet as p
from pathlib import Path


class UR5RobotComponent:
    """UR5 robot with RG2 gripper component - low-level interface"""
    
    def __init__(self, position, orientation, urdf_path=None, use_fixed_base=True):
        """
        Initialize robot component.
        
        Args:
            position: [x, y, z] base position
            orientation: quaternion [x, y, z, w] or None for default
            urdf_path: Path to robot URDF file (uses default if None)
            use_fixed_base: Whether robot base is fixed
        """
        self.position = position
        self.orientation = orientation if orientation is not None else p.getQuaternionFromEuler([0, 0, -np.pi/2])
        self.use_fixed_base = use_fixed_base
        
        # Joint limits for UR5 (6 revolute joints)
        self.joint_lower_limits = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_upper_limits = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        self.joint_ranges = self.joint_upper_limits - self.joint_lower_limits
        
        # Gripper limits
        self.gripper_closed = 0.0
        self.gripper_open = 0.8  # Wide opening for grasping
        
        # Default URDF path
        if urdf_path is None:
            urdf_path = Path(__file__).parent.parent / "assets" / "robots" / "predefined" / "ur5" / "urdf" / "ur5_rg2.urdf"
        self.urdf_path = str(urdf_path)
        
        # Robot ID and joint indices (set during load)
        self.robot_id = None
        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        self.end_effector_link_idx = None
        
        # Home position
        self.home_positions = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]
    
    def load(self):
        """Load robot into PyBullet simulation"""
        self.robot_id = p.loadURDF(
            self.urdf_path,
            self.position,
            self.orientation,
            useFixedBase=self.use_fixed_base,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Get joint information
        num_joints = p.getNumJoints(self.robot_id)
        gripper_tip_candidates = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            
            # Look for gripper tip link (usually tool0, ee_link, or gripper_tip)
            if any(name in link_name.lower() for name in ['tool0', 'ee_link', 'tip', 'tcp']):
                gripper_tip_candidates.append((i, link_name))
            
            if joint_type == p.JOINT_REVOLUTE:
                # Check if it's a gripper joint
                if 'finger' in joint_name.lower() or 'gripper' in joint_name.lower():
                    self.gripper_joint_indices.append(i)
                    # Set high friction on gripper fingers for better grasping
                    p.changeDynamics(self.robot_id, i, lateralFriction=2.0, spinningFriction=0.1)
                else:
                    self.arm_joint_indices.append(i)
        
        # Ensure we have 6 arm joints
        if len(self.arm_joint_indices) > 6:
            print(f"  Robot has {len(self.arm_joint_indices)} joints, using first 6")
            self.arm_joint_indices = self.arm_joint_indices[:6]
        elif len(self.arm_joint_indices) < 6:
            print(f"  Warning: Robot has only {len(self.arm_joint_indices)} joints")
        
        # Set end effector link - prefer gripper tip if found, otherwise last arm joint
        if gripper_tip_candidates:
            self.end_effector_link_idx = gripper_tip_candidates[-1][0]  # Use last candidate
            print(f"  Using gripper tip link: {gripper_tip_candidates[-1][1]} (index {self.end_effector_link_idx})")
        else:
            self.end_effector_link_idx = self.arm_joint_indices[-1] if self.arm_joint_indices else 0
            print(f"  Using last arm joint as end effector (index {self.end_effector_link_idx})")
        
        print(f"Robot loaded: {len(self.arm_joint_indices)} arm joints, {len(self.gripper_joint_indices)} gripper joints")
        
        # Set initial pose
        self.reset_to_home()
        
        return self.robot_id
    
    def reset_to_home(self):
        """Reset robot to home position with gripper open"""
        num_joints = min(len(self.arm_joint_indices), len(self.home_positions))
        
        for i in range(num_joints):
            joint_idx = self.arm_joint_indices[i]
            p.resetJointState(self.robot_id, joint_idx, self.home_positions[i])
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=self.home_positions[i],
                force=500
            )
        
        # Set gripper to open position
        self.set_gripper(1)  # 1 = open
    
    def get_joint_states(self):
        """
        Get current joint states (positions and velocities).
        
        Returns:
            positions: array of joint positions
            velocities: array of joint velocities
        """
        positions = []
        velocities = []
        
        for joint_idx in self.arm_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
            velocities.append(joint_state[1])
        
        # Pad to 6 if needed
        while len(positions) < 6:
            positions.append(0.0)
            velocities.append(0.0)
        
        return np.array(positions, dtype=np.float32), np.array(velocities, dtype=np.float32)
    
    def get_gripper_state(self):
        """
        Get gripper joint positions.
        
        Returns:
            gripper_positions: array of gripper joint positions
        """
        gripper_positions = []
        
        for joint_idx in self.gripper_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            gripper_positions.append(joint_state[0])
        
        # If no gripper joints found, return placeholder
        if len(gripper_positions) == 0:
            gripper_positions = [0.0, 0.0]
        
        return np.array(gripper_positions, dtype=np.float32)
    
    def get_end_effector_pose(self):
        """
        Get end effector position and orientation.
        
        Returns:
            position: [x, y, z]
            orientation: quaternion [x, y, z, w]
        """
        if self.end_effector_link_idx is not None:
            ee_state = p.getLinkState(self.robot_id, self.end_effector_link_idx)
            return np.array(ee_state[0]), np.array(ee_state[1])
        return np.array([0, 0, 0]), np.array([0, 0, 0, 1])
    
    def set_joint_positions(self, positions, force=500):
        """
        Set target joint positions using position control.
        
        Args:
            positions: array of target joint positions (length <= 6)
            force: motor force
        """
        for i, joint_idx in enumerate(self.arm_joint_indices):
            if i < len(positions):
                p.setJointMotorControl2(
                    self.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=positions[i],
                    force=force
                )
    
    def set_joint_velocities(self, velocities, force=500):
        """
        Set target joint velocities using velocity control.
        
        Args:
            velocities: array of target joint velocities (length <= 6)
            force: motor force
        """
        for i, joint_idx in enumerate(self.arm_joint_indices):
            if i < len(velocities):
                p.setJointMotorControl2(
                    self.robot_id, joint_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=velocities[i],
                    force=force
                )
    
    def set_gripper(self, gripper_command, force=150):
        """
        Set gripper state.
        
        Args:
            gripper_command: -1 (close) to 1 (open), or specific position
            force: gripper motor force
        """
        # Map command to gripper position
        if gripper_command < 0:  # Close
            target_pos = self.gripper_closed
        else:  # Open - use maximum open position
            target_pos = self.gripper_open
        
        # Apply to all gripper joints with higher force for full opening
        for joint_idx in self.gripper_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=force,
                maxVelocity=5.0  # Increased velocity for faster closing
            )
    
    def get_state_vector(self):
        """
        Get full state vector: [6 joint positions + 2 gripper positions]
        
        Returns:
            state: 8-dimensional state vector
        """
        joint_positions, _ = self.get_joint_states()
        gripper_positions = self.get_gripper_state()
        
        # Ensure gripper has 2 values
        if len(gripper_positions) > 2:
            gripper_positions = gripper_positions[:2]
        elif len(gripper_positions) < 2:
            gripper_positions = np.pad(gripper_positions, (0, 2 - len(gripper_positions)))
        
        return np.concatenate([joint_positions[:6], gripper_positions[:2]])
