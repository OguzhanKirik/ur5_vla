"""
Robot Controller Component - Control mode implementations

Handles different control strategies for the robot.
Separates control logic from low-level robot operations.
"""

import numpy as np
import pybullet as p
from typing import Union


CONTROL_MODES = [
    "inverse_kinematics",  # Control via IK (Cartesian space)
    "joint_positions",      # Control via joint angles
    "joint_velocities",     # Control via joint velocities
    "joint_target"         # Control with target deviation
]


class RobotController:
    """Controller for robot with multiple control modes"""
    
    def __init__(self, robot_component, control_mode: Union[int, str] = "joint_velocities",
                 use_physics_sim=True, sim_step=1/30.0,
                 ik_xyz_delta=0.015, ik_rpy_delta=0.015, joint_target_delta=0.5):
        """
        Initialize robot controller.
        
        Args:
            robot_component: UR5RobotComponent instance
            control_mode: Control mode (0-3 or string name from CONTROL_MODES)
            use_physics_sim: Use physics simulation vs teleportation
            sim_step: Simulation time step
            ik_xyz_delta: Max IK position change per step (mode 0)
            ik_rpy_delta: Max IK rotation change per step (mode 0)
            joint_target_delta: Max joint deviation from target (mode 3)
        """
        self.robot = robot_component
        
        # Control mode
        if isinstance(control_mode, str):
            assert control_mode in CONTROL_MODES, f"Unknown control mode: {control_mode}"
            self.control_mode = CONTROL_MODES.index(control_mode)
        else:
            assert 0 <= control_mode < len(CONTROL_MODES), f"Control mode must be 0-{len(CONTROL_MODES)-1}"
            self.control_mode = control_mode
        
        self.use_physics_sim = use_physics_sim
        self.sim_step = sim_step
        
        # Mode-specific parameters
        self.ik_xyz_delta = ik_xyz_delta
        self.ik_rpy_delta = ik_rpy_delta
        self.joint_target_delta = joint_target_delta
        
        # For control mode 3
        self.control_target = []
    
    def process_action(self, action: np.ndarray, current_ee_pos=None, current_ee_orn=None):
        """
        Process action based on control mode.
        
        Args:
            action: Action vector (-1 to 1 range)
            current_ee_pos: Current end effector position (for mode 0, optional)
            current_ee_orn: Current end effector orientation (for mode 0, optional)
            
        Control Modes:
            0 - inverse_kinematics: action is [dx, dy, dz, droll, dpitch, dyaw]
            1 - joint_positions: action is desired joint angles (normalized)
            2 - joint_velocities: action is joint velocities (normalized)
            3 - joint_target: action is deviation from control_target
        """
        if self.control_mode == 0:
            self._process_ik_action(action, current_ee_pos, current_ee_orn)
        elif self.control_mode == 1:
            self._process_joint_position_action(action)
        elif self.control_mode == 2:
            self._process_joint_velocity_action(action)
        elif self.control_mode == 3:
            self._process_joint_target_action(action)
    
    def _process_ik_action(self, action, current_ee_pos=None, current_ee_orn=None):
        """
        Control mode 0: Inverse kinematics
        Action: [dx, dy, dz, droll, dpitch, dyaw] in range [-1, 1]
        """
        if current_ee_pos is None or current_ee_orn is None:
            current_ee_pos, current_ee_orn = self.robot.get_end_effector_pose()
        
        # Compute position delta
        pos_delta = action[:3] * self.ik_xyz_delta
        new_pos = current_ee_pos + pos_delta
        
        # Compute rotation delta (convert quaternion to euler, add delta, convert back)
        current_rpy = np.array(p.getEulerFromQuaternion(current_ee_orn))
        rpy_delta = action[3:6] * self.ik_rpy_delta
        new_rpy = current_rpy + rpy_delta
        
        # Move to new pose via IK
        self.move_to_pose_ik(new_pos, new_rpy)
    
    def _process_joint_position_action(self, action):
        """
        Control mode 1: Joint positions
        Action: normalized joint positions [-1, 1] mapped to joint limits
        """
        # Map from [-1, 1] to joint limits
        joint_ranges = self.robot.joint_upper_limits - self.robot.joint_lower_limits
        new_joints = action[:6] * (joint_ranges / 2) + (self.robot.joint_lower_limits + self.robot.joint_upper_limits) / 2
        
        # If not using physics sim, clamp to max velocity
        if not self.use_physics_sim:
            current_joints, _ = self.robot.get_joint_states()
            joint_delta = new_joints - current_joints
            joint_dist = np.linalg.norm(joint_delta)
            
            if joint_dist > 1e-6:
                joint_delta = joint_delta / joint_dist
                # Estimate max velocity if not available
                max_step = 0.1 * self.sim_step  # Conservative default
                
                if joint_dist > max_step:
                    joint_delta = joint_delta * max_step
                else:
                    joint_delta = joint_delta * joint_dist
                
                new_joints = current_joints + joint_delta
        
        self.robot.set_joint_positions(new_joints)
    
    def _process_joint_velocity_action(self, action):
        """
        Control mode 2: Joint velocities
        Action: normalized joint velocities [-1, 1]
        """
        # Default max velocity
        max_vels = np.ones(6) * 2.0
        
        # Map from [-1, 1] to velocity limits
        velocities = action[:6] * max_vels
        
        if not self.use_physics_sim:
            # Compute position change and apply
            joint_delta = velocities * self.sim_step
            current_joints, _ = self.robot.get_joint_states()
            new_joints = current_joints + joint_delta
            self.robot.set_joint_positions(new_joints)
        else:
            # Use velocity control
            self.robot.set_joint_velocities(velocities)
    
    def _process_joint_target_action(self, action):
        """
        Control mode 3: Joint target with deviation
        Action: deviation from control_target [-1, 1]
        Requires control_target to be set externally
        """
        if len(self.control_target) == 0:
            raise ValueError("Control mode 3 requires control_target to be set! Use set_control_target().")
        
        # Add deviation to target
        deviation = action[:6] * self.joint_target_delta
        new_joints = np.array(self.control_target[:6]) + deviation
        
        self.robot.set_joint_positions(new_joints)
    
    def move_to_pose_ik(self, position, orientation_rpy=None):
        """
        Move robot to desired end effector pose using inverse kinematics.
        
        Args:
            position: [x, y, z] target position
            orientation_rpy: [roll, pitch, yaw] target orientation (None to ignore)
        """
        if orientation_rpy is not None:
            target_orn = p.getQuaternionFromEuler(orientation_rpy)
        else:
            target_orn = None
        
        # Solve IK
        joint_positions = self._solve_ik(position, target_orn)
        
        # Apply joint positions
        if joint_positions is not None:
            self.robot.set_joint_positions(joint_positions[:6])
    
    def _solve_ik(self, target_position, target_orientation=None):
        """
        Solve inverse kinematics for target pose.
        
        Args:
            target_position: [x, y, z]
            target_orientation: quaternion [x, y, z, w] or None
            
        Returns:
            joint_positions: array of joint angles
        """
        if self.robot.robot_id is None or self.robot.end_effector_link_idx is None:
            return None
        
        if target_orientation is not None:
            joint_positions = p.calculateInverseKinematics(
                self.robot.robot_id,
                self.robot.end_effector_link_idx,
                target_position,
                target_orientation,
                lowerLimits=self.robot.joint_lower_limits.tolist(),
                upperLimits=self.robot.joint_upper_limits.tolist(),
                jointRanges=self.robot.joint_ranges.tolist(),
                restPoses=self.robot.home_positions
            )
        else:
            joint_positions = p.calculateInverseKinematics(
                self.robot.robot_id,
                self.robot.end_effector_link_idx,
                target_position,
                lowerLimits=self.robot.joint_lower_limits.tolist(),
                upperLimits=self.robot.joint_upper_limits.tolist(),
                jointRanges=self.robot.joint_ranges.tolist(),
                restPoses=self.robot.home_positions
            )
        
        return np.array(joint_positions)
    
    def set_control_target(self, target_joints):
        """
        Set control target for control mode 3.
        
        Args:
            target_joints: array of target joint positions
        """
        self.control_target = np.array(target_joints)
    
    def get_control_mode_name(self):
        """Get the name of current control mode"""
        return CONTROL_MODES[self.control_mode]
    
    def set_control_mode(self, mode: Union[int, str]):
        """
        Change control mode.
        
        Args:
            mode: New control mode (0-3 or string name)
        """
        if isinstance(mode, str):
            assert mode in CONTROL_MODES, f"Unknown control mode: {mode}"
            self.control_mode = CONTROL_MODES.index(mode)
        else:
            assert 0 <= mode < len(CONTROL_MODES), f"Control mode must be 0-{len(CONTROL_MODES)-1}"
            self.control_mode = mode
        
        print(f"Control mode changed to: {self.get_control_mode_name()}")
