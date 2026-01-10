from typing import Union, List
import numpy as np
from modular_drl_env.robot.robot import Robot
from time import process_time

__all__ = [
    'UR5',
    'UR5_Gripper',
]

class UR5(Robot):

    def __init__(self, name: str,
                       id_num: int,
                       world,
                       sim_step: float,
                       use_physics_sim: bool,
                       base_position: Union[list, np.ndarray], 
                       base_orientation: Union[list, np.ndarray], 
                       resting_angles: Union[list, np.ndarray], 
                       control_mode: Union[int, str], 
                       ik_xyz_delta: float=0.005,
                       ik_rpy_delta: float=0.005,
                       jt_joint_delta: float=0.5,
                       joint_velocities_overwrite: Union[float, List]=1,
                       joint_limits_overwrite: Union[float, List]=1,
                       controlled_joints: list=[],
                       self_collision: bool=True):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, ik_xyz_delta, ik_rpy_delta, jt_joint_delta, joint_velocities_overwrite, joint_limits_overwrite, controlled_joints, self_collision)
        self.end_effector_link_id = "ee_link"
        self.base_link_id = "base_link"

        self.urdf_path = "assets/robots/predefined/ur5/urdf/ur5.urdf"  
    
class UR5_Gripper(UR5):
    def __init__(self, name: str,
                       id_num: int,
                       world,
                       sim_step: float,
                       use_physics_sim: bool,
                       base_position: Union[list, np.ndarray], 
                       base_orientation: Union[list, np.ndarray], 
                       resting_angles: Union[list, np.ndarray], 
                       control_mode: Union[int, str], 
                       ik_xyz_delta: float=0.005,
                       ik_rpy_delta: float=0.005,
                       jt_joint_delta: float=0.5,
                       joint_velocities_overwrite: Union[float, List]=1,
                       joint_limits_overwrite: Union[float, List]=1,
                       controlled_joints: list=[],
                       self_collision: bool=True,
                       gripper_control: bool=True):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, ik_xyz_delta, ik_rpy_delta, jt_joint_delta, joint_velocities_overwrite, joint_limits_overwrite, controlled_joints, self_collision)

        self.urdf_path = "assets/robots/predefined/ur5/urdf/ur5_rg2.urdf"
        self.end_effector_link_id = "tool0"  # ur5_rg2.urdf uses 'tool0' instead of 'ee_link'
        
        # Gripper control parameters
        self.gripper_control = gripper_control  # Whether gripper is controllable
        self.gripper_state = 0.0  # 0.0 = open, 1.0 = closed
        self.gripper_threshold = 0.5  # Threshold to consider gripper closed
        self.grasped_object_id = None  # ID of currently grasped object
        self.grasp_constraint = None  # PyBullet constraint for grasping
        self.grasp_distance = 0.08  # Maximum distance to grasp an object (8cm)
        
        # Finger joint parameters (will be set after robot is loaded)
        self.left_finger_joint_index = None
        self.right_finger_joint_index = None
        self.finger_open_angle = 1.18  # Fully open (max rotation in radians for revolute joints)
        self.finger_closed_angle = 0.0  # Fully closed (min rotation in radians)
    
    def build(self):
        """
        Override build to initialize finger joints and exclude them from controlled joints.
        """
        # First, build the robot to load URDF and get all joint IDs
        super().build()
        
        if self.gripper_control:
            # Find and store finger joint indices
            self._initialize_finger_joints()
            
            # Exclude finger joints from controlled_joints_ids
            # (They will be controlled separately via _control_finger_joints)
            if self.left_finger_joint_index is not None:
                original_controlled = list(self.controlled_joints_ids)
                finger_joints = [self.left_finger_joint_index, self.right_finger_joint_index]
                
                # Remove finger joints from controlled list
                self.controlled_joints_ids = [j for j in original_controlled if j not in finger_joints]
                
                # Recalculate indices_controlled
                self.indices_controlled = []
                for idx, joint_id in enumerate(self.all_joints_ids):
                    if joint_id in self.controlled_joints_ids:
                        self.indices_controlled.append(idx)
                self.indices_controlled = np.array(self.indices_controlled)
                
                # Update the exposed subset of controlled joint attributes
                self.joints_limits_lower = self._joints_limits_lower[self.indices_controlled]
                self.joints_limits_upper = self._joints_limits_upper[self.indices_controlled]
                self.joints_range = self._joints_range[self.indices_controlled]
                self.joints_max_forces = self._joints_max_forces[self.indices_controlled]
                self.joints_max_velocities = self._joints_max_velocities[self.indices_controlled]
                self.resting_pose_angles = self._resting_pose_angles[self.indices_controlled]
    
    def get_action_space_dims(self):
        """
        Override to add gripper dimension to action space.
        Returns (joint_dims + gripper_dim, ik_dims + gripper_dim).
        """
        base_dims = super().get_action_space_dims()
        if self.gripper_control:
            return (base_dims[0] + 1, base_dims[1] + 1)
        else:
            return base_dims
    
    def _initialize_finger_joints(self):
        """
        Find and store the indices of the gripper finger joints.
        Should be called after the robot URDF is loaded.
        """
        import pybullet as pyb
        from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
        
        robot_pyb_id = pyb_u.to_pb(self.object_id)
        
        # Find finger joint indices (for ur5_rg2.urdf: rg2_finger_joint1 and rg2_finger_joint2)
        for i in range(pyb.getNumJoints(robot_pyb_id)):
            joint_info = pyb.getJointInfo(robot_pyb_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            if joint_name == "rg2_finger_joint1":
                self.left_finger_joint_index = i
            elif joint_name == "rg2_finger_joint2":
                self.right_finger_joint_index = i
        
        if self.left_finger_joint_index is None or self.right_finger_joint_index is None:
            print(f"Warning: Could not find finger joints. Left: {self.left_finger_joint_index}, Right: {self.right_finger_joint_index}")
    
    def _control_finger_joints(self):
        """
        Control the physical finger joints based on gripper_state.
        gripper_state: 0.0 = fully open, 1.0 = fully closed
        For revolute joints: angle 0.0 = closed, angle 1.18 radians = open
        """
        import pybullet as pyb
        from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
        
        if self.left_finger_joint_index is None:
            return
        
        # Map gripper_state (0=open, 1=closed) to finger angle
        # finger_angle: 0.0 = closed, 1.18 = open
        target_angle = self.finger_open_angle * (1.0 - self.gripper_state)
        
        robot_pyb_id = pyb_u.to_pb(self.object_id)
        
        # Control both fingers explicitly (PyBullet doesn't auto-handle mimic joints)
        pyb.setJointMotorControl2(
            bodyUniqueId=robot_pyb_id,
            jointIndex=self.left_finger_joint_index,
            controlMode=pyb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=10.6,  # Match effort from URDF
            maxVelocity=1.57  # Match velocity from URDF
        )
        
        if self.right_finger_joint_index is not None:
            pyb.setJointMotorControl2(
                bodyUniqueId=robot_pyb_id,
                jointIndex=self.right_finger_joint_index,
                controlMode=pyb.POSITION_CONTROL,
                targetPosition=target_angle,
                force=10.6,
                maxVelocity=1.57
            )
    
    def process_action(self, action: np.ndarray):
        """
        Override to handle gripper action.
        Last dimension of action controls gripper: -1 = open, +1 = close
        """
        if self.gripper_control:
            # Split action into arm and gripper components
            arm_action = action[:-1]
            gripper_action = action[-1]
            
            # Update gripper state (map from [-1, 1] to [0, 1])
            self.gripper_state = (gripper_action + 1.0) / 2.0
            
            # Execute arm movement FIRST
            result = super().process_action(arm_action)
            
            # THEN control the physical finger joints (after arm movement)
            # This ensures finger control doesn't get overridden
            self._control_finger_joints()
            
            # Handle grasping logic (constraint-based for physics)
            self._update_grasp()
            
            return result
        else:
            return super().process_action(action)
    
    def _update_grasp(self):
        """
        Updates grasping state based on gripper state.
        Creates/removes constraints to simulate grasping.
        """
        import pybullet as pyb
        
        # Check if gripper should close
        if self.gripper_state > self.gripper_threshold:
            # Try to grasp if not already grasping
            if self.grasp_constraint is None:
                self._attempt_grasp()
        else:
            # Release if gripper is open
            if self.grasp_constraint is not None:
                self._release_grasp()
    
    def _attempt_grasp(self):
        """
        Attempts to grasp nearby objects by creating a constraint.
        """
        import pybullet as pyb
        from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
        
        # Get robot PyBullet ID
        robot_pyb_id = pyb_u.to_pb(self.object_id)
        
        # Get end-effector link index
        ee_link_index = -1
        for i in range(pyb.getNumJoints(robot_pyb_id)):
            link_info = pyb.getJointInfo(robot_pyb_id, i)
            link_name = link_info[12].decode('utf-8')
            if link_name == self.end_effector_link_id:
                ee_link_index = i
                break
        
        # Get end-effector position
        ee_state = pyb.getLinkState(robot_pyb_id, ee_link_index)
        ee_pos = np.array(ee_state[0])
        
        # Find nearby graspable objects
        closest_obj = None
        closest_dist = self.grasp_distance
        
        for obj in self.world.objects:
            # Skip non-graspable objects (e.g., ground, table)
            if hasattr(obj, 'seen_by_obstacle_sensor') and not obj.seen_by_obstacle_sensor:
                continue
            
            obj_pos = np.array(obj.position)
            dist = np.linalg.norm(ee_pos - obj_pos)
            
            if dist < closest_dist:
                closest_obj = obj
                closest_dist = dist
        
        # Create constraint if object found
        if closest_obj is not None:
            self.grasped_object_id = pyb_u.to_pb(closest_obj.object_id)
            self.grasp_constraint = pyb.createConstraint(
                parentBodyUniqueId=robot_pyb_id,
                parentLinkIndex=ee_link_index,
                childBodyUniqueId=self.grasped_object_id,
                childLinkIndex=-1,
                jointType=pyb.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
    
    def _release_grasp(self):
        """
        Releases grasped object by removing constraint.
        """
        import pybullet as pyb
        
        if self.grasp_constraint is not None:
            pyb.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            self.grasped_object_id = None
    
    def reset(self):
        """
        Override reset to release any grasped objects.
        """
        self._release_grasp()
        self.gripper_state = 0.0
        super().reset()