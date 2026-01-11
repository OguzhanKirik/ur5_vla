"""
UR5 Robot Controller with RRT Planner and Camera Visualization
Combines camera views, joint control, and RRT path planning for command-based robot movement.
"""

import numpy as np
import cv2
from pybullet_env import UR5GraspEnv
import pybullet as p
import time
import threading
from queue import Queue


class RobotControllerRRT:
    """
    Interactive robot controller with RRT path planning capabilities.
    Combines camera visualization, joint control sliders, and command-based movement.
    """
    
    def __init__(self, gui=True, fps=30):
        """
        Initialize the robot controller with environment and path planner.
        
        Args:
            gui: Whether to show PyBullet GUI window
            fps: Frames per second for simulation
        """
        print("Initializing UR5 Robot Controller with RRT Planner...")
        print("=" * 70)
        
        # Create environment
        self.env = UR5GraspEnv(gui=gui, fps=fps)
        self.obs = self.env.reset()
        
        # Home position in radians and degrees
        self.home_positions_rad = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]
        self.home_positions_deg = [int(np.degrees(angle)) for angle in self.home_positions_rad]
        
        # Gripper home position
        self.gripper_home = 0
        
        # Current target position (for RRT planning)
        self.target_joint_positions = None
        self.planned_path = None
        self.current_path_index = 0
        self.is_executing_path = False
        self.manual_control_enabled = True  # Flag to enable/disable slider control
        
        # Command queue for thread-safe command processing
        self.command_queue = Queue()
        self.running = True
        
        # Setup control interface
        self._setup_control_window()
        self._setup_camera_windows()
        
        print("\n✓ Robot Controller Initialized Successfully")
        print(f"  - Robot has {len(self.env.arm_joint_indices)} controllable joints")
        print(f"  - {len(self.obs['images'])} cameras configured")
        print(f"  - Home position (degrees): {self.home_positions_deg}")
        print("=" * 70)
        
    def _setup_control_window(self):
        """Setup control window with sliders for joint control and camera parameters"""
        self.control_window = "Robot Control Panel"
        cv2.namedWindow(self.control_window)
        
        # Create trackbars for 6 arm joints (-180 to 180 degrees)
        for i in range(6):
            initial_value = self.home_positions_deg[i] + 180  # Offset by 180 for 0-360 range
            cv2.createTrackbar(f"Joint {i+1}", self.control_window, initial_value, 360, lambda x: None)
        
        # Create trackbars for gripper (2 joints, typically controlled together)
        cv2.createTrackbar("Gripper L", self.control_window, self.gripper_home + 50, 100, lambda x: None)
        cv2.createTrackbar("Gripper R", self.control_window, self.gripper_home + 50, 100, lambda x: None)
        
        # Create trackbars for wrist camera offset (cm, range -50 to 50)
        cv2.createTrackbar("Cam Offset X", self.control_window, 35, 100, lambda x: None)
        cv2.createTrackbar("Cam Offset Y", self.control_window, 55, 100, lambda x: None)
        cv2.createTrackbar("Cam Offset Z", self.control_window, 50, 100, lambda x: None)
        
        # Create trackbars for wrist camera target (cm, range -50 to 50)
        cv2.createTrackbar("Cam Target X", self.control_window, 50, 100, lambda x: None)
        cv2.createTrackbar("Cam Target Y", self.control_window, 85, 100, lambda x: None)
        cv2.createTrackbar("Cam Target Z", self.control_window, 50, 100, lambda x: None)
        
        # Create trackbars for camera orientation
        cv2.createTrackbar("Cam Yaw", self.control_window, 180, 360, lambda x: None)
        cv2.createTrackbar("Cam Pitch", self.control_window, 180, 360, lambda x: None)
        cv2.createTrackbar("Cam Roll", self.control_window, 180, 360, lambda x: None)
        
    def _setup_camera_windows(self):
        """Setup windows for camera visualization"""
        cv2.namedWindow("Top Camera")
        cv2.namedWindow("Front Camera")
        cv2.namedWindow("Wrist Camera")
        
    def _read_joint_angles_from_sliders(self):
        """Read current joint angles from control sliders"""
        joint_angles = []
        for i in range(6):
            value = cv2.getTrackbarPos(f"Joint {i+1}", self.control_window)
            angle_deg = value - 180  # Convert from 0-360 back to -180 to 180
            angle_rad = np.radians(angle_deg)
            joint_angles.append(angle_rad)
        return joint_angles
    
    def _read_gripper_positions(self):
        """Read gripper positions from sliders"""
        gripper_l = cv2.getTrackbarPos("Gripper L", self.control_window) - 50
        gripper_r = cv2.getTrackbarPos("Gripper R", self.control_window) - 50
        gripper_l_rad = np.radians(gripper_l)
        gripper_r_rad = np.radians(gripper_r)
        return gripper_l_rad, gripper_r_rad
    
    def _read_camera_parameters(self):
        """Read camera parameters from sliders"""
        # Camera offset (convert from cm to meters)
        cam_offset_x = (cv2.getTrackbarPos("Cam Offset X", self.control_window) - 50) / 100.0
        cam_offset_y = (cv2.getTrackbarPos("Cam Offset Y", self.control_window) - 50) / 100.0
        cam_offset_z = (cv2.getTrackbarPos("Cam Offset Z", self.control_window) - 50) / 100.0
        
        # Camera target (convert from cm to meters)
        cam_target_x = (cv2.getTrackbarPos("Cam Target X", self.control_window) - 50) / 100.0
        cam_target_y = (cv2.getTrackbarPos("Cam Target Y", self.control_window) - 50) / 100.0
        cam_target_z = (cv2.getTrackbarPos("Cam Target Z", self.control_window) - 50) / 100.0
        
        # Camera orientation (convert from 0-360 to -180 to 180 degrees, then to radians)
        cam_yaw_rad = np.radians(cv2.getTrackbarPos("Cam Yaw", self.control_window) - 180)
        cam_pitch_rad = np.radians(cv2.getTrackbarPos("Cam Pitch", self.control_window) - 180)
        cam_roll_rad = np.radians(cv2.getTrackbarPos("Cam Roll", self.control_window) - 180)
        
        return {
            'offset': [cam_offset_x, cam_offset_y, cam_offset_z],
            'target': [cam_target_x, cam_target_y, cam_target_z],
            'yaw': cam_yaw_rad,
            'pitch': cam_pitch_rad,
            'roll': cam_roll_rad
        }
    
    def _apply_joint_positions(self, joint_angles):
        """Apply joint positions to robot"""
        for i, joint_idx in enumerate(self.env.arm_joint_indices[:6]):
            p.setJointMotorControl2(
                self.env.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=500
            )
    
    def _apply_gripper_positions(self, gripper_l_rad, gripper_r_rad):
        """Apply gripper positions to robot (if gripper joints exist)"""
        if len(self.env.arm_joint_indices) >= 8:
            p.setJointMotorControl2(
                self.env.robot_id, self.env.arm_joint_indices[6],
                p.POSITION_CONTROL,
                targetPosition=gripper_l_rad,
                force=50
            )
            p.setJointMotorControl2(
                self.env.robot_id, self.env.arm_joint_indices[7],
                p.POSITION_CONTROL,
                targetPosition=gripper_r_rad,
                force=50
            )
    
    def _update_camera_parameters(self, cam_params):
        """Update environment camera parameters"""
        self.env.wrist_cam_offset = cam_params['offset']
        self.env.wrist_cam_target = cam_params['target']
        self.env.wrist_cam_yaw = cam_params['yaw']
        self.env.wrist_cam_pitch = cam_params['pitch']
        self.env.wrist_cam_roll = cam_params['roll']
    
    def _display_cameras(self, obs):
        """Display camera images in OpenCV windows"""
        # Map environment keys to display window names
        key_mapping = {
            'observation.images.image_with_depth': 'Top Camera',
            'observation.images.image': 'Front Camera',
            'observation.images.hand_image': 'Wrist Camera'
        }
        
        for env_key, window_name in key_mapping.items():
            if env_key in obs['images']:
                img_bgr = cv2.cvtColor(obs['images'][env_key], cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, img_bgr)
    
    def get_current_joint_positions(self):
        """Get current joint positions from the robot"""
        joint_states = []
        for joint_idx in self.env.arm_joint_indices[:6]:
            state = p.getJointState(self.env.robot_id, joint_idx)
            joint_states.append(state[0])  # Position
        return np.array(joint_states)
    
    def get_end_effector_pose(self):
        """Get current end-effector position and orientation"""
        ee_link_idx = self.env.arm_joint_indices[-1] if self.env.arm_joint_indices else 0
        ee_state = p.getLinkState(self.env.robot_id, ee_link_idx)
        position = np.array(ee_state[0])
        orientation = np.array(ee_state[1])  # Quaternion
        euler = p.getEulerFromQuaternion(orientation)
        return position, euler
    
    def solve_ik(self, target_pos, target_orn=None):
        """
        Solve inverse kinematics to get joint angles for target position.
        
        Args:
            target_pos: Target position [x, y, z] in meters
            target_orn: Target orientation as euler angles [roll, pitch, yaw] (optional)
        
        Returns:
            joint_angles: Array of 6 joint angles in radians, or None if IK failed
        """
        ee_link_idx = self.env.arm_joint_indices[-1] if self.env.arm_joint_indices else 0
        
        # Convert target orientation to quaternion if provided
        if target_orn is not None:
            target_orn_quat = p.getQuaternionFromEuler(target_orn)
        else:
            # Use current orientation if not specified
            _, current_orn = self.get_end_effector_pose()
            target_orn_quat = p.getQuaternionFromEuler(current_orn)
        
        # Solve IK
        joint_poses = p.calculateInverseKinematics(
            self.env.robot_id,
            ee_link_idx,
            target_pos,
            target_orn_quat,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        
        # Return only the first 6 joints (arm joints)
        if joint_poses:
            return np.array(joint_poses[:6])
        return None
    
    def move_to_position(self, target_pos, target_orn=None, use_rrt=False):
        """
        Move to a target Cartesian position using inverse kinematics.
        
        Args:
            target_pos: Target position [x, y, z] in meters
            target_orn: Target orientation [roll, pitch, yaw] in radians (optional)
            use_rrt: Whether to use RRT planning
        """
        print(f"\n[IK] Computing joint angles for position: {target_pos}")
        if target_orn is not None:
            print(f"     with orientation: {np.degrees(target_orn)} degrees")
        
        # Solve IK
        joint_angles_rad = self.solve_ik(target_pos, target_orn)
        
        if joint_angles_rad is None:
            print("✗ IK solution failed!")
            return
        
        # Convert to degrees for display
        joint_angles_deg = np.degrees(joint_angles_rad)
        print(f"[IK] Solution: {np.round(joint_angles_deg, 1)}")
        
        # Move to the computed joint angles
        self.move_to_joint_position(joint_angles_deg.tolist(), use_rrt=use_rrt)
    
    def simple_rrt_plan(self, start_config, goal_config, max_iterations=1000, step_size=0.1):
        """
        Simple RRT path planner (without collision checking for now).
        
        Args:
            start_config: Starting joint configuration (6D array)
            goal_config: Goal joint configuration (6D array)
            max_iterations: Maximum planning iterations
            step_size: Step size for extending tree
            
        Returns:
            path: List of joint configurations forming path from start to goal
        """
        print(f"\n[RRT] Planning path from current position to goal...")
        print(f"  Start: {np.round(np.degrees(start_config), 1)}")
        print(f"  Goal:  {np.round(np.degrees(goal_config), 1)}")
        
        # Tree stores nodes (each node is a configuration)
        tree = [start_config]
        parent = {0: None}  # Parent indices
        
        goal_threshold = 0.1  # Radians
        
        for iteration in range(max_iterations):
            # Sample random configuration (with goal bias)
            if np.random.random() < 0.2:  # 20% goal bias
                rand_config = goal_config
            else:
                rand_config = np.random.uniform(
                    self.env.joint_lower_limits,
                    self.env.joint_upper_limits
                )
            
            # Find nearest node in tree
            nearest_idx = 0
            min_dist = np.linalg.norm(tree[0] - rand_config)
            for i, node in enumerate(tree):
                dist = np.linalg.norm(node - rand_config)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            # Extend from nearest node toward random config
            nearest_config = tree[nearest_idx]
            direction = rand_config - nearest_config
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            new_config = nearest_config + step_size * direction
            
            # Clamp to joint limits
            new_config = np.clip(new_config, self.env.joint_lower_limits, self.env.joint_upper_limits)
            
            # Add to tree
            tree.append(new_config)
            parent[len(tree) - 1] = nearest_idx
            
            # Check if we reached the goal
            if np.linalg.norm(new_config - goal_config) < goal_threshold:
                print(f"[RRT] ✓ Path found in {iteration + 1} iterations!")
                
                # Reconstruct path
                path = []
                current_idx = len(tree) - 1
                while current_idx is not None:
                    path.append(tree[current_idx])
                    current_idx = parent[current_idx]
                path.reverse()
                
                print(f"[RRT] Path has {len(path)} waypoints")
                return path
        
        print(f"[RRT] ✗ Failed to find path in {max_iterations} iterations")
        # Return straight line as fallback
        return [start_config, goal_config]
    
    def execute_planned_path(self, path, steps_per_waypoint=30):
        """
        Execute a planned path by moving through each waypoint.
        
        Args:
            path: List of joint configurations
            steps_per_waypoint: Number of simulation steps per waypoint
        """
        print(f"\n[Execute] Starting path execution with {len(path)} waypoints...")
        
        # Disable manual control during execution
        self.manual_control_enabled = False
        
        for i, waypoint in enumerate(path):
            print(f"  Moving to waypoint {i+1}/{len(path)}")
            
            # Set target position for all joints
            self._apply_joint_positions(waypoint)
            
            # Simulate for several steps to allow movement
            for step in range(steps_per_waypoint):
                p.stepSimulation()
                
                # Update camera view every few steps
                if step % 5 == 0:
                    obs = {
                        'images': self.env.get_camera_images(),
                        'state': self.env.get_state()
                    }
                    self._display_cameras(obs)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.manual_control_enabled = True
                    return False
                
                time.sleep(self.env.dt)
        
        print(f"[Execute] ✓ Path execution complete!")
        
        # Re-enable manual control
        self.manual_control_enabled = True
        return True
    
    def move_to_joint_position(self, target_joint_angles_deg, use_rrt=False):
        """
        Move to target joint position. Can use RRT planning or simple interpolation.
        
        Args:
            target_joint_angles_deg: Target joint angles in degrees [j1, j2, j3, j4, j5, j6]
            use_rrt: If True, use RRT path planning. If False, use simple interpolation (faster)
        """
        # Convert degrees to radians
        target_joint_angles_rad = np.array([np.radians(angle) for angle in target_joint_angles_deg])
        
        # Get current position
        current_position = self.get_current_joint_positions()
        
        if use_rrt:
            # Plan path using RRT (for obstacle avoidance)
            print("[Planning] Using RRT path planner...")
            path = self.simple_rrt_plan(current_position, target_joint_angles_rad)
        else:
            # Simple linear interpolation in joint space (faster, no collision checking)
            print("[Planning] Using direct joint interpolation...")
            path = self.interpolate_joint_path(current_position, target_joint_angles_rad, num_waypoints=20)
        
        # Execute path
        success = self.execute_planned_path(path)
        
        if success:
            print(f"✓ Successfully moved to target position!")
        else:
            print(f"✗ Movement interrupted")
    
    def interpolate_joint_path(self, start_config, goal_config, num_waypoints=20):
        """
        Create a smooth path by linear interpolation in joint space.
        This is faster than RRT and works well when there are no obstacles.
        
        Args:
            start_config: Starting joint configuration (6D array)
            goal_config: Goal joint configuration (6D array)
            num_waypoints: Number of intermediate waypoints
            
        Returns:
            path: List of joint configurations
        """
        path = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            waypoint = (1 - alpha) * start_config + alpha * goal_config
            path.append(waypoint)
        return path
    
    def move_home(self, use_rrt=False):
        """Move robot to home position"""
        print("\n[Command] Moving to HOME position...")
        self.move_to_joint_position(self.home_positions_deg, use_rrt=use_rrt)
    
    def move_to_preset(self, preset_name, use_rrt=False):
        """
        Move to a preset position.
        
        Args:
            preset_name: Name of preset ('home', 'inspect', 'reach', 'retract')
            use_rrt: If True, use RRT planning. If False, use simple interpolation
        """
        presets = {
            'home': [0, -90, 90, -90, -90, 90],
            'inspect': [0, -45, 45, -90, -90, 90],  # Tilted up for inspection
            'reach': [0, -60, 120, -150, -90, 90],  # Extended forward
            'retract': [0, -120, 60, -30, -90, 90],  # Pulled back
        }
        
        if preset_name.lower() in presets:
            print(f"\n[Command] Moving to preset: {preset_name.upper()}")
            self.move_to_joint_position(presets[preset_name.lower()], use_rrt=use_rrt)
        else:
            print(f"✗ Unknown preset: {preset_name}")
            print(f"  Available presets: {list(presets.keys())}")
    
    def print_current_position(self):
        """Print current joint positions and end-effector pose"""
        current_pos = self.get_current_joint_positions()
        current_pos_deg = np.degrees(current_pos)
        ee_pos, ee_orn = self.get_end_effector_pose()
        
        print(f"\n[Info] Current Robot State:")
        print(f"  Joint Positions:")
        for i, (rad, deg) in enumerate(zip(current_pos, current_pos_deg)):
            print(f"    Joint {i+1}: {deg:7.2f}° ({rad:7.4f} rad)")
        
        print(f"\n  End-Effector Pose:")
        print(f"    Position: x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f} m")
        print(f"    Orientation: roll={np.degrees(ee_orn[0]):.1f}°, pitch={np.degrees(ee_orn[1]):.1f}°, yaw={np.degrees(ee_orn[2]):.1f}°")
    
    def process_command(self, command):
        """
        Process a text command from the user.
        
        Args:
            command: String command to process
        """
        command = command.strip().lower()
        
        if not command:
            return True
        
        parts = command.split()
        cmd = parts[0]
        
        if cmd in ['q', 'quit', 'exit']:
            return False
        
        elif cmd in ['h', 'home']:
            self.move_home(use_rrt=False)
        
        elif cmd == 'home-rrt':
            self.move_home(use_rrt=True)
        
        elif cmd in ['p', 'pos', 'position']:
            self.print_current_position()
        
        elif cmd == 'help':
            self.print_help()
        
        elif cmd == 'preset':
            if len(parts) >= 2:
                self.move_to_preset(parts[1], use_rrt=False)
            else:
                print("✗ Usage: preset <name>")
                print("  Available: inspect, reach, retract")
        
        elif cmd == 'move':
            if len(parts) >= 7:
                try:
                    joint_angles = [float(parts[i]) for i in range(1, 7)]
                    print(f"[Command] Moving to: {joint_angles}")
                    self.move_to_joint_position(joint_angles, use_rrt=False)
                except ValueError:
                    print("✗ Invalid joint angles. Use: move J1 J2 J3 J4 J5 J6")
            else:
                print("✗ Usage: move J1 J2 J3 J4 J5 J6 (angles in degrees)")
                print("  Example: move 0 -45 90 -90 -90 45")
        
        elif cmd == 'move-rrt':
            if len(parts) >= 7:
                try:
                    joint_angles = [float(parts[i]) for i in range(1, 7)]
                    print(f"[Command] Moving to: {joint_angles} (with RRT)")
                    self.move_to_joint_position(joint_angles, use_rrt=True)
                except ValueError:
                    print("✗ Invalid joint angles. Use: move-rrt J1 J2 J3 J4 J5 J6")
            else:
                print("✗ Usage: move-rrt J1 J2 J3 J4 J5 J6 (angles in degrees)")
        
        elif cmd in ['goto', 'moveto']:
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    target_orn = None
                    # Check if orientation is provided (roll, pitch, yaw)
                    if len(parts) >= 7:
                        roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
                        target_orn = [np.radians(roll), np.radians(pitch), np.radians(yaw)]
                    
                    print(f"[Command] Moving to position: ({x}, {y}, {z})")
                    self.move_to_position([x, y, z], target_orn, use_rrt=False)
                except ValueError:
                    print("✗ Invalid position. Use: goto x y z [roll pitch yaw]")
            else:
                print("✗ Usage: goto x y z [roll pitch yaw] (position in meters, angles in degrees)")
                print("  Example: goto 0.5 0.3 0.4")
                print("  Example: goto 0.5 0.3 0.4 0 90 0")
        
        elif cmd == 'goto-rrt':
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    target_orn = None
                    if len(parts) >= 7:
                        roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
                        target_orn = [np.radians(roll), np.radians(pitch), np.radians(yaw)]
                    
                    print(f"[Command] Moving to position: ({x}, {y}, {z}) with RRT")
                    self.move_to_position([x, y, z], target_orn, use_rrt=True)
                except ValueError:
                    print("✗ Invalid position. Use: goto-rrt x y z [roll pitch yaw]")
            else:
                print("✗ Usage: goto-rrt x y z [roll pitch yaw]")
        
        elif cmd == 'open':
            print("[Gripper] Opening")
            gripper_l_rad = np.radians(50)  # Open position
            gripper_r_rad = np.radians(50)
            self._apply_gripper_positions(gripper_l_rad, gripper_r_rad)
        
        elif cmd == 'close':
            print("[Gripper] Closing")
            gripper_l_rad = np.radians(-50)  # Closed position
            gripper_r_rad = np.radians(-50)
            self._apply_gripper_positions(gripper_l_rad, gripper_r_rad)
        
        elif cmd == 'gripper' and len(parts) >= 2:
            try:
                state = float(parts[1])
                state = np.clip(state, 0.0, 1.0)
                # Map 0-1 to gripper angle range (-50 to 50 degrees)
                angle_deg = -50 + state * 100
                angle_rad = np.radians(angle_deg)
                print(f"[Gripper] Setting to {state:.2f} (0=closed, 1=open)")
                self._apply_gripper_positions(angle_rad, angle_rad)
            except ValueError:
                print("✗ Invalid gripper value. Use 0.0 (closed) to 1.0 (open)")
        
        else:
            print(f"✗ Unknown command: {cmd}")
            print("  Type 'help' for available commands")
        
        return True
    
    def print_help(self):
        """Print help information about available commands"""
        print("\n" + "=" * 70)
        print("ROBOT CONTROLLER COMMANDS")
        print("=" * 70)
        print("\nBasic Commands:")
        print("  home           - Move to home position (fast interpolation)")
        print("  home-rrt       - Move to home position (with RRT planning)")
        print("  pos            - Print current joint positions and end-effector pose")
        print("  quit           - Quit the application")
        print("  help           - Show this help message")
        print("\nPreset Movements (fast interpolation):")
        print("  preset inspect - Move to inspection position")
        print("  preset reach   - Move to reaching position")
        print("  preset retract - Move to retracted position")
        print("\nJoint Space Movement:")
        print("  move J1 J2 J3 J4 J5 J6     - Move to joint angles (degrees)")
        print("  move-rrt J1 J2 J3 J4 J5 J6 - Move with RRT planning")
        print("\nCartesian Space Movement (using Inverse Kinematics):")
        print("  goto x y z                    - Move to position (meters)")
        print("  goto x y z roll pitch yaw     - Move to position with orientation (deg)")
        print("  goto-rrt x y z [r p y]        - Move with RRT planning")
        print("\nGripper Control:")
        print("  open              - Open gripper")
        print("  close             - Close gripper")
        print("  gripper <0-1>     - Set gripper (0=closed, 1=open)")
        print("\nExamples:")
        print("  > move 0 -45 90 -90 -90 45")
        print("  > goto 0.5 0.3 0.4")
        print("  > goto 0.5 0.2 0.5 0 90 0")
        print("  > open")
        print("  > close")
        print("  > gripper 0.5")
        print("  > preset reach")
        print("  > home")
        print("\nKeyboard Shortcuts:")
        print("  h - Quick home    p - Show position    q - Quit")
        print("\nSlider Controls:")
        print("  - Use sliders in 'Robot Control Panel' for manual joint control")
        print("  - Camera parameters can be adjusted with camera sliders")
        print("=" * 70)
    
    def run(self):
        """Main control loop with command interface"""
        print("\n" + "=" * 70)
        print("ROBOT CONTROLLER - INTERACTIVE MODE")
        print("=" * 70)
        print("\nCamera views are displayed. Type commands to control the robot.")
        print("Type 'help' for available commands.\n")
        print("Command examples:")
        print("  > home                        (move to home position)")
        print("  > goto 0.5 0.3 0.4            (move to x,y,z position)")
        print("  > move 0 -45 90 -90 -90 45    (move to joint angles)")
        print("  > preset reach                (move to reach preset)")
        print("  > pos                         (show current position)")
        print("=" * 70)
        print("\n> ", end='', flush=True)
        
        # Start command input thread
        def command_input_thread():
            while self.running:
                try:
                    command = input()
                    self.command_queue.put(command)
                    if not self.running:
                        break
                except EOFError:
                    break
                except:
                    pass
        
        input_thread = threading.Thread(target=command_input_thread, daemon=True)
        input_thread.start()
        
        # Start in non-blocking mode
        cv2.startWindowThread()
        
        try:
            while True:
                # Process any pending commands
                while not self.command_queue.empty():
                    command = self.command_queue.get()
                    if not self.process_command(command):
                        self.running = False
                        break
                    print("> ", end='', flush=True)
                
                if not self.running:
                    break
                
                # Read control inputs from sliders (only if manual control is enabled)
                if self.manual_control_enabled:
                    joint_angles = self._read_joint_angles_from_sliders()
                    gripper_l, gripper_r = self._read_gripper_positions()
                    cam_params = self._read_camera_parameters()
                    
                    # Apply to robot
                    self._apply_joint_positions(joint_angles)
                    self._apply_gripper_positions(gripper_l, gripper_r)
                    self._update_camera_parameters(cam_params)
                    
                    # Step simulation
                    p.stepSimulation()
                    
                    # Get and display camera images
                    obs = {
                        'images': self.env.get_camera_images(),
                        'state': self.env.get_state()
                    }
                    self._display_cameras(obs)
                else:
                    # During automated movement, just update camera parameters
                    cam_params = self._read_camera_parameters()
                    self._update_camera_parameters(cam_params)
                
                # Check for keyboard shortcuts
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Quit] Exiting...")
                    self.running = False
                    break
                elif key == ord('h'):
                    self.move_home(use_rrt=False)
                    print("> ", end='', flush=True)
                elif key == ord('p'):
                    self.print_current_position()
                    print("> ", end='', flush=True)
                
                time.sleep(self.env.dt)
                
        except KeyboardInterrupt:
            print("\n[Interrupt] Shutting down...")
            self.running = False
        finally:
            cv2.destroyAllWindows()
            p.disconnect()


def main():
    """Main entry point"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "UR5 ROBOT CONTROLLER WITH RRT PLANNER" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Create controller
    controller = RobotControllerRRT(gui=True, fps=30)
    
    # Print help
    controller.print_help()
    
    # Run main loop
    controller.run()


if __name__ == "__main__":
    main()
