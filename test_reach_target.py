"""
Test script: Create environment from components and reach target object

Demonstrates:
- Setting up environment with modular components
- Finding target object (red)
- Moving robot to target using IK control
"""

import numpy as np
import pybullet as p
import pybullet_data
import time

from components import UR5RobotComponent, CameraComponent, ObjectsComponent, RobotController


def main():
    # Initialize PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
    print("\n=== Setting up Environment ===")
    
    # Create objects component first to create table
    objects = ObjectsComponent()
    
    # Create table matching pybullet_env.py
    # Table: 1m x 1m square, 36cm high, center at [0.5, 0.5, 0.18]
    table_pos = [0.5, 0.5, 0.18]
    table_size = [0.5, 0.5, 0.18]
    table_id = objects.create_table(
        position=table_pos,
        size=table_size
    )
    
    # Get accurate table height for robot placement
    table_aabb = p.getAABB(table_id)
    table_height = table_aabb[1][2]  # Top surface Z coordinate
    print(f"✓ Table created: 1m x 1m, top at Z = {table_height:.4f}")
    
    # Create robot component matching pybullet_env.py
    # Robot base at front edge center of table
    robot_pos = [0.5, 0, table_height]
    robot_orn = p.getQuaternionFromEuler([0, 0, -np.pi/2])  # -90 degree rotation
    
    robot = UR5RobotComponent(
        position=robot_pos,
        orientation=robot_orn,
        use_fixed_base=True
    )
    robot.load()
    print(f"✓ Robot loaded at front edge center: {robot.position}")
    
    # Spawn graspable objects on the table (includes a red box)
    spawned_ids = objects.spawn_graspable_objects(
        table_height=table_height,
        workspace_bounds=None  # Use default workspace
    )
    print(f"✓ Spawned {len(spawned_ids)} objects on table")
    
    # Create container box next to robot base for placing objects
    # Robot is at [0.5, 0, table_height], place box at left front corner of table
    # Table extends from X=0 to X=1.0, Y=0 to Y=1.0
    container_pos = [0.15, 0.15, table_height]  # Left front corner with slight inset
    objects.create_container_box(
        position=container_pos,
        size=[0.15, 0.15, 0.08],  # 15x15cm base, 8cm high
        wall_thickness=0.01,
        color=[0.2, 0.2, 0.5, 1]  # Dark blue color
    )
    print(f"✓ Container box created at left corner: {container_pos}")
    
    # The blue cylinder is the third object (index 2) according to objects_component.py
    # spawned_ids[0] = red box, spawned_ids[1] = green sphere, spawned_ids[2] = blue cylinder
    target_object_id = spawned_ids[2]
    target_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    print(f"\n=== Target Object ===")
    print(f"Blue cylinder at position: {target_pos}")
    
    # Create controller with IK mode
    controller = RobotController(
        robot_component=robot,
        control_mode="inverse_kinematics",
        ik_xyz_delta=0.05,
        ik_rpy_delta=0.05
    )
    print(f"\n✓ Controller initialized with mode: {controller.get_control_mode_name()}")
    
    # Create camera component matching pybullet_env.py
    camera = CameraComponent()
    
    # Camera 1: Main view with depth
    camera.add_fixed_camera(
        name="main_view",
        eye_position=[0.5, 1.0, 1.0],
        target_position=[0.5, 0.5, 0.36]
    )
    
    # Camera 2: Top-down view
    camera.add_fixed_camera(
        name="top_view",
        eye_position=[0.5, 1.0, 1.0],
        target_position=[0.5, 0.5, 0.36]
    )
    
    # Camera 3: Wrist camera (attached to robot last arm joint, not gripper tip)
    # Must use last arm joint like in pybullet_env.py, not the gripper tip
    wrist_link_idx = robot.arm_joint_indices[-1] if robot.arm_joint_indices else 0
    camera.add_wrist_camera(
        name="wrist_view",
        robot_component=robot,
        link_idx=wrist_link_idx
    )
    
    # Let physics settle
    print("\n=== Settling Physics ===")
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Get updated target object position after settling
    target_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    target_pos = np.array(target_pos)
    print(f"Blue cylinder settled at: {target_pos}")
    
    # Phase 1: Move to position 20cm above object
    approach_pos = target_pos.copy()
    approach_pos[2] += 0.20  # 20cm above object
    
    print(f"\n=== Phase 1: Moving Above Target ===")
    print(f"Approach position (20cm above): {approach_pos}")
    
    # Get current end effector pose
    current_pos, current_orn = robot.get_end_effector_pose()
    print(f"Current EE position: {current_pos}")
    print(f"Distance to approach: {np.linalg.norm(approach_pos - current_pos):.3f}m")
    
    # Move to approach position
    num_steps = 200
    distance_threshold = 0.02  # Stop when within 2cm of target
    print(f"\nMoving to approach position...")
    
    for step in range(num_steps):
        # Get current position
        current_pos, current_orn = robot.get_end_effector_pose()
        
        # Calculate delta to approach position
        delta = approach_pos - current_pos
        distance = np.linalg.norm(delta)
        
        # Check if reached approach position
        if distance < distance_threshold:
            print(f"\n✓ Reached approach position! Final distance: {distance:.4f}m")
            print(f"  Approach position: {approach_pos}")
            print(f"  Final EE position: {current_pos}")
            break
        
        # Normalize delta to unit vector, then scale to [-1, 1] range
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        # Create action (position delta + no rotation change + gripper open)
        action = np.concatenate([
            action_pos,
            [0, 0, 0],
            [1]  # gripper open
        ])
        
        # Process action through controller
        controller.process_action(action)
        
        # Step simulation multiple times for smoother motion
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Capture camera images every 10 steps
        if step % 10 == 0:
            images = camera.capture_all()
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"  Step {step:3d}: Distance to approach: {distance:.4f}m")
    
    # Phase 2: Descend to grasp position (1cm above object)
    grasp_pos = target_pos.copy()
    grasp_pos[2] += 0.01  # 1cm above object
    
    print(f"\n=== Phase 2: Descending to Grasp Position ===")
    print(f"Grasp position (1cm above): {grasp_pos}")
    
    for step in range(num_steps):
        # Get current position
        current_pos, current_orn = robot.get_end_effector_pose()
        
        # Calculate delta to grasp position
        delta = grasp_pos - current_pos
        distance = np.linalg.norm(delta)
        
        # Check if reached grasp position
        if distance < distance_threshold:
            print(f"\n✓ Reached grasp position! Final distance: {distance:.4f}m")
            print(f"  Grasp position: {grasp_pos}")
            print(f"  Final EE position: {current_pos}")
            break
        
        # Normalize delta to unit vector, then scale to [-1, 1] range
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 5.0, -1, 1)
        else:
            action_pos = np.zeros(3)
        
        # Create action (position delta + no rotation change + gripper open)
        action = np.concatenate([
            action_pos,     # [dx, dy, dz] normalized
            [0, 0, 0],      # [droll, dpitch, dyaw] - no rotation change
            [1]             # gripper open
        ])
        
        # Process action through controller
        controller.process_action(action)
        
        # Step simulation multiple times for smoother motion
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Capture camera images every 10 steps
        if step % 10 == 0:
            images = camera.capture_all()
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"  Step {step:3d}: Distance to grasp: {distance:.4f}m")
    
    # Final position check
    final_pos, _ = robot.get_end_effector_pose()
    final_distance = np.linalg.norm(grasp_pos - final_pos)
    
    print(f"\n=== Movement Complete ===")
    print(f"Final EE position: {final_pos}")
    print(f"Grasp position: {grasp_pos}")
    print(f"Final distance: {final_distance:.4f}m")
    
    if final_distance < 0.05:
        print("✓ Successfully reached grasp position!")
        
        # Close gripper to grasp object - adaptive approach
        print("\n=== Closing Gripper (Adaptive) ===")
        robot.set_gripper(-1)  # -1 = close
        
        # Adaptive gripper closing - detect contact via force/torque feedback
        print("  Closing gripper adaptively (force-based detection)...")
        max_close_steps = 1000
        check_interval = 1  # Check every step for immediate detection
        torque_threshold = 2.5  # Torque threshold indicating firm grasp (N·m)
        min_steps = 60  # Don't check in first 60 steps (initial movement)
        
        for close_step in range(max_close_steps):
            p.stepSimulation()
            time.sleep(1./240.)
            
            # Check gripper torque every step after initial movement
            if (close_step + 1) % check_interval == 0 and close_step >= min_steps:
                # Get joint states for both gripper joints
                gripper_torques = []
                gripper_positions = []
                for joint_idx in robot.gripper_joint_indices:
                    joint_state = p.getJointState(robot.robot_id, joint_idx)
                    gripper_positions.append(joint_state[0])  # Position
                    gripper_torques.append(abs(joint_state[3]))  # Applied motor torque (absolute value)
                
                max_torque = max(gripper_torques)
                avg_position = np.mean(gripper_positions)
                
                if (close_step + 1) % 50 == 0:  # Print every 50 steps
                    print(f"  Step {close_step + 1}: Position: {avg_position:.4f}, Max torque: {max_torque:.3f} N·m")
                
                # Check if torque exceeds threshold (object contact detected)
                if max_torque > torque_threshold:
                    print(f"✓ Object contact detected - torque {max_torque:.3f} N·m > {torque_threshold} N·m")
                    print(f"  Detected at step {close_step + 1}")
                    print(f"  Gripper position: {avg_position:.4f}")
                    
                    # Immediately stop gripper by holding current position
                    for i, joint_idx in enumerate(robot.gripper_joint_indices):
                        current_pos = p.getJointState(robot.robot_id, joint_idx)[0]
                        p.setJointMotorControl2(
                            robot.robot_id, joint_idx,
                            p.POSITION_CONTROL,
                            targetPosition=current_pos,  # Hold current position
                            force=150,
                            maxVelocity=0  # Stop moving
                        )
                    
                    # Give it a moment to settle
                    for _ in range(10):
                        p.stepSimulation()
                        time.sleep(1./240.)
                    break
        
        final_gripper_state = robot.get_gripper_state()
        print(f"✓ Gripper closed adaptively - Final state: {final_gripper_state}")
        
        # Save the exact gripper positions to hold during carry
        grasp_positions = []
        for joint_idx in robot.gripper_joint_indices:
            grasp_positions.append(p.getJointState(robot.robot_id, joint_idx)[0])
        print(f"  Holding gripper at positions: {grasp_positions}")
        
        # First lift up to avoid collisions with other objects
        print(f"\n=== Lifting Object ===")
        current_pos, current_orn = robot.get_end_effector_pose()
        lift_height = 0.15  # Lift 15cm above current position
        lift_pos = current_pos.copy()
        lift_pos[2] += lift_height
        print(f"Lifting to: {lift_pos}")
        
        # Move up
        num_lift_steps = 100
        for step in range(num_lift_steps):
            current_pos, current_orn = robot.get_end_effector_pose()
            
            delta = lift_pos - current_pos
            distance = np.linalg.norm(delta)
            
            if distance < distance_threshold:
                print(f"✓ Lifted! Height: {current_pos[2]:.4f}m")
                break
            
            if distance > 0.001:
                delta_normalized = delta / distance
                action_pos = np.clip(delta_normalized * 1.5, -1, 1)
            else:
                action_pos = np.zeros(3)
            
            action = np.concatenate([
                action_pos,
                [0, 0, 0],
                [-1]  # gripper closed
            ])
            
            controller.process_action(action)
            
            # Maintain exact gripper position during lift
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
        
        # Move to box position (15cm above middle of container box)
        box_center = np.array([0.15, 0.15, table_height])
        drop_pos = box_center.copy()
        drop_pos[2] += 0.15  # 15cm above box
        
        print(f"\n=== Moving to Drop Position ===")
        print(f"Drop position: {drop_pos}")
        
        # Move to drop position
        num_steps_to_box = 200
        for step in range(num_steps_to_box):
            current_pos, current_orn = robot.get_end_effector_pose()
            
            delta = drop_pos - current_pos
            distance = np.linalg.norm(delta)
            
            # Check if reached drop position
            if distance < distance_threshold:
                print(f"\n✓ Reached drop position! Final distance: {distance:.4f}m")
                break
            
            # Normalize and scale delta - SLOWER during carry to avoid slippage
            if distance > 0.001:
                delta_normalized = delta / distance
                action_pos = np.clip(delta_normalized * 1.5, -1, 1)  # Much slower: 1.5 instead of 5.0
            else:
                action_pos = np.zeros(3)
            
            # Create action with gripper closed
            action = np.concatenate([
                action_pos,
                [0, 0, 0],
                [-1]  # gripper closed
            ])
            
            controller.process_action(action)
            
            # Maintain exact gripper position during carry (don't let it slip!)
            for i, joint_idx in enumerate(robot.gripper_joint_indices):
                if i < len(grasp_positions):
                    p.setJointMotorControl2(
                        robot.robot_id, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=grasp_positions[i],
                        force=300,  # Very high force to maintain grip during movement
                        maxVelocity=0  # Don't move
                    )
            
            for _ in range(4):
                p.stepSimulation()
                time.sleep(1./240.)
            
            if step % 10 == 0:
                images = camera.capture_all()
            
            if step % 20 == 0:
                print(f"  Step {step:3d}: Distance to drop: {distance:.4f}m")
        
        # Wait 1 second at drop position before opening gripper
        print("\n=== Waiting at drop position ===")
        print("  Holding position for 1 second...")
        for _ in range(240):  # 240 steps = 1 second at 240Hz
            # Maintain gripper hold during wait
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
        print("✓ Wait complete")
        
        # Open gripper to release object
        print("\n=== Opening Gripper ===")
        robot.set_gripper(1)  # 1 = open
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
        print("✓ Gripper opened - object released!")
        
        # Let object fall into box
        print("\n=== Letting object settle ===")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("✓ Task complete!")
    else:
        print("⚠ Did not fully reach target")
    
    # Keep simulation running
    print("\n=== Simulation Running ===")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    p.disconnect()


if __name__ == "__main__":
    main()
