"""
Grasp Objects Script - Select and grasp objects interactively

Demonstrates:
- Interactive object selection (individual or all)
- Pick and place workflow for selected objects
- Two-phase approach (above then descend)
- Adaptive torque-based grasping
"""

import numpy as np
import pybullet as p
import pybullet_data
import time

from components import UR5RobotComponent, CameraComponent, ObjectsComponent, RobotController


def grasp_and_place_object(robot, controller, camera, target_object_id, container_pos, table_height, distance_threshold=0.02):
    """
    Grasp a single object and place it in the container.
    
    Args:
        robot: UR5RobotComponent instance
        controller: RobotController instance
        camera: CameraComponent instance
        target_object_id: PyBullet ID of object to grasp
        container_pos: [x, y, z] position of container
        table_height: Height of table surface
        distance_threshold: Distance threshold for reaching targets
    """
    # Get target object position
    target_pos, _ = p.getBasePositionAndOrientation(target_object_id)
    target_pos = np.array(target_pos)
    print(f"\nTarget object at: {target_pos}")
    
    # Phase 1: Move to position 20cm above object
    approach_pos = target_pos.copy()
    approach_pos[2] += 0.20  # 20cm above object
    
    print(f"\n=== Phase 1: Moving Above Target ===")
    print(f"Approach position (20cm above): {approach_pos}")
    
    num_steps = 200
    for step in range(num_steps):
        current_pos, current_orn = robot.get_end_effector_pose()
        
        delta = approach_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            print(f"✓ Reached approach position! Distance: {distance:.4f}m")
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
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: Distance: {distance:.4f}m")
    
    # Phase 2: Descend to grasp position (1cm above object)
    grasp_pos = target_pos.copy()
    grasp_pos[2] += 0.01  # 1cm above object
    
    print(f"\n=== Phase 2: Descending to Grasp Position ===")
    print(f"Grasp position (1cm above): {grasp_pos}")
    
    for step in range(num_steps):
        current_pos, current_orn = robot.get_end_effector_pose()
        
        delta = grasp_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            print(f"✓ Reached grasp position! Distance: {distance:.4f}m")
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
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: Distance: {distance:.4f}m")
    
    # Close gripper with adaptive torque detection
    print("\n=== Closing Gripper (Adaptive) ===")
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
            
            if (close_step + 1) % 50 == 0:
                print(f"  Step {close_step + 1}: Max torque: {max_torque:.3f} N·m")
            
            if max_torque > torque_threshold:
                print(f"✓ Object grasped - torque {max_torque:.3f} N·m")
                
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
    
    # Save gripper positions for holding during carry
    grasp_positions = []
    for joint_idx in robot.gripper_joint_indices:
        grasp_positions.append(p.getJointState(robot.robot_id, joint_idx)[0])
    
    # Lift object
    print(f"\n=== Lifting Object ===")
    current_pos, current_orn = robot.get_end_effector_pose()
    lift_pos = current_pos.copy()
    lift_pos[2] += 0.15  # 15cm up
    
    for step in range(100):
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
    
    # Move to drop position
    drop_pos = np.array([container_pos[0], container_pos[1], table_height + 0.15])
    
    print(f"\n=== Moving to Drop Position ===")
    print(f"Drop position: {drop_pos}")
    
    for step in range(200):
        current_pos, current_orn = robot.get_end_effector_pose()
        
        delta = drop_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < distance_threshold:
            print(f"✓ Reached drop position! Distance: {distance:.4f}m")
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
        
        if step % 10 == 0:
            camera.capture_all()
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: Distance: {distance:.4f}m")
    
    # Wait 1 second
    print("\n=== Waiting 1 second ===")
    for _ in range(240):
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
    
    # Open gripper
    print("\n=== Opening Gripper ===")
    # Open gripper wider for release
    for joint_idx in robot.gripper_joint_indices:
        p.setJointMotorControl2(
            robot.robot_id, joint_idx,
            p.POSITION_CONTROL,
            targetPosition=1.0,  # Open wider than normal (0.8)
            force=200,
            maxVelocity=15.0  # Fast opening for quick release
        )
    for _ in range(30):  # Shorter wait - gripper opens fast enough
        p.stepSimulation()
        time.sleep(1./240.)
    print("✓ Object released!")
    
    # Let settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    print("✓ Pick and place complete!")
    
    # Return to home position smoothly
    print("\n=== Returning to Home Position ===")
    home_ee_pos = np.array([0.5, 0.4, table_height + 0.3])  # Safe home position above table
    
    for step in range(300):
        current_pos, current_orn = robot.get_end_effector_pose()
        
        delta = home_ee_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance < 0.02:
            print(f"✓ Reached home position!")
            break
        
        if distance > 0.001:
            delta_normalized = delta / distance
            action_pos = np.clip(delta_normalized * 2.0, -1, 1)  # Moderate speed
        else:
            action_pos = np.zeros(3)
        
        action = np.concatenate([action_pos, [0, 0, 0], [1]])  # gripper open
        controller.process_action(action)
        
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.)
        
        if step % 50 == 0:
            print(f"  Returning... Distance: {distance:.4f}m")
    
    print("✓ Returned to home\n")


def main():
    # Initialize PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
    print("\n=== Setting up Environment ===")
    
    # Create objects component
    objects = ObjectsComponent()
    
    # Create table
    table_pos = [0.5, 0.5, 0.18]
    table_size = [0.5, 0.5, 0.18]
    table_id = objects.create_table(position=table_pos, size=table_size)
    
    table_aabb = p.getAABB(table_id)
    table_height = table_aabb[1][2]
    print(f"✓ Table created at Z = {table_height:.4f}")
    
    # Create robot
    robot_pos = [0.5, 0, table_height]
    robot_orn = p.getQuaternionFromEuler([0, 0, -np.pi/2])
    
    robot = UR5RobotComponent(
        position=robot_pos,
        orientation=robot_orn,
        use_fixed_base=True
    )
    robot.load()
    print(f"✓ Robot loaded at {robot.position}")
    
    # Spawn graspable objects
    spawned_ids = objects.spawn_graspable_objects(
        table_height=table_height,
        workspace_bounds=None
    )
    print(f"✓ Spawned {len(spawned_ids)} objects")
    
    # Object names for reference
    object_names = ["red box", "green sphere", "blue cylinder", "blue box"]
    
    # Create container box
    container_pos = [0.15, 0.15, table_height]
    objects.create_container_box(
        position=container_pos,
        size=[0.15, 0.15, 0.08],
        wall_thickness=0.01,
        color=[0.2, 0.2, 0.5, 1]
    )
    print(f"✓ Container box created at {container_pos}")
    
    # Create controller
    controller = RobotController(
        robot_component=robot,
        control_mode="inverse_kinematics",
        ik_xyz_delta=0.05,
        ik_rpy_delta=0.05
    )
    print(f"✓ Controller initialized")
    
    # Create cameras
    camera = CameraComponent()
    camera.add_fixed_camera(
        name="main_view",
        eye_position=[0.5, 1.0, 1.0],
        target_position=[0.5, 0.5, 0.36]
    )
    camera.add_fixed_camera(
        name="top_view",
        eye_position=[0.5, 1.0, 1.0],
        target_position=[0.5, 0.5, 0.36]
    )
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
    
    # Track grasped objects
    grasped_objects = set()
    
    # Get object positions after settling
    print("\n=== Available Objects ===")
    for i, (obj_id, name) in enumerate(zip(spawned_ids, object_names)):
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        print(f"{i}: {name} at {pos}")
    
    # Main interaction loop
    print("\n=== Starting Interactive Grasping ===")
    print("Enter 'quit' or 'exit' to stop")
    
    while True:
        # Get user selection
        print("\n=== Object Selection ===")
        available_objects = [i for i in range(len(spawned_ids)) if i not in grasped_objects]
        if not available_objects:
            print("No objects remaining on table!")
            print("\n=== Resetting Environment ===")
            
            # Clear objects
            objects.clear_objects()
            grasped_objects.clear()
            
            # Respawn objects with randomized positions
            spawned_ids = objects.spawn_graspable_objects(
                table_height=table_height,
                workspace_bounds=[[0.3, 0.7], [0.3, 0.7]],
                randomize=True
            )
            
            # Let physics settle
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1./240.)
            
            print("✓ Environment reset with new object positions!")
            print("\n=== Available Objects ===")
            for i, (obj_id, name) in enumerate(zip(spawned_ids, object_names)):
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                print(f"{i}: {name} at {pos}")
            continue
        
        print(f"Available objects: {available_objects}")
        print(f"Enter object number or 'all' to grasp all remaining objects:")
        user_input = input("> ").strip().lower()
        
        # Check for exit
        if user_input in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        # Determine which objects to grasp
        if user_input == 'all':
            objects_to_grasp = available_objects
            print(f"Selected: All remaining objects")
        else:
            try:
                obj_idx = int(user_input)
                if obj_idx in available_objects:
                    objects_to_grasp = [obj_idx]
                    print(f"Selected: {object_names[obj_idx]}")
                elif obj_idx in grasped_objects:
                    print(f"Object {obj_idx} already grasped!")
                    continue
                else:
                    print(f"Invalid selection. Available: {available_objects}")
                    continue
            except ValueError:
                print(f"Invalid input. Please enter a number or 'all'")
                continue
        
        # Execute grasping for selected objects
        print(f"\n=== Starting Pick and Place Sequence ===")
        for obj_idx in objects_to_grasp:
            print(f"\n{'='*60}")
            print(f"Grasping object {obj_idx}: {object_names[obj_idx]}")
            print(f"{'='*60}")
            
            grasp_and_place_object(
                robot=robot,
                controller=controller,
                camera=camera,
                target_object_id=spawned_ids[obj_idx],
                container_pos=container_pos,
                table_height=table_height
            )
            
            # Mark as grasped
            grasped_objects.add(obj_idx)
        
        print(f"\n{'='*60}")
        print(f"✓ All selected objects processed!")
        print(f"{'='*60}")
    
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
