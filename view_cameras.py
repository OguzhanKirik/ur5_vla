"""
Visualize the 3 camera views from the UR5 environment with joint control sliders
"""

import numpy as np
import cv2
from pybullet_env import UR5GraspEnv
import pybullet as p
import time


def main():
    print("Starting UR5 Grasp Environment with Camera Visualization and Joint Control...")
    print("Press 'q' to quit")
    print("-" * 70)
    
    # Create environment with GUI
    env = UR5GraspEnv(gui=True, fps=30)
    
    # Reset and get initial observation
    obs = env.reset()
    
    # Home position in degrees (matching the reset_robot function)
    home_positions_rad = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]
    home_positions_deg = [int(np.degrees(angle)) for angle in home_positions_rad]
    
    # Gripper home position (closed = 0, open = depends on gripper model)
    gripper_home = 0
    
    print("\nCamera Configuration:")
    print(f"  Total cameras: {len(obs['images'])}")
    for key, img in obs['images'].items():
        print(f"  {key}: {img.shape}")
    
    print(f"\nRobot has {len(env.arm_joint_indices)} controllable joints")
    print(f"Home position (degrees): {home_positions_deg}")
    
    # Create control window with sliders for each joint
    control_window = "Joint Control"
    cv2.namedWindow(control_window)
    
    # Create trackbars for arm joints (6 joints)
    for i in range(6):
        # Range: -180 to 180 degrees, offset by 180 to make it 0-360 for trackbar
        initial_value = home_positions_deg[i] + 180
        cv2.createTrackbar(f"Joint {i+1}", control_window, initial_value, 360, lambda x: None)
    
    # Create trackbars for gripper joints (2 joints, but typically controlled together)
    cv2.createTrackbar("Gripper L", control_window, gripper_home + 50, 100, lambda x: None)
    cv2.createTrackbar("Gripper R", control_window, gripper_home + 50, 100, lambda x: None)
    
    # Create trackbars for wrist camera position (offset from end-effector in cm)
    # Range: -50 to 50 cm, trackbar 0-100, subtract 50
    # Defaults: X=35, Y=55, Z=50
    cv2.createTrackbar("Cam Offset X", control_window, 35, 100, lambda x: None)  # -15 cm
    cv2.createTrackbar("Cam Offset Y", control_window, 55, 100, lambda x: None)  # 5 cm
    cv2.createTrackbar("Cam Offset Z", control_window, 50, 100, lambda x: None)  # 0 cm
    
    # Create trackbars for wrist camera target (what it looks at, relative to ee in cm)
    # Defaults: X=50, Y=85, Z=50
    cv2.createTrackbar("Cam Target X", control_window, 50, 100, lambda x: None)  # 0 cm
    cv2.createTrackbar("Cam Target Y", control_window, 85, 100, lambda x: None)  # 35 cm forward
    cv2.createTrackbar("Cam Target Z", control_window, 50, 100, lambda x: None)  # 0 cm
    
    # Create trackbar for camera yaw (rotate left/right)
    cv2.createTrackbar("Cam Yaw", control_window, 180, 360, lambda x: None)  # 0-360 deg, default 180 (0 deg)
    
    # Create trackbar for camera pitch (tilt up/down)
    cv2.createTrackbar("Cam Pitch", control_window, 180, 360, lambda x: None)  # 0-360 deg, default 180 (0 deg)
    
    # Create trackbar for camera roll (rotation around view axis)
    cv2.createTrackbar("Cam Roll", control_window, 180, 360, lambda x: None)  # 0-360 deg, default 180 (0 deg)
    
    print("\nDisplaying camera views with joint control... (press 'q' in any window to exit)")
    print("Wrist camera sliders: Offset and Target are in cm relative to end-effector")
    print("Cam Yaw: Rotate camera left/right (-180 to 180 degrees)")
    print("Cam Pitch: Tilt camera up/down (-180 to 180 degrees)")
    print("Cam Roll: Rotate camera view (-180 to 180 degrees)")
    
    step = 0
    try:
        while True:
            # Read joint angles from trackbars
            joint_angles = []
            for i in range(6):
                value = cv2.getTrackbarPos(f"Joint {i+1}", control_window)
                # Convert from 0-360 back to -180 to 180
                angle_deg = value - 180
                angle_rad = np.radians(angle_deg)
                joint_angles.append(angle_rad)
            
            # Read gripper positions
            gripper_l = cv2.getTrackbarPos("Gripper L", control_window) - 50
            gripper_r = cv2.getTrackbarPos("Gripper R", control_window) - 50
            gripper_l_rad = np.radians(gripper_l)
            gripper_r_rad = np.radians(gripper_r)
            
            # Apply joint positions to robot
            for i, joint_idx in enumerate(env.arm_joint_indices[:6]):
                p.setJointMotorControl2(
                    env.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_angles[i],
                    force=500
                )
            
            # Apply gripper positions (if gripper joints exist)
            if len(env.arm_joint_indices) >= 8:
                p.setJointMotorControl2(
                    env.robot_id, env.arm_joint_indices[6],
                    p.POSITION_CONTROL,
                    targetPosition=gripper_l_rad,
                    force=50
                )
                p.setJointMotorControl2(
                    env.robot_id, env.arm_joint_indices[7],
                    p.POSITION_CONTROL,
                    targetPosition=gripper_r_rad,
                    force=50
                )
            
            # Step physics
            p.stepSimulation()
            
            # Read wrist camera parameters from sliders
            cam_offset_x = (cv2.getTrackbarPos("Cam Offset X", control_window) - 50) / 100.0  # Convert to meters
            cam_offset_y = (cv2.getTrackbarPos("Cam Offset Y", control_window) - 50) / 100.0
            cam_offset_z = (cv2.getTrackbarPos("Cam Offset Z", control_window) - 50) / 100.0
            
            cam_target_x = (cv2.getTrackbarPos("Cam Target X", control_window) - 50) / 100.0
            cam_target_y = (cv2.getTrackbarPos("Cam Target Y", control_window) - 50) / 100.0
            cam_target_z = (cv2.getTrackbarPos("Cam Target Z", control_window) - 50) / 100.0
            
            # Read camera yaw, pitch, and roll (0-360 degrees, convert to radians offset by 180)
            cam_yaw_deg = cv2.getTrackbarPos("Cam Yaw", control_window) - 180
            cam_yaw_rad = np.radians(cam_yaw_deg)
            cam_pitch_deg = cv2.getTrackbarPos("Cam Pitch", control_window) - 180
            cam_pitch_rad = np.radians(cam_pitch_deg)
            cam_roll_deg = cv2.getTrackbarPos("Cam Roll", control_window) - 180
            cam_roll_rad = np.radians(cam_roll_deg)
            
            # Override the environment's camera parameters temporarily
            env.wrist_cam_offset = [cam_offset_x, cam_offset_y, cam_offset_z]
            env.wrist_cam_target = [cam_target_x, cam_target_y, cam_target_z]
            env.wrist_cam_yaw = cam_yaw_rad
            env.wrist_cam_pitch = cam_pitch_rad
            env.wrist_cam_roll = cam_roll_rad
            env.wrist_cam_roll = cam_roll_rad
            
            # Get current camera images
            obs['images'] = env.get_camera_images()
            obs['state'] = env.get_state()
            
            # Display each camera view
            for key, img in obs['images'].items():
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Add text label
                clean_name = key.replace('observation.images.', '')
                cv2.putText(img_bgr, clean_name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add frame counter
                cv2.putText(img_bgr, f"Frame: {step}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show image
                cv2.imshow(clean_name, img_bgr)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            step += 1
            
            # Small delay for smoother display
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        env.close()
        print("\nCamera visualization complete!")


if __name__ == "__main__":
    main()
