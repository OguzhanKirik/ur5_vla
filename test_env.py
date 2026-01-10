"""
Test script for UR5 Grasping Environment

Run this to verify the environment is working correctly.
"""

import numpy as np
from pybullet_env import UR5GraspEnv
import time


def test_basic_functionality():
    """Test basic environment operations"""
    print("Test 1: Basic Functionality")
    print("-" * 50)
    
    env = UR5GraspEnv(gui=True, fps=30)
    
    # Test reset
    obs = env.reset()
    print("✓ Environment reset successful")
    print(f"  State shape: {obs['state'].shape}")
    assert obs['state'].shape == (8,), "State shape should be (8,)"
    
    # Test camera images
    print(f"  Number of cameras: {len(obs['images'])}")
    assert len(obs['images']) == 3, "Should have 3 camera views"
    
    for key, img in obs['images'].items():
        print(f"    {key}: {img.shape}")
        assert img.shape == (480, 640, 3), f"Image shape should be (480, 640, 3), got {img.shape}"
    
    # Test action execution
    action = np.zeros(7)
    obs, reward, done, info = env.step(action)
    print("✓ Action execution successful")
    print(f"  Reward: {reward:.4f}")
    print(f"  Info: {info}")
    
    env.close()
    print("\n✓ Test 1 passed!\n")


def test_random_rollout():
    """Test random action rollout"""
    print("Test 2: Random Action Rollout")
    print("-" * 50)
    
    env = UR5GraspEnv(gui=True, fps=30)
    obs = env.reset()
    
    num_steps = 50
    rewards = []
    
    print(f"Running {num_steps} random steps...")
    for step in range(num_steps):
        action = np.random.randn(7) * 0.5
        action = np.clip(action, -1, 1)
        
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: reward={reward:.4f}, dist={info['distance_to_object']:.4f}")
    
    print(f"\nRollout statistics:")
    print(f"  Mean reward: {np.mean(rewards):.4f}")
    print(f"  Std reward: {np.std(rewards):.4f}")
    print(f"  Min reward: {np.min(rewards):.4f}")
    print(f"  Max reward: {np.max(rewards):.4f}")
    
    env.close()
    print("\n✓ Test 2 passed!\n")


def test_state_consistency():
    """Test that state is consistent across steps"""
    print("Test 3: State Consistency")
    print("-" * 50)
    
    env = UR5GraspEnv(gui=False, fps=30)  # Headless for speed
    obs = env.reset()
    
    # Apply zero action (should maintain state)
    zero_action = np.zeros(7)
    
    states = []
    for _ in range(10):
        obs, _, _, _ = env.step(zero_action)
        states.append(obs['state'].copy())
    
    # Check that states don't diverge much with zero action
    state_changes = [np.linalg.norm(states[i+1] - states[i]) for i in range(len(states)-1)]
    max_change = max(state_changes)
    
    print(f"  Max state change with zero action: {max_change:.6f}")
    assert max_change < 0.1, "State should be relatively stable with zero action"
    print("✓ State is consistent")
    
    env.close()
    print("\n✓ Test 3 passed!\n")


def test_action_space():
    """Test action space boundaries"""
    print("Test 4: Action Space")
    print("-" * 50)
    
    env = UR5GraspEnv(gui=False, fps=30)
    obs = env.reset()
    
    # Test extreme actions
    print("  Testing extreme positive action...")
    extreme_pos = np.ones(7) * 2.0
    obs1, _, _, _ = env.step(extreme_pos)
    
    print("  Testing extreme negative action...")
    extreme_neg = np.ones(7) * -2.0
    obs2, _, _, _ = env.step(extreme_neg)
    
    print("  Testing mixed action...")
    mixed = np.array([1, -1, 0.5, -0.5, 0, 1, -1])
    obs3, _, _, _ = env.step(mixed)
    
    print("✓ Environment handles various actions without crashing")
    
    env.close()
    print("\n✓ Test 4 passed!\n")


def interactive_test():
    """Interactive test with keyboard control (basic)"""
    print("Test 5: Interactive Control")
    print("-" * 50)
    print("Environment running in GUI mode...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    env = UR5GraspEnv(gui=True, fps=30)
    obs = env.reset()
     
    try:
        step = 0
        while True:
            # Simple sinusoidal motion for demonstration
            t = step * 0.02
            action = np.array([
                0.5 * np.sin(t),
                0.3 * np.cos(t),
                0.2 * np.sin(2*t),
                0.1 * np.cos(2*t),
                0.0,
                0.0,
                0.0
            ])
            
            obs, reward, done, info = env.step(action)
            step += 1
            
            if step % 30 == 0:
                print(f"Step {step}: reward={reward:.4f}")
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        env.close()
        print("\n✓ Test 5 complete!\n")


def main():
    """Run all tests"""
    print("="*70)
    print("UR5 Grasp Environment Test Suite")
    print("="*70)
    print()
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Random Rollout", test_random_rollout),
        ("State Consistency", test_state_consistency),
        ("Action Space", test_action_space),
    ]
    
    for i, (name, test_func) in enumerate(tests, 1):
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Test {i} ({name}) failed with error:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("="*70)
    print("All tests passed! ✓")
    print("="*70)
    
    # Ask if user wants interactive test
    try:
        response = input("\nRun interactive test? (y/n): ")
        if response.lower() == 'y':
            interactive_test()
    except KeyboardInterrupt:
        print("\n\nSkipped interactive test")


if __name__ == "__main__":
    main()
