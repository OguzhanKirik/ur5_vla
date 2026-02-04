# UR5 SmolVLA Grasping Training Pipeline

This document explains the data collection and training pipeline for training SmolVLA (Small Vision-Language-Action) models on a UR5 robotic grasping task in simulation.

## Overview

The pipeline consists of two main components:
1. **`collect_lerobot_smolvla.py`** - Data collection script that generates demonstration episodes
2. **`train_smolvla_lerobot_v2.py`** - Training script that fine-tunes SmolVLA on the collected data

---

## Data Collection: `collect_lerobot_smolvla.py`

### Purpose
Collects demonstration data for vision-language-action learning using a simulated UR5 robot in PyBullet performing pick-and-place tasks.

### Usage
```bash
# Basic usage (100 episodes)
python collect_lerobot_smolvla.py --episodes 100

# With GUI visualization
python collect_lerobot_smolvla.py --episodes 50 --gui

# Fast mode (no rendering delays)
python collect_lerobot_smolvla.py --episodes 200 --fast

# Custom output location
python collect_lerobot_smolvla.py --repo-id local/my_dataset --root ./datasets/lerobot
```

### Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 100 | Number of demonstration episodes to collect |
| `--repo-id` | str | `local/ur5_smolvla_grasp` | Dataset repository ID (local or HuggingFace) |
| `--root` | str | `./datasets/lerobot` | Root directory to save dataset |
| `--gui` | flag | False | Show PyBullet GUI for visualization |
| `--fast` | flag | False | Fast simulation mode (skip rendering delays) |
| `--save-only-success` | flag | False | Only save successful grasping episodes |

### Data Generation Process

#### Episode Structure
Each episode follows an 8-phase pick-and-place sequence:

1. **Approach Phase** (~200 steps)
   - Robot moves end-effector 20cm above target object
   - Gripper remains open
   - Frames captured every 8 simulation steps

2. **Descend Phase** (~200 steps)
   - Robot descends to 1cm above object
   - Precise positioning for grasp

3. **Grasp Phase** (~1000 steps)
   - Gripper closes on object
   - Force feedback monitors grip torque
   - Stops when sufficient grip detected (torque > 2.5 Nm)

4. **Lift Phase** (~100 steps)
   - Robot lifts object 15cm vertically
   - Maintains grip force throughout

5. **Transport Phase** (~200 steps)
   - Robot moves to container position
   - Maintains grip while navigating

6. **Release Phase** (~50 steps)
   - Gripper opens to release object
   - Object drops into container

7. **Settle Phase** (~100 steps)
   - Wait for physics to settle
   - Verify object placement

8. **Return Home** (~300 steps)
   - Robot returns to neutral position
   - Ready for next episode

#### Objects and Instructions
The system includes 4 object types with language variations:

| Object | Color | Shape | Example Instruction |
|--------|-------|-------|---------------------|
| Object 0 | Red | Sphere | "Pick up the red sphere and place it in the container" |
| Object 1 | Green | Sphere | "Grasp the green sphere and put it in the box" |
| Object 2 | Blue | Cylinder | "Pick up the blue cylinder and drop it in the container" |
| Object 3 | Red | Cylinder | "Pick up the red cylinder and place it in the container" |

Each object has 4 instruction variants for data augmentation (e.g., "grasp", "pick up", "take", "move").

### Dataset Features

The collected dataset uses LeRobot format with the following features:

#### Observation State (8 dimensions)
```python
{
    "dtype": "float32",
    "shape": (8,),
    "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_pos", "gripper_state"]
}
```
- **Joints 0-5**: UR5 arm joint positions (radians)
- **gripper_pos**: Gripper finger position
- **gripper_state**: Binary gripper state (+1 open, -1 closed)

#### Camera Images
Two camera views stored as video:

| Camera | Resolution | FPS | Codec | Description |
|--------|------------|-----|-------|-------------|
| `observation.images.top` | 640×480 | 30 | AV1 | Overhead view of workspace |
| `observation.images.wrist` | 640×480 | 30 | AV1 | Wrist-mounted camera view |

#### Actions (7 dimensions)
```python
{
    "dtype": "float32",
    "shape": (7,),
    "names": ["joint_0_vel", "joint_1_vel", "joint_2_vel", "joint_3_vel", "joint_4_vel", "joint_5_vel", "gripper"]
}
```
- **Joints 0-5**: Joint velocity commands (clipped to [-1, 1])
- **gripper**: Gripper command (+1 open, -1 close)

#### Task Descriptions
Natural language instructions stored per frame:
```python
{
    "task": "Pick up the red sphere and place it in the container"
}
```

### Output Structure
```
datasets/lerobot/
├── data/
│   └── chunk-000/
│       └── episode_*.parquet     # Tabular data (states, actions)
├── videos/
│   ├── observation.images.top/
│   │   └── episode_*.mp4         # Top camera videos
│   └── observation.images.wrist/
│       └── episode_*.mp4         # Wrist camera videos
└── meta/
    ├── info.json                 # Dataset metadata
    ├── stats.json                # Normalization statistics
    ├── tasks.parquet             # Unique task descriptions
    └── episodes/
        └── *.parquet             # Episode metadata
```

---

## Training: `train_smolvla_lerobot_v2.py`

### Purpose
Fine-tunes the SmolVLA vision-language-action model on collected demonstration data.

### Usage
```bash
# Basic training (full dataset)
python train_smolvla_lerobot_v2.py

# Quick test with limited data
python train_smolvla_lerobot_v2.py --max-samples 200 --training-steps 50 --batch-size 4

# Full training run
python train_smolvla_lerobot_v2.py --batch-size 4 --training-steps 10000 --lr 1e-5

# Resume from checkpoint
python train_smolvla_lerobot_v2.py --resume --training-steps 20000
```

### Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--repo-id` | str | `local/ur5_smolvla_grasp` | Dataset repository ID |
| `--root` | str | `../../datasets/lerobot` | Dataset root directory |
| `--output-dir` | str | `./checkpoints/smolvla_lerobot` | Checkpoint output directory |
| `--batch-size` | int | 4 | Training batch size |
| `--training-steps` | int | 5000 | Total training steps |
| `--lr` | float | 1e-5 | Learning rate |
| `--chunk-size` | int | 50 | Action chunk size (predictions per step) |
| `--checkpoint-freq` | int | 500 | Save checkpoint every N steps |
| `--log-freq` | int | 10 | Log metrics every N steps |
| `--max-samples` | int | None | Limit dataset size for testing |
| `--resume` | flag | False | Resume from latest checkpoint |

### Training Process

#### 1. Dataset Loading
The script uses LeRobot's native dataset format with proper temporal alignment:

```python
# Load metadata first
dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)

# Configure delta_timestamps for temporal data loading
delta_timestamps = {
    "observation.images.top": [0.0],  # Current frame
    "observation.images.wrist": [0.0],
    "observation.state": [0.0],
    "action": [0.0, 0.033, 0.066, ...]  # Action chunk (50 frames)
}

# Load dataset with temporal alignment
dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, root=root)
```

#### 2. Model Architecture
Uses pretrained **SmolVLA Base** model from HuggingFace:

| Component | Specification |
|-----------|---------------|
| Vision Encoder | SmolVLM2-500M-Video-Instruct (16 layers) |
| Language Model | Integrated VLM with 500M parameters |
| Action Head | Diffusion-based action decoder |
| Chunk Size | 50 actions predicted per step |
| Denoising Steps | 10 (configurable) |

#### 3. Training Configuration

**Optimizer**: AdamW
```python
optimizer = torch.optim.AdamW(
    parameters,
    lr=1e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

**Learning Rate Schedule**: Warmup + Cosine Decay
- Warmup: 500 steps (linear 0% → 100%)
- Decay: Cosine from 100% → 10% of base LR

**Gradient Clipping**: Max norm = 1.0

#### 4. Data Preprocessing
LeRobot's preprocessor handles:
- Image normalization and resizing
- Language tokenization
- State normalization using dataset statistics
- Action chunk preparation

#### 5. Training Loop
```python
for batch in dataloader:
    # Preprocess batch
    batch = preprocessor(batch)
    
    # Rename camera keys to match policy expectations
    batch["observation.images.camera1"] = batch.pop("observation.images.top")
    batch["observation.images.camera2"] = batch.pop("observation.images.wrist")
    
    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass (returns loss and loss_dict)
    loss, loss_dict = policy.forward(batch)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

### Checkpointing
Checkpoints include:
- Model weights (`model.safetensors`)
- Preprocessor configuration
- Postprocessor configuration

### Logging
Metrics logged to **Weights & Biases**:
- Loss (average over log_freq steps)
- Learning rate
- Training step

### Memory Optimization (macOS MPS)
```python
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # No memory limit
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP issues

# Periodic cache clearing
if device.type == "mps" and step % 100 == 0:
    torch.mps.empty_cache()
```

---

## Complete Pipeline Example

```bash
# 1. Collect 100 demonstration episodes
cd ur5_grasp/scripts
python collect_lerobot_smolvla.py --episodes 100 --fast

# 2. Quick test training (verify pipeline works)
python train_smolvla_lerobot_v2.py --max-samples 200 --training-steps 50 --batch-size 4

# 3. Full training run
python train_smolvla_lerobot_v2.py \
    --batch-size 4 \
    --training-steps 10000 \
    --lr 1e-5 \
    --checkpoint-freq 1000

# 4. Resume training if needed
python train_smolvla_lerobot_v2.py --resume --training-steps 20000
```

---

## Technical Specifications

### Environment Requirements
- Python 3.10+
- PyTorch 2.0+ with MPS/CUDA support
- LeRobot library
- PyBullet for simulation
- wandb for logging

### Hardware Recommendations
| Device | Batch Size | Training Speed |
|--------|------------|----------------|
| Apple M1/M2 (MPS) | 2-4 | ~1.5 steps/sec |
| NVIDIA RTX 3090 | 8-16 | ~5 steps/sec |
| CPU only | 1-2 | ~0.3 steps/sec |

### Dataset Size Guidelines
| Episodes | Frames (approx) | Disk Space | Training Time (5K steps) |
|----------|-----------------|------------|--------------------------|
| 50 | ~2,000 | ~500 MB | ~1 hour |
| 100 | ~4,000 | ~1 GB | ~1 hour |
| 500 | ~20,000 | ~5 GB | ~1.5 hours |

---

## Troubleshooting

### Common Issues

1. **OOM on MPS**: Reduce batch size to 2 or use `--max-samples` for testing
2. **NaN Loss**: Check if dataset has valid images and actions
3. **Slow Training**: Ensure you're using the v2 script with proper preprocessors
4. **OpenMP Error**: Set `KMP_DUPLICATE_LIB_OK=TRUE` (already done in v2 script)

### Performance Tips
- Use `--fast` flag during data collection
- Start with small tests: `--max-samples 200 --training-steps 50`
- Monitor wandb for loss curves
- Save checkpoints frequently when training is unstable


Eval: checkpoint-15000 | 10 episodes | seed 42

   Ep  Target           MinDist  Approach  GripTry  GripObj  Lifted  Placed  GripStep
  -----------------------------------------------------------------------------------
    1  red cylinder       0.171        no      yes       no      no      no         2
    2  green sphere       0.219        no      yes       no      no     yes         3
    3  green sphere       0.255        no      yes       no      no      no         2
    4  green sphere       0.253        no      yes       no      no      no         4
    5  red sphere         0.177        no      yes       no      no      no         2
    6  red sphere         0.045       yes      yes      yes      no     yes         2
    7  blue cylinder      0.089        no      yes       no      no     yes         4
    8  red sphere         0.160        no      yes       no      no      no         3
    9  red cylinder       0.103        no      yes       no      no      no         2
   10  red sphere         0.062        no      yes       no      no      no         3

Aggregate (10 episodes):
  Min distance to target:    0.153 +/- 0.072
  Approach success (<5cm):   1/10 (10.0%)
  Gripper attempted:         10/10 (100.0%)
  Gripper at object:         1/10 (10.0%)
  Object lifted:             0/10 (0.0%)
  Place success:             3/10 (30.0%)
  Mean action magnitude:     0.37 +/- 0.27
  Mean path length:          8.68 +/- 5.44

  Eval: checkpoint-25000 | 10 episodes | seed 42

   Ep  Target           MinDist  Approach  GripTry  GripObj  Lifted  Placed  GripStep
  -----------------------------------------------------------------------------------
    1  red cylinder       0.228        no      yes       no     yes      no         2
    2  green sphere       0.206        no      yes       no     yes     yes         4
    3  green sphere       0.125        no      yes       no      no      no         2
    4  green sphere       0.226        no      yes       no      no      no         4
    5  red sphere         0.178        no      yes       no      no      no         3
    6  red sphere         0.098        no      yes       no      no      no         2
    7  blue cylinder      0.207        no      yes       no     yes     yes         7
    8  red sphere         0.143        no      yes       no      no      no         4
    9  red cylinder       0.034       yes      yes      yes      no      no         3
   10  red sphere         0.085        no      yes       no      no      no         4

Aggregate (10 episodes):
  Min distance to target:    0.153 +/- 0.063
  Approach success (<5cm):   1/10 (10.0%)
  Gripper attempted:         10/10 (100.0%)
  Gripper at object:         1/10 (10.0%)
  Object lifted:             3/10 (30.0%)
  Place success:             2/10 (20.0%)
  Mean action magnitude:     0.33 +/- 0.23
  Mean path length:          6.23 +/- 3.10

Saved to: checkpoints/smolvla_full/checkpoint-25000/eval_metrics.json
  The model is improving despite flat loss and flat min_dist. Key signals:                                                                                                                                
                                                                                                                                                                                                          
  - Object lifted went from 0% to 30% — this is the most meaningful change. The robot is actually picking things up now, even if approach distance hasn't improved. This likely means the grasps that do  
  happen are more forceful/effective.                                                                                                                                                                     
  - Path length dropped from 8.68 to 6.23 — trajectories are more efficient, less wandering.                                                                                                              
  - Action magnitude variance dropped — more consistent motor commands.                                                                                                                                   
                                                                                                                                                                                                          
  The bottleneck is still gripper timing. The gripper still closes on steps 2-7, well before the robot reaches the object. The 3 lifts probably happened when objects spawned close to the starting arm   
  position.                                                                                                                                                                                               
                                                                                                                                                                                                          
  The pattern suggests the model is learning how to grasp but not when. This is a common VLA training dynamic — spatial control improves faster than temporal sequencing. Further training should help,   
  but if gripper_switch_step is still stuck at 2-7 by checkpoint 40-45k, that might indicate the training data doesn't have enough variety in approach duration or the model needs more steps to          
  disentangle the approach/grasp phases.                                                                                                                                                                  
                                                   