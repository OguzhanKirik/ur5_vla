# UR5 SmolVLA Grasping Training Pipeline

A complete pipeline for collecting demonstration data, training SmolVLA vision-language-action models, and evaluating performance on UR5 robotic grasping tasks in PyBullet simulation.


## Overview

This project implements a complete vision-language-action learning pipeline:

1. **Data Collection**: Generates demonstration episodes using a simulated UR5 robot performing pick-and-place tasks
2. **Model Training**: Fine-tunes SmolVLA (Small Vision-Language-Action) models on collected data
3. **Evaluation**: Tests trained models in simulation with success metrics

### Key Features

- ✅ **Multi-camera observations**: Top-down and wrist-mounted camera views
- ✅ **Language instructions**: Natural language task descriptions with variants for data augmentation
- ✅ **LeRobot format**: Compatible with HuggingFace's LeRobot framework
- ✅ **Vision-Language Grounding**: Language-conditioned action prediction
- ✅ **Action Chunking**: Predicts sequence of actions per timestep
- ✅ **PyBullet Simulation**: Safe, fast data collection and testing

### Setup

```bash
# Clone repository and navigate to project
cd /path/to/ur5_grasp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install LeRobot (if not included in requirements.txt)
pip install -e ./ur5_grasp/src/lerobot

# Install PyTorch with MPS support (macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA support (NVIDIA GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

Run the complete pipeline in ~2-3 minutes:

```bash
cd ur5_grasp/scripts

# 1. Collect 10 quick demo episodes
python collect_lerobot_smolvla.py --episodes 10 --fast

# 2. Train for 50 steps (testing)
python train_smolvla_lerobot_v2.py --max-samples 100 --training-steps 50 --batch-size 2

# 3. Evaluate on 2 episodes
python eval_smolvla_lerobot.py --checkpoint-dir ./checkpoints/smolvla_lerobot/final_model --num-episodes 2
```

**Expected output**: Model learns basic pick-and-place behavior in 50 steps (mostly random due to limited training).

---

## Data Generation

### Overview

The data collection script (`collect_lerobot_smolvla.py`) generates demonstration episodes of the UR5 robot performing pick-and-place tasks. Each episode includes:

- **Multi-camera video**: Top-down and wrist views (640×480, 30 FPS)
- **Robot state**: 6 joint positions + 2 gripper values = 8D state
- **Actions**: 6 joint velocities + 1 gripper command = 7D actions
- **Language instruction**: Natural language task description
- **Metadata**: Episode length, success status, object type

### Usage

```bash
python collect_lerobot_smolvla.py [OPTIONS]
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 100 | Number of episodes to collect |
| `--repo-id` | str | `local/ur5_smolvla_grasp` | Dataset repository ID |
| `--root` | str | `./datasets/lerobot` | Root directory to save dataset |
| `--gui` | flag | False | Show PyBullet GUI visualization |
| `--fast` | flag | False | Fast mode (skip rendering delays) |
| `--save-only-success` | flag | False | Only save successful grasps |

#### Examples

```bash
# Collect 100 episodes with GUI
python collect_lerobot_smolvla.py --episodes 100 --gui

# Fast collection (no rendering delays)
python collect_lerobot_smolvla.py --episodes 500 --fast

# Only save successful episodes
python collect_lerobot_smolvla.py --episodes 200 --save-only-success

# Custom output location
python collect_lerobot_smolvla.py --repo-id custom/my_dataset --root /data/lerobot
```

### Data Collection Process

Each episode follows an 8-phase pick-and-place sequence:

#### Phase 1: Approach (~200 steps)
- Robot end-effector moves to 20cm above target object
- Gripper remains open
- Establishes initial grasp approach

#### Phase 2: Descend (~200 steps)
- Gradual descent to 1cm above object surface
- Precise positioning for optimal grasp
- Visual alignment with target

#### Phase 3: Grasp (~1000 steps)
- Gripper closes on object
- Force feedback monitors grip quality
- Stops when torque exceeds threshold (>2.5 Nm)
- Ensures stable grip before lifting

#### Phase 4: Lift (~100 steps)
- Robot lifts object 15cm vertically
- Maintains constant grip force
- Confirms successful grasp

#### Phase 5: Transport (~200 steps)
- Robot moves to container position
- Navigates while maintaining grip
- Smooth trajectory planning

#### Phase 6: Release (~50 steps)
- Gripper opens to release object
- Object drops into container
- Verifies successful placement

#### Phase 7: Settle (~100 steps)
- Physics engine settles
- Verifies object stayed in container
- Marks episode as success/failure

#### Phase 8: Return Home (~300 steps)
- Robot returns to neutral position
- Resets for next episode
- Ready for continuous data collection

### Dataset Format

#### Objects and Instructions

| Object | Color | Shape | Example Instruction |
|--------|-------|-------|---------------------|
| 0 | Red | Sphere | "Pick up the red sphere and place it in the container" |
| 1 | Green | Sphere | "Grasp the green sphere and put it in the box" |
| 2 | Blue | Cylinder | "Pick up the blue cylinder and drop it in the container" |
| 3 | Red | Cylinder | "Pick up the red cylinder and place it in the container" |

Each object has 4 instruction variants for data augmentation:
- "Pick up the X" → "Grasp the X" → "Take the X" → "Move the X"

#### Feature Specification

**Observation State** (8 dimensions):
```python
{
    "dtype": "float32",
    "shape": (8,),
    "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_pos", "gripper_state"]
}
```

**Camera Images** (2 views):
- `observation.images.top`: 640×480, 30 FPS, overhead view
- `observation.images.wrist`: 640×480, 30 FPS, wrist camera view

**Actions** (7 dimensions):
```python
{
    "dtype": "float32",
    "shape": (7,),
    "names": ["joint_0_vel", "joint_1_vel", "joint_2_vel", "joint_3_vel", "joint_4_vel", "joint_5_vel", "gripper"]
}
```

**Task Description**:
```python
{
    "dtype": "str",
    "example": "Pick up the red sphere and place it in the container"
}
```

#### Output Directory Structure

```
datasets/lerobot/local/ur5_smolvla_grasp/
├── data/
│   └── chunk-000/
│       └── episode_*.parquet        # Tabular data (states, actions)
├── videos/
│   ├── observation.images.top/
│   │   └── episode_*.mp4            # Top camera videos
│   └── observation.images.wrist/
│       └── episode_*.mp4            # Wrist camera videos
├── meta/
│   ├── info.json                    # Dataset metadata
│   ├── stats.json                   # Normalization statistics
│   ├── tasks.json                   # Task descriptions
│   └── episodes/
│       └── *.parquet                # Episode metadata
└── README.md                         # Dataset documentation
```

#### Normalization Statistics

The `stats.json` file contains mean/std for each feature:

```json
{
    "observation.state": {
        "mean": [0.0, ...],
        "std": [1.0, ...]
    },
    "action": {
        "mean": [0.0, ...],
        "std": [1.0, ...]
    }
}
```

These statistics are used during training for input normalization.

---

## Training SmolVLA Model

### Overview

The training script (`train_smolvla_lerobot_v2.py`) fine-tunes the SmolVLA model on collected demonstration data. SmolVLA is a lightweight vision-language-action model that:

- Processes multi-camera observations
- Understands natural language instructions
- Predicts action sequences (chunks)

### Model Architecture

| Component | Specification |
|-----------|---------------|
| **Vision Encoder** | SmolVLM2-500M-Video-Instruct |
| **Language Model** | Integrated VLM backbone |
| **Parameters** | ~500M total |
| **Action Head** | Diffusion-based action decoder |
| **Chunk Size** | 50 actions per prediction |
| **Latency** | ~200ms per prediction |

### Usage

```bash
python train_smolvla_lerobot_v2.py [OPTIONS]
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--repo-id` | str | `local/ur5_smolvla_grasp` | Dataset repository ID |
| `--root` | str | `../../datasets/lerobot` | Dataset root directory |
| `--output-dir` | str | `./checkpoints/smolvla_lerobot` | Checkpoint output directory |
| `--batch-size` | int | 4 | Batch size for training |
| `--training-steps` | int | 5000 | Total training steps |
| `--lr` | float | 1e-5 | Learning rate |
| `--chunk-size` | int | 50 | Action chunk size |
| `--checkpoint-freq` | int | 500 | Save checkpoint every N steps |
| `--log-freq` | int | 10 | Log metrics every N steps |
| `--max-samples` | int | None | Limit dataset size for testing |
| `--resume` | flag | False | Resume from latest checkpoint |

#### Examples

```bash
# Full training (recommended)
python train_smolvla_lerobot_v2.py \
    --batch-size 4 \
    --training-steps 10000 \
    --lr 1e-5 \
    --checkpoint-freq 1000

# Quick test (50 steps, 200 samples)
python train_smolvla_lerobot_v2.py \
    --max-samples 200 \
    --training-steps 50 \
    --batch-size 2

# Resume training from checkpoint
python train_smolvla_lerobot_v2.py \
    --resume \
    --training-steps 20000 \
    --lr 5e-6  # Use lower LR for fine-tuning
```

### Training Configuration

#### Optimizer: AdamW
```python
{
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8
}
```

#### Learning Rate Schedule
- **Warmup**: 500 steps (linear 0% → 100%)
- **Decay**: Cosine schedule from 100% → 10% of base LR
- **Total**: 5,000 steps = ~1 minute on M1 MPS

#### Gradient Clipping
- **Max norm**: 1.0
- **Prevents**: Exploding gradients

#### Batch Processing
1. **Load batch** from dataloader
2. **Preprocess**: Normalize images, tokenize language, normalize state/action
3. **Rename cameras**: `top` → `camera1`, `wrist` → `camera2`
4. **Move to device**: GPU/MPS
5. **Forward pass**: Policy processes batch, returns loss
6. **Backward pass**: Compute gradients
7. **Optimize**: Update weights with gradient clipping
8. **Log metrics**: Loss, LR, step

### Data Pipeline

#### Temporal Alignment
LeRobot loads data with proper temporal offsets:

```python
delta_timestamps = {
    "observation.images.top": [0.0],      # Current frame
    "observation.images.wrist": [0.0],    # Current frame
    "observation.state": [0.0],           # Current state
    "action": [0.0, 0.033, 0.066, ...]   # Action chunk (50 frames)
}
```

This ensures:
- Images and state are aligned to current timestep
- Actions span 50 future timesteps (~1.67 seconds at 30 FPS)

#### Preprocessing Steps
1. **Image normalization**: Resize to 224×224, normalize RGB values
2. **Language tokenization**: Convert task string to token IDs
3. **State normalization**: Apply z-score normalization using dataset statistics
4. **Action normalization**: Normalize action space for stable learning

### Checkpointing

Checkpoints saved every `--checkpoint-freq` steps:

```
checkpoints/smolvla_lerobot/
├── checkpoint-1000/
│   ├── model.safetensors      # Model weights
│   ├── preprocessor_config.json
│   ├── postprocessor_config.json
│   └── config.json
├── checkpoint-2000/
├── final_model/               # Best checkpoint
└── training_logs.jsonl        # Training metrics
```

### Monitoring Training

Training metrics logged to Weights & Biases:

- **Loss**: Action prediction loss (MSE)
- **Learning Rate**: Current LR in schedule
- **Throughput**: Steps per second

View logs:
```bash
wandb login
wandb sync ./checkpoints/smolvla_lerobot/
```

### Hardware Requirements

| Device | Batch Size | Speed | Memory |
|--------|------------|-------|--------|
| Apple M1/M2 (MPS) | 2-4 | ~1-2 steps/sec | ~8GB |
| NVIDIA RTX 3090 | 8-16 | ~5-10 steps/sec | ~24GB |
| Intel CPU | 1-2 | ~0.2 steps/sec | ~4GB |

---

## Evaluation

### Overview

The evaluation script (`eval_smolvla_lerobot.py`) tests a trained SmolVLA model by:

1. Loading trained checkpoint
2. Running inference in simulation
3. Computing success metrics
4. Recording episode statistics

### Usage

```bash
python eval_smolvla_lerobot.py [OPTIONS]
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint-dir` | str | Required | Path to trained model checkpoint |
| `--num-episodes` | int | 10 | Number of evaluation episodes |
| `--gui` | flag | False | Show PyBullet GUI |
| `--save-videos` | flag | False | Save video recordings |
| `--output-dir` | str | `./eval_results` | Output directory for results |

#### Examples

```bash
# Evaluate best model (10 episodes)
python eval_smolvla_lerobot.py \
    --checkpoint-dir ./checkpoints/smolvla_lerobot/final_model

# Quick eval with visualization (2 episodes)
python eval_smolvla_lerobot.py \
    --checkpoint-dir ./checkpoints/smolvla_lerobot/checkpoint-5000 \
    --num-episodes 2 \
    --gui

# Full eval with video recording
python eval_smolvla_lerobot.py \
    --checkpoint-dir ./checkpoints/smolvla_lerobot/final_model \
    --num-episodes 50 \
    --save-videos \
    --output-dir ./eval_results/run_001
```

### Evaluation Process

For each episode:

1. **Reset environment**: Random object placement, clear container
2. **Sample task**: Randomly select object and instruction variant
3. **Initialize history**: Current images, state, and empty action history
4. **Loop for max steps** (typically 2000-3000):
   - Preprocess observations (images, state, language)
   - Run model inference
   - Postprocess predicted actions
   - Execute action chunk (all 50 predicted actions)
   - Update state/images
   - Check termination conditions
5. **Record metrics**: Success, num steps, object type, final position


### Output Structure

```
eval_results/
├── metrics.json              # Aggregate statistics
├── episodes.jsonl            # Per-episode data
├── success_rate.txt          # Human-readable summary
└── videos/                   # (if --save-videos)
    ├── episode_00_success.mp4
    ├── episode_01_failure.mp4
    └── ...
```

## References

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [SmolVLA Paper](https://huggingface.co/blog/smolvla)
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwMHQiMK27IgA1U)
- [UR5 Specifications](https://www.universal-robots.com/products/ur5-robot/)

---

## License

[Specify your license here]

## Citation

If you use this code in research, please cite:

```bibtex
@misc{ur5smolvla2024,
  title={UR5 SmolVLA Grasping Training Pipeline},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```

---

**Last Updated**: January 2026

