# MAPPO Pipeline from Scratch

A clean, modular, and extensible implementation of **Multi-Agent Proximal Policy Optimization (MAPPO)** in **PyTorch** for the **PettingZoo MPE2 Simple Spread** environment.

This project follows the **Centralized Training, Decentralized Execution (CTDE)** paradigm and was developed from scratch to understand the complete MAPPO training pipeline. It serves as a foundation for future research on communication-based multi-agent reinforcement learning.

---

## Features

- ✅ MAPPO implementation from scratch using PyTorch
- ✅ PettingZoo MPE2 Simple Spread environment
- ✅ Shared actor network
- ✅ Centralized critic
- ✅ Generalized Advantage Estimation (GAE)
- ✅ PPO clipped objective
- ✅ Entropy regularization
- ✅ Gradient clipping
- ✅ Linear learning-rate decay
- ✅ TensorBoard logging
- ✅ Automatic checkpoint saving
- ✅ Evaluation utilities
- ✅ Human rendering of trained policies
- ✅ CSV evaluation reports
- ✅ Modular and extensible codebase
- ✅ Reference trained model included

---

# Repository Structure

```text
.
├── configs/
│   └── simple_spread.yaml
│
├── experiments/
│   └── simple_spread_v3/
│       └── reference_lr0.0001_clip0.15_ent0.01_ep10_mb16_train30000_seed42_20260729_2341/
│           ├── checkpoints/
│           │   └── best_model.pt
│           ├── config.yaml
│           ├── evaluation_results.csv
│           ├── summary.txt
│           └── README.md
│
├── src/
│   └── mappo_ss/
│       ├── algorithms/
│       ├── buffers/
│       ├── communication/
│       ├── config/
│       ├── envs/
│       ├── networks/
│       ├── trainers/
│       └── utils/
│
├── tests/
│
├── train.py
├── evaluate.py
├── play.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Requirements

- Python 3.11+
- PyTorch
- PettingZoo
- MPE2
- Gymnasium
- TensorBoard

All dependencies are automatically installed using

```bash
pip install -e .
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/AkArun12/mappo-pipeline-from-scratch.git
cd mappo-pipeline-from-scratch
```

## 2. Create a virtual environment

```bash
python -m venv mappo_venv
```

## 3. Activate the environment

### macOS / Linux

```bash
source mappo_venv/bin/activate
```

### Windows

```powershell
mappo_venv\Scripts\activate
```

## 4. Install the project

```bash
pip install -e .
```

---

# Training

Start training with

```bash
python train.py
```

Each training run automatically creates a new experiment directory.

```text
experiments/
└── run_name/
    ├── checkpoints/
    ├── tensorboard/
    └── config.yaml
```

Visualize training using TensorBoard

```bash
tensorboard --logdir experiments
```

---

# Evaluation

Evaluate a trained checkpoint

```bash
python evaluate.py experiments/<run_name>
```

Evaluation automatically generates

- `evaluation_results.csv`
- `summary.txt`

Example output

```text
============================================================
Evaluation Summary
============================================================

Average Reward : -23.13
Std Reward     : 5.52
Best Reward    : -10.85
Worst Reward   : -36.41
```

---

# Play

Render a trained policy

```bash
python play.py experiments/<run_name> --episodes 5
```

Example

```bash
python play.py \
experiments/simple_spread_v3/reference_lr0.0001_clip0.15_ent0.01_ep10_mb16_train30000_seed42_20260729_2341 \
--episodes 5
```

---

# Hyperparameter Search

The following hyperparameters were explored during experimentation.

| Hyperparameter | Values Explored |
|----------------|-----------------|
| Learning Rate | 3e-4, 1e-4 |
| PPO Clip Ratio | 0.20, 0.15 |
| Entropy Coefficient | 0.005, 0.01, 0.02 |
| PPO Epochs | 10, 15 |
| Mini-batch Size | 32, 16 |
| GAE Lambda | 0.95, 0.97 |

---

# Best Configuration

The following configuration achieved the best evaluation performance.

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| PPO Clip Ratio | 0.15 |
| Entropy Coefficient | 0.01 |
| PPO Epochs | 10 |
| Mini-batch Size | 16 |
| GAE Lambda | 0.95 |

---

# Evaluation Results

Reference model performance over **100 evaluation episodes**.

| Metric | Value |
|---------|-------|
| Average Reward | **-23.13** |
| Standard Deviation | **5.52** |
| Best Episode | **-10.85** |
| Worst Episode | **-36.41** |

---

# Reference Trained Model

A fully trained reference model is included in

```text
experiments/simple_spread_v3/reference_lr0.0001_clip0.15_ent0.01_ep10_mb16_train30000_seed42_20260729_2341/
```

Contents include

- `best_model.pt`
- `config.yaml`
- `evaluation_results.csv`
- `summary.txt`
- `README.md`

You can evaluate it immediately without retraining.

```bash
python evaluate.py \
experiments/simple_spread_v3/reference_lr0.0001_clip0.15_ent0.01_ep10_mb16_train30000_seed42_20260729_2341
```

---

# Method

This implementation follows the **Centralized Training, Decentralized Execution (CTDE)** framework.

### During Training

- Shared actor network across all agents
- Centralized critic receives the global state
- Advantages computed using Generalized Advantage Estimation (GAE)
- PPO clipped objective for stable policy updates
- Entropy regularization encourages exploration

### During Execution

- Only the trained actor is used
- Each agent acts independently
- Decisions are based solely on local observations

---

# Roadmap

Current progress

- ✅ MAPPO implementation
- ✅ Training pipeline
- ✅ Evaluation pipeline
- ✅ Hyperparameter tuning
- ✅ Reference trained model
- ⏳ Communication module
- ⏳ Speaker–Listener environment
- ⏳ Graph Neural Network communication
- ⏳ Transformer-based MARL
- ⏳ SMAC benchmark support

---

# Future Work

Potential extensions include

- Explicit agent communication
- Learned message passing
- Attention-based communication
- Graph Neural Networks
- Transformer policies
- Recurrent MAPPO
- Additional PettingZoo environments
- StarCraft Multi-Agent Challenge (SMAC)

---

# References

1. Schulman et al. (2017). **Proximal Policy Optimization Algorithms.**

2. Yu et al. (2022). **The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.**

3. PettingZoo: Multi-Agent Reinforcement Learning Environments.

4. MPE2 (Multi-Agent Particle Environments).

---

# License



---

# Acknowledgements

This project builds upon ideas introduced in PPO and MAPPO and uses the PettingZoo MPE2 environments for benchmarking cooperative multi-agent reinforcement learning algorithms.

---

# Author

**Arun Kathariya**

GitHub: https://github.com/AkArun12