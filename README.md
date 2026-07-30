# MAPPO for PettingZoo Simple Spread

A clean and modular implementation of **Multi-Agent Proximal Policy Optimization (MAPPO)** in **PyTorch** for the **PettingZoo MPE2 Simple Spread** environment.

The project follows the **Centralized Training, Decentralized Execution (CTDE)** paradigm and is intended for learning, experimentation, and future research on cooperative multi-agent reinforcement learning.

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
- ✅ CSV evaluation reports
- ✅ Human rendering of trained policies
- ✅ Modular and extensible codebase

---

## Repository Structure

```text
.
├── configs/
│   └── simple_spread.yaml
│___ experiments
|     |__simple_spread
|         |-checkpoints
|         |-tensorboards
|
|
├── src/
│   ├── algorithms/
│   ├── buffers/
│   ├── config/
│   ├── envs/
│   ├── networks/
│   ├── trainers/
│   └── utils/
│
├── train.py
├── evaluate.py
├── play.py
├── pyproject.toml
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AkArun12/mappo-simple-spread.git
cd mappo-simple-spread
```

### 2. Create a virtual environment

```bash
python -m venv mappo_venv
```

### 3. Activate the environment

#### macOS / Linux

```bash
source mappo_venv/bin/activate
```

#### Windows

```powershell
mappo_venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -e .
```

---

## Training

Start training with

```bash
python train.py
```

Each training run creates a new experiment directory:

```text
experiments/
└── run_name/
    ├── checkpoints/
    ├── tensorboard/
    └── config.yaml
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir experiments
```

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python evaluate.py experiments/<run_name>
```

Evaluation automatically generates

- `evaluation_results.csv`
- `summary.txt`

Example output:

```text
Average Reward : -23.13
Std Reward     : 5.52
Best Reward    : -10.85
Worst Reward   : -36.41
```

---

## Play

Render the trained policy:

```bash
python play.py experiments/<run_name> --episodes 5
```

---

## Hyperparameter Search

The following hyperparameters were explored during experimentation.

| Hyperparameter | Values |
|----------------|--------|
| Learning Rate | 3e-4, 1e-4 |
| PPO Clip Ratio | 0.20, 0.15 |
| Entropy Coefficient | 0.005, 0.01, 0.02 |
| PPO Epochs | 10, 15 |
| Mini-batch Size | 32, 16 |
| GAE Lambda | 0.95, 0.97 |

---

## Best Configuration

The best-performing configuration obtained during tuning:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| PPO Clip Ratio | 0.15 |
| Entropy Coefficient | 0.01 |
| PPO Epochs | 10 |
| Mini-batch Size | 16 |
| GAE Lambda | 0.95 |

---

## Method

The implementation follows the **Centralized Training, Decentralized Execution (CTDE)** framework.

During training:

- Each agent shares the same actor network.
- A centralized critic receives the global state.
- Advantages are estimated using Generalized Advantage Estimation (GAE).
- Policies are updated using the PPO clipped objective.

During execution:

- Agents act independently using only their local observations.

---

## Future Work

Potential extensions include:

- Agent communication
- Speaker–Listener environment
- Learned message passing
- Attention-based communication
- Graph Neural Networks (GNNs)
- Transformer-based MARL
- SMAC benchmark support

---

## References

- Schulman et al. (2017), **Proximal Policy Optimization Algorithms**
- Yu et al. (2022), **The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**
- PettingZoo
- MPE2

---

## Author

Arun Kathariya