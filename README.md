# Normalizing Flows in SAC

This repository incorporates **Normalizing Flows** into a Soft Actor-Critic (SAC) agent.

## Motivation

1. **Richer Policy Distributions**: Standard policies in continuous control tasks often assume unimodal Gaussian distributions, limiting the expressiveness of the agent. By contrast, **Normalizing Flows** allow arbitrarily complex policy shapes, capturing multimodal, skewed, or otherwise intricate distributions.
2. **Better Exploration & Performance**: Having a richer policy can improve exploration, stability, and convergence in environments where a simple Gaussian policy might get stuck.

## Setup & Installation

1. **Clone the Repo**

```bash
git clone https://github.com/naivoder/NF-SAC.git
cd NF-SAC
```

2. **Create & Activate a Virtual Environment** (example using conda)

```bash
conda create --name nfsac python=3.11 -y
conda activate nfsac
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

If you run into errors compiling certain libraries (e.g., Box2D), ensure your system has the necessary C/C++ toolchains installed.

## Running the Code

Below are sample commands for running with and without normalizing flows:

```bash
# Run SAC with a normal Gaussian actor
python main.py --env_name HalfCheetah-v4 

# Run SAC with a normalizing-flow–based actor
python main.py --env_name HalfCheetah-v4 --norm_flow
```

## Acknowledgments

- **nflows** library for normalizing-flow transforms.
- **PyTorch** for deep learning.
- **Gymnasium** for continuous control environments.

If you have any questions, open an issue or submit a pull request!
