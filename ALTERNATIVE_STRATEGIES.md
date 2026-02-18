# Alternative Training Strategies for Space Invaders

This document describes alternative algorithms and architectures that could replace or complement the current NEAT implementation. Each section explains how the approach works, why it would be effective for this game, and what it would look like in practice.

---

## 1. PPO — Proximal Policy Optimization

### What it is
PPO is a gradient-based reinforcement learning algorithm developed by OpenAI. Instead of evolving a population, a single agent continuously improves itself by playing the game and adjusting its weights using backpropagation — guided by the existing reward signals already defined in `training_types.py`.

### Why it would work here
PPO learns from **every single frame** rather than waiting for a generational selection cycle. This makes it orders of magnitude more sample-efficient than NEAT. It has also been the dominant algorithm for Atari-style games since 2017, and Space Invaders is an Atari benchmark.

Key properties that match this project:
- Works directly with the existing reward profile system (`kill_reward`, `survival_reward`, etc.)
- Handles discrete action spaces (move left / move right / shoot) natively
- Stable training — the "proximal" constraint prevents large destructive weight updates

### How it would be implemented

```python
# pip install stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Wrap the existing engine as a Gymnasium environment
env = SpaceInvadersEnv(reward_profile=REWARD_PROFILES["kill_focus"])
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
model.learn(total_timesteps=1_000_000)
model.save("ppo_space_invaders")
```

The `SpaceInvadersEnv` wrapper would call `step_frame()` from `engine.py` on each `env.step()` call and `build_observation()` on each `env.reset()` / `env.step()` call — the internals are already compatible.

### Expected improvement
PPO typically converges to a competent policy in ~500k–1M frames. The current NEAT setup with pop_size=100 and 50 generations evaluates roughly `100 × 3 × 3600 = 1,080,000` frames total — but most of those frames are wasted on poor genomes. PPO uses all frames productively.

---

## 2. DQN — Deep Q-Network

### What it is
DQN (DeepMind, 2013) trains a neural network to estimate the **Q-value** (expected future reward) of each action in each state. The agent always picks the action with the highest Q-value. A **replay buffer** stores past experiences and is sampled randomly during training, breaking temporal correlations.

### Why it would work here
DQN was literally benchmarked on Atari Space Invaders in the original paper and achieved superhuman performance. The action space (3 discrete actions) is a textbook DQN setup.

Key components:
- **Replay buffer**: stores `(state, action, reward, next_state, done)` tuples; replayed for stable training
- **Target network**: a frozen copy of the Q-network updated periodically, preventing runaway feedback loops
- **ε-greedy exploration**: randomly picks actions early in training, exploiting learned policy later

### How it would be implemented

```python
# pip install stable-baselines3
from stable_baselines3 import DQN

env = SpaceInvadersEnv(reward_profile=REWARD_PROFILES["kill_focus"])
model = DQN(
    "MlpPolicy",
    env,
    buffer_size=100_000,
    learning_starts=10_000,
    target_update_interval=1000,
    exploration_fraction=0.2,
    verbose=1,
)
model.learn(total_timesteps=500_000)
```

### Dueling DQN variant
An extension that splits the Q-network into two streams — one estimating state value V(s), one estimating action advantage A(s,a) — and recombines them. This is especially beneficial for survival tasks where "staying alive" has value independent of which action is taken, matching the `survival_reward` already in the reward profiles.

---

## 3. A2C / A3C — Advantage Actor-Critic

### What it is
Actor-Critic methods maintain two networks simultaneously:
- **Actor**: outputs a probability distribution over actions (like a policy)
- **Critic**: estimates how good the current state is (like a value function)

The actor is updated using the **advantage** — how much better an action was than the average expected return. A3C runs multiple independent agents in parallel (asynchronous), each sending gradient updates to a shared model.

### Why it would work here
A2C/A3C is a natural middle ground between PPO and DQN:
- Lower memory usage than DQN (no replay buffer)
- Faster wall-clock training via parallelism (A3C)
- The **advantage** signal is a cleaner training signal than raw rewards, reducing variance

A3C could run one agent per CPU core, all training the same network — far more efficient than NEAT's sequential per-genome evaluation.

### How it would be implemented

```python
from stable_baselines3 import A2C

env = make_vec_env(SpaceInvadersEnv, n_envs=8)  # 8 parallel environments
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

---

## 4. CMA-ES — Covariance Matrix Adaptation Evolution Strategy

### What it is
CMA-ES is an evolutionary algorithm that, unlike NEAT, works with **fixed-topology networks**. It maintains a probability distribution over weight vectors and updates the distribution's mean and covariance matrix each generation based on which samples performed best. This allows it to capture correlations between weights and take more informed mutation steps.

### Why it would work here
If staying with evolution is a requirement but NEAT's topology-search overhead feels wasteful, CMA-ES is the right alternative:
- More mathematically principled weight search than NEAT's random mutations
- No speciation overhead
- Scales well to larger networks
- Works with the same fitness function and episode evaluation already in place

### How it would be implemented

```python
# pip install cma
import cma
import numpy as np

# Flatten all network weights into a single vector
def fitness(weight_vector):
    net = build_network(weight_vector, topology)
    results = [run_single_agent_episode(net, rng=...) for _ in range(3)]
    return -np.mean([r.fitness_delta for r in results])  # CMA-ES minimises

es = cma.CMAEvolutionStrategy(initial_weights, sigma0=0.5, {'popsize': 50})
es.optimize(fitness)
```

CMA-ES would use exactly the same `run_single_agent_episode` and reward profiles — only the outer optimisation loop changes.

---

## 5. OpenAI Evolution Strategies (OpenAI-ES)

### What it is
OpenAI-ES (Salimans et al., 2017) treats the neural network weights as parameters to be optimised by estimating gradients through random perturbations. For each generation, N copies of the weights are created with Gaussian noise added; each copy is evaluated; the gradient is estimated from the correlation between noise and fitness.

### Why it would work here
OpenAI-ES is embarrassingly parallel — every perturbation can run independently on a separate CPU — and it has shown competitive performance with PPO on many continuous control and game tasks. It also requires no replay buffer and no backpropagation.

Key advantages over NEAT in this setting:
- Fixed network size (no topology search overhead)
- Scales linearly with more CPU cores
- Simpler hyperparameter surface than NEAT

### Sketch

```python
import numpy as np

def evaluate(weights):
    net = build_network(weights)
    return np.mean([run_single_agent_episode(net, ...).fitness_delta for _ in range(3)])

weights = np.random.randn(num_params) * 0.1
for generation in range(200):
    noise = np.random.randn(population_size, num_params)
    rewards = np.array([evaluate(weights + sigma * noise[i]) for i in range(population_size)])
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    weights += learning_rate / (population_size * sigma) * (noise.T @ rewards)
```

---

## 6. HyperNEAT

### What it is
HyperNEAT extends NEAT by using a **Compositional Pattern Producing Network (CPPN)** to generate the weights of a larger substrate network. Instead of evolving the weights directly, it evolves a function `f(x1, y1, x2, y2) → weight` that maps geometric relationships between neurons to connection weights.

### Why it would work here
Space Invaders has strong geometric structure — enemies appear in grids, lasers travel in straight lines, the player is confined to the horizontal axis. HyperNEAT exploits this regularity: a weight pattern that works for dodging a laser on the left should generalise to the right without relearning, because the CPPN encodes spatial symmetry.

Current NEAT treats each weight independently and must re-discover symmetric strategies from scratch. HyperNEAT would naturally develop spatially-aware weight patterns.

### Implementation note
The `neat-python` library does not include HyperNEAT out of the box. It requires either the `HyperNEAT` extension in `pureples` or a custom CPPN substrate layer on top of the existing NEAT genome.

```
pip install pureples
```

---

## 7. Better Neural Network Architectures

These architectural improvements apply regardless of which training algorithm is used.

### Frame Stacking
The current agent sees a single snapshot of the world each frame — it has no sense of velocity or trajectory. Frame stacking feeds the last N observations (typically 4) concatenated as a single input, giving the network enough context to infer movement direction without a recurrent layer.

```
Input: [obs_t-3, obs_t-2, obs_t-1, obs_t]  →  4 × 19 = 76 inputs
```

This is cheap, deterministic, and has a well-understood effect on training stability.

### CNN on Raw Pixels
Rather than hand-crafting 19 features, a convolutional neural network (CNN) can learn directly from the 750×750 pixel game surface (typically downsampled to 84×84 grayscale). This removes the need for any feature engineering and allows the agent to discover signals that `build_observation()` does not capture.

```
Input:  84×84×4 (grayscale + frame stack)
Conv1:  32 filters, 8×8, stride 4
Conv2:  64 filters, 4×4, stride 2
Conv3:  64 filters, 3×3, stride 1
FC:     512 units → 3 action outputs
```

This is the exact architecture used by DeepMind's original DQN paper on Atari Space Invaders.

### LSTM / GRU Layer
Rather than relying on recurrent NEAT connections (which evolve slowly), an explicit LSTM or GRU layer added to the policy network gives the agent persistent memory with known, well-trained dynamics. This is especially useful for tracking the boss's horizontal bounce pattern over time.

```python
from stable_baselines3 import PPO
model = PPO("MlpLstmPolicy", env, verbose=1)  # built-in LSTM support in SB3
```

### Dueling Network Architecture
Splits the final fully-connected layers into:
- **Value stream** V(s): how good is this state in general?
- **Advantage stream** A(s, a): how much better is each action than average?

Combined as `Q(s,a) = V(s) + A(s,a) - mean(A(s,·))`. This is particularly effective when the best action is often "do nothing" (survive), which matches the survival-reward structure of this game.

---

## Summary Comparison

| Strategy | Sample Efficiency | Topology Search | Memory | Parallelism | Complexity |
|---|---|---|---|---|---|
| **NEAT (current)** | Low | Yes | Low | Limited | Medium |
| **PPO** | High | No | Low | Via vec envs | Low |
| **DQN** | High | No | High (replay buffer) | Limited | Low |
| **A2C / A3C** | Medium-High | No | Low | Excellent | Low |
| **CMA-ES** | Medium | No | Low | Medium | Low |
| **OpenAI-ES** | Medium | No | Low | Excellent | Low |
| **HyperNEAT** | Low | Yes (geometric) | Low | Limited | High |

For this project, the recommended path is:
1. **Quick win**: PPO via `stable-baselines3` — same reward signals, 10× faster convergence
2. **Best performance**: PPO + LSTM policy + frame stacking
3. **Stay evolutionary**: CMA-ES or OpenAI-ES over a fixed topology
