# NEAT Space Invaders

This project explores whether a NEAT-evolved feed-forward network can learn to play a simplified Space Invaders environment in `pygame`.

## Purpose

The project has two goals:

1. Build a minimal neuroevolution benchmark on an arcade-style control task.
2. Learn which simulation/reward/observation design choices help or block training.

Each generation evaluates multiple genomes in-game and evolves policies for:
- horizontal movement (`left`, `right`)
- shooting
- survival against incoming enemies and lasers

## Repository Layout

- `space_invaders.py`: main game loop, NEAT evaluation, reward function, metrics logging
- `Object.py`: game entities (`Ship`, `Player`, `Enemy`, `Laser`) and collision utility
- `config-feedforward.txt`: NEAT hyperparameters
- `assets/`: sprites/background
- `training_metrics.csv`: per-generation reward/component telemetry (generated at runtime)

## Quick Start

```bash
cd /Users/lauret/workspace/NEAT-Space_Invaders
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 space_invaders.py
```

## Current Training Setup

- Population: `pop_size = 10`
- Run length: up to `50` generations (`p.run(eval_genomes, 50)`)
- Network inputs: `player.y`, `abs(player.x - target_enemy.x)`, nearest enemy-laser vertical distance
- Network outputs: move right, move left, shoot

### Active Reward Function

Current reward shaping in `space_invaders.py`:
- survival reward: `+0.02` per frame alive
- kill reward: `+4.0` per enemy killed
- wave clear reward: `+2.0` per cleared wave
- shot penalty: `-0.01` per shot fired
- death penalty: `-4.0` on player death
- enemy escape penalty: `-0.75` when an enemy reaches the bottom (applied to alive genomes)
- level fail penalty: `-2.0` when lives reach zero

## Implemented Findings So Far

### Simulation fixes already applied

- player shooting/cooldown now updates each frame during training
- player lasers now move and can kill enemies in `eval_genomes`
- enemy shooting is called per enemy (not from a leaked loop variable)
- unsafe removal patterns were reduced using copy iteration and reverse-index removal

### Observed results (through generations ~46-47)

From runtime logs and `training_metrics.csv`:
- agents still fail to clear wave 1 consistently (`wave_clear_reward_total` often `0`)
- shooting efficiency is low (`kill_per_shot` around `0.02-0.03`)
- training stagnates in a single species for many generations
- large positive survival totals are offset by death/escape/fail penalties

Example log snapshot:
- Gen 46: `shots=34`, `kills=1`, `k/shot=0.029`
- Gen 47: `shots=107`, `kills=2`, `k/shot=0.019`

Interpretation: agents fire, but mostly inaccurately; policy quality is limited more by weak observations and low diversity than by NEAT itself.

## Why Performance Is Still Limited

1. **Observation bottleneck**
- `abs(player.x - enemy.x)` removes left/right direction information.
- `player.y` is effectively constant in this environment.
- only one enemy/laser summary is observed.

2. **Search diversity bottleneck**
- `pop_size = 10` and persistent one-species dynamics reduce exploration.

3. **Credit assignment bottleneck**
- Multi-agent shared simulation introduces noisy reward attribution.

## Telemetry

Per-generation metrics are printed as:
- `[RewardLog][Gen X] ...`

And appended to:
- `training_metrics.csv`

Tracked fields include:
- fitness summary (`avg`, `best`, `worst`)
- event counts (`shots_fired`, `kills`, `enemy_escapes`, `player_deaths`, `wave_clears`)
- reward component totals (survival, kill, wave, shot/death/escape/fail penalties)
- `kill_per_shot`

## Recommended Next Steps

1. Improve observation features first (signed and normalized relative positions, nearest laser/enemy `dx` and `dy`).
2. Update `num_inputs` in `config-feedforward.txt` to match the richer observation vector.
3. Increase population size (`30-50`) to reduce stagnation.
4. Run multiple seeds and compare `training_metrics.csv` before changing algorithm family.
5. If progress remains flat after the above, evaluate PPO as the next baseline.
