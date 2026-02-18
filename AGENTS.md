# AGENTS.md

## Mission

This repository is a research/prototype codebase for training NEAT agents to play a simplified Space Invaders environment.  
Primary objective: improve learning reliability by first fixing simulation correctness, then improving observations and rewards.

## Project Map

- `space_invaders.py`: game loop + NEAT training/evaluation entry point.
- `Object.py`: domain objects (`Ship`, `Player`, `Enemy`, `Laser`) and collision logic.
- `config-feedforward.txt`: NEAT hyperparameters and network topology.
- `assets/`: sprite and background assets.

## Current Intent (Behavioral Summary)

- Multiple genomes are evaluated each generation.
- Each genome controls one `Player` in a shared enemy-wave simulation.
- Policy outputs control left/right movement and shooting.
- Fitness is currently survival-heavy.

## Priority Order For Changes

1. **P0: Simulation correctness (must-do before tuning NEAT)**
   - Fix enemy shooting call location and frame update order.
   - Ensure enemy lasers move once per frame (not once per player).
   - Stop mutating `players`/`enemies` lists while iterating.
   - Remove circular module coupling (`Object.py` importing runtime constants from `space_invaders.py`).

2. **P1: Learning signal quality**
   - Improve state features (signed relative positions, more threat context, normalization).
   - Improve rewards (kills, wave progress, safer positioning), not survival-only.

3. **P2: Engineering quality**
   - Split simulation engine from training orchestration.
   - Add deterministic execution mode via explicit seeds.
   - Save best genomes and support replay.
   - Add tests for collision, cooldown, and entity lifecycle invariants.

## Guardrails

- Do not tune NEAT config aggressively until P0 is complete.
- Keep gameplay logic deterministic and single-pass per frame.
- Prefer explicit state transitions over side effects in nested loops.
- If you change observations, rewards, or action semantics, update both `README.md` and `config-feedforward.txt` comments.
- Preserve ASCII text unless file already requires Unicode.

## Minimum Validation Before Finishing Work

Run at least:

```bash
python3 -m py_compile space_invaders.py Object.py
```

If a graphical environment is available, also run:

```bash
python3 space_invaders.py
```

## Suggested Work Chunking

1. Constants/settings extraction and import cleanup.
2. Frame-step refactor (movement, shooting, collisions, removals).
3. Observation/reward redesign.
4. Checkpointing, replay, and tests.
