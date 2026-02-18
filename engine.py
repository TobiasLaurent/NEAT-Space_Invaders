from dataclasses import dataclass, field

from Object import Enemy, Boss, collide


@dataclass
class EpisodeState:
    player: object
    enemies: list = field(default_factory=list)
    wave_length: int = 5
    boss_active: bool = False
    level: int = 0
    lives: int = 5


@dataclass
class StepResult:
    fitness_delta: float
    terminal: bool
    active_boss: object


def _spawn_regular_wave(state, rng, world_width):
    state.level += 1
    state.wave_length += 5
    invaders_to_spawn = max(1, int(round(state.wave_length * 0.8)))
    for _ in range(invaders_to_spawn):
        state.enemies.append(
            Enemy(
                rng.randrange(50, world_width - 100),
                rng.randrange(-1500, -100),
                rng.choice(["red", "blue", "green"]),
            )
        )


def _spawn_boss_wave(state, world_width):
    boss_x = (world_width - Boss.BOSS_WIDTH) // 2
    state.enemies.append(Boss(boss_x, -Boss.BOSS_HEIGHT, state.level))


def _apply_reward(delta, reward_key, reward_totals):
    if reward_totals is not None:
        reward_totals[reward_key] += delta
    return delta


def step_frame(
    state,
    net,
    rng,
    build_observation,
    world_width,
    world_height,
    reward_values=None,
    event_totals=None,
    reward_totals=None,
):
    """Advance the simulation by one frame for both training and replay."""
    fitness_delta = 0.0

    survival_reward = reward_values["survival_reward"] if reward_values else 0.0
    kill_reward = reward_values["kill_reward"] if reward_values else 0.0
    boss_kill_reward = reward_values["boss_kill_reward"] if reward_values else 0.0
    wave_clear_reward = reward_values["wave_clear_reward"] if reward_values else 0.0
    shot_penalty = reward_values["shot_penalty"] if reward_values else 0.0
    laser_hit_penalty = reward_values["laser_hit_penalty"] if reward_values else 0.0
    death_penalty = reward_values["death_penalty"] if reward_values else 0.0
    enemy_escape_penalty = reward_values["enemy_escape_penalty"] if reward_values else 0.0
    level_fail_penalty = reward_values["level_fail_penalty"] if reward_values else 0.0

    if len(state.enemies) == 0:
        if state.boss_active:
            state.boss_active = False
            _spawn_regular_wave(state, rng, world_width)
        elif state.level > 0:
            if event_totals is not None:
                event_totals["wave_clears"] += 1
            if reward_values:
                fitness_delta += _apply_reward(wave_clear_reward, "wave_clear_reward_total", reward_totals)
            state.boss_active = True
            _spawn_boss_wave(state, world_width)
        else:
            state.boss_active = False
            _spawn_regular_wave(state, rng, world_width)

    if reward_values:
        fitness_delta += _apply_reward(survival_reward, "survival_reward_total", reward_totals)

    output = net.activate(build_observation(state.player, state.enemies))
    if output[0] > 0.5 and state.player.x + state.player.PLAYER_VEL + state.player.get_width() < world_width:
        state.player.move_right()
    if output[1] > 0.5 and state.player.x - state.player.PLAYER_VEL > 0:
        state.player.move_left()
    if output[2] > 0.5:
        shots_before = len(state.player.lasers)
        state.player.shoot()
        if len(state.player.lasers) > shots_before:
            if event_totals is not None:
                event_totals["shots_fired"] += 1
            if reward_values:
                fitness_delta += _apply_reward(-shot_penalty, "shot_penalty_total", reward_totals)

    state.player.cooldown()
    for laser in state.player.lasers[:]:
        laser.move(-1)
        if laser.off_screen(world_height):
            state.player.lasers.remove(laser)
            continue

        for enemy in state.enemies[:]:
            if laser.collision(enemy):
                enemy.health -= 100
                if laser in state.player.lasers:
                    state.player.lasers.remove(laser)
                if enemy.health <= 0 and enemy in state.enemies:
                    state.enemies.remove(enemy)
                    if event_totals is not None:
                        event_totals["kills"] += 1
                        if isinstance(enemy, Boss):
                            event_totals["boss_kills"] += 1
                    if reward_values:
                        reward_delta = boss_kill_reward if isinstance(enemy, Boss) else kill_reward
                        fitness_delta += _apply_reward(reward_delta, "kill_reward_total", reward_totals)
                break

    for enemy in state.enemies[:]:
        if isinstance(enemy, Boss):
            enemy.move(world_width)
            shoot_window = max(25, 50 - (state.level * 2))
        else:
            enemy.move()
            shoot_window = 2 * 60

        if rng.randrange(0, shoot_window) == 1:
            enemy.shoot()

        enemy.cooldown()
        for laser in enemy.lasers[:]:
            laser.move(1)
            if laser.off_screen(world_height):
                enemy.lasers.remove(laser)
                continue

            if laser.collision(state.player):
                state.player.health -= 100
                if event_totals is not None:
                    event_totals["laser_hits_taken"] += 1
                if reward_values:
                    fitness_delta += _apply_reward(-laser_hit_penalty, "death_penalty_total", reward_totals)
                if laser in enemy.lasers:
                    enemy.lasers.remove(laser)

        if enemy.y + enemy.get_height() > world_height:
            lives_loss = 2 if isinstance(enemy, Boss) else 1
            state.lives -= lives_loss
            if event_totals is not None:
                event_totals["enemy_escapes"] += 1
            if reward_values:
                reward_delta = -enemy_escape_penalty * lives_loss
                fitness_delta += _apply_reward(reward_delta, "enemy_escape_penalty_total", reward_totals)
            state.enemies.remove(enemy)
            continue

        if collide(enemy, state.player):
            state.player.health -= 100
            if not isinstance(enemy, Boss) and enemy in state.enemies:
                state.enemies.remove(enemy)

    terminal = False
    if state.player.health <= 0:
        terminal = True
        if event_totals is not None:
            event_totals["player_deaths"] += 1
        if reward_values:
            fitness_delta += _apply_reward(-death_penalty, "death_penalty_total", reward_totals)
    elif state.lives <= 0:
        terminal = True
        if event_totals is not None:
            event_totals["level_failures"] += 1
        if reward_values:
            fitness_delta += _apply_reward(-level_fail_penalty, "level_fail_penalty_total", reward_totals)

    active_boss = next((enemy for enemy in state.enemies if isinstance(enemy, Boss)), None)
    return StepResult(
        fitness_delta=fitness_delta,
        terminal=terminal,
        active_boss=active_boss,
    )
