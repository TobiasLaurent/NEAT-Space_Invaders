# In this version of Space Invaders the player can only move left and right.
# The player loses when an enemy ship collides with him,
# the enemy hits the player with a laser, or when 5
# enemy ships successfully lands on earth,
# i.e. surpasses the space ship of the player.
import pygame
import os
import csv
import argparse
import pickle
# import time
import random
import neat

from Object import *
from engine import EpisodeState, step_frame
from observations import build_observation as build_observation_for_world
from training_types import (
    BenchmarkMetrics,
    EpisodeResult,
    EventTotals,
    ExperimentResult,
    GenerationMetricsRow,
    GenomeEpisodeMetrics,
    RewardProfile,
    RewardTotals,
)

pygame.font.init()
hud_font = pygame.font.SysFont("comicsans", 30)
stats_font = pygame.font.SysFont("comicsans", 26)
pygame.display.set_caption("Space Shooter Tutorial")

WIDTH, HEIGHT = 750, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
BG = pygame.transform.scale(pygame.image.load(
    os.path.join("assets", "background-black.png")), (WIDTH, HEIGHT))
TRAINING_METRICS_PATH = os.path.join(os.path.dirname(__file__), "training_metrics.csv")
BEST_GENOME_PATH = os.path.join(os.path.dirname(__file__), "best_genome.pkl")
EXPERIMENT_SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "experiment_summary.csv")

REWARD_PROFILES = {
    "kill_focus": RewardProfile(
        survival_reward=0.005,
        kill_reward=12.0,
        boss_kill_reward=32.0,
        wave_clear_reward=4.0,
        shot_penalty=0.005,
        laser_hit_penalty=10.0,
        death_penalty=10.0,
        enemy_escape_penalty=1.0,
        level_fail_penalty=4.0,
    ),
    "balanced": RewardProfile(
        survival_reward=0.01,
        kill_reward=8.0,
        boss_kill_reward=20.0,
        wave_clear_reward=3.0,
        shot_penalty=0.005,
        laser_hit_penalty=6.0,
        death_penalty=7.0,
        enemy_escape_penalty=1.0,
        level_fail_penalty=3.0,
    ),
    "precision": RewardProfile(
        survival_reward=0.003,
        kill_reward=10.0,
        boss_kill_reward=26.0,
        wave_clear_reward=3.0,
        shot_penalty=0.02,
        laser_hit_penalty=9.0,
        death_penalty=9.0,
        enemy_escape_penalty=1.2,
        level_fail_penalty=4.0,
    ),
}

ACTIVE_REWARD_PROFILE_NAME = "kill_focus"
ACTIVE_REWARD_PROFILE = REWARD_PROFILES[ACTIVE_REWARD_PROFILE_NAME]
ACTIVE_TRAINING_METRICS_PATH = TRAINING_METRICS_PATH
EVAL_EPISODES_PER_GENOME = 3
EVAL_MAX_FRAMES = 3600

PROFILE_VISUALS = {
    "kill_focus": {
        "ship_tint": (255, 255, 255),
        "laser_tint": (255, 255, 255),
        "ui_color": (255, 220, 140),
        "label": "Kill Focus",
    },
    "balanced": {
        "ship_tint": (185, 255, 210),
        "laser_tint": (170, 255, 210),
        "ui_color": (140, 255, 200),
        "label": "Balanced",
    },
    "precision": {
        "ship_tint": (200, 210, 255),
        "laser_tint": (190, 220, 255),
        "ui_color": (160, 190, 255),
        "label": "Precision",
    },
}
PROFILE_ASSET_CACHE = {}


def append_training_metrics_row(metrics_row: GenerationMetricsRow):
    fieldnames = GenerationMetricsRow.fieldnames()
    file_exists = os.path.exists(ACTIVE_TRAINING_METRICS_PATH)
    with open(ACTIVE_TRAINING_METRICS_PATH, "a", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_row.as_dict())


def load_config(config_file):
    return neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )


def save_best_genome(genome, path):
    with open(path, "wb") as genome_file:
        pickle.dump(genome, genome_file)


def load_best_genome(path):
    with open(path, "rb") as genome_file:
        return pickle.load(genome_file)


def set_active_reward_profile(profile_name):
    global ACTIVE_REWARD_PROFILE_NAME, ACTIVE_REWARD_PROFILE
    if profile_name not in REWARD_PROFILES:
        raise ValueError(f"Unknown reward profile: {profile_name}")
    ACTIVE_REWARD_PROFILE_NAME = profile_name
    ACTIVE_REWARD_PROFILE = REWARD_PROFILES[profile_name]


def get_profile_visual(profile_name):
    return PROFILE_VISUALS.get(profile_name, PROFILE_VISUALS["kill_focus"])


def tint_surface(surface, tint_rgb):
    tinted = surface.copy()
    tint = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    tint.fill((*tint_rgb, 255))
    tinted.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return tinted


def get_profile_player_assets(profile_name):
    if profile_name in PROFILE_ASSET_CACHE:
        return PROFILE_ASSET_CACHE[profile_name]

    visuals = get_profile_visual(profile_name)
    ship = tint_surface(Player.YELLOW_SPACE_SHIP, visuals["ship_tint"])
    laser = tint_surface(Player.YELLOW_LASER, visuals["laser_tint"])
    PROFILE_ASSET_CACHE[profile_name] = (ship, laser)
    return ship, laser


def apply_profile_visuals_to_player(player, profile_name):
    ship_img, laser_img = get_profile_player_assets(profile_name)
    player.ship_img = ship_img
    player.laser_img = laser_img
    player.mask = pygame.mask.from_surface(player.ship_img)


def infer_profile_from_genome_path(path):
    filename = os.path.basename(path).lower()
    for profile_name in REWARD_PROFILES:
        if profile_name in filename:
            return profile_name
    return "kill_focus"


def resolve_start_mode(cli_mode):
    if cli_mode in ("fresh", "best"):
        return cli_mode

    print("\nSelect start mode:")
    print("  1) Fresh training run")
    print("  2) Replay best saved genome")
    try:
        choice = input("Enter 1 or 2 [default: 1]: ").strip()
    except EOFError:
        return "fresh"

    if choice == "2":
        return "best"
    return "fresh"


def build_observation(player, enemies):
    return build_observation_for_world(player, enemies, WIDTH, HEIGHT)


def draw_window(win, enemies, players, gen, level, lives_remaining, boss=None):

    if gen == 0:
        gen = 1

    win.blit(BG, (0, 0))

    for enemy in enemies:
        enemy.draw(win)

    profile_visual = get_profile_visual(ACTIVE_REWARD_PROFILE_NAME)
    for player in players:
        player.draw(win)

    def draw_stats_panel(panel_x, panel_y, rows, accent_color, align_right=False):
        row_gap = 6
        label_value_gap = 12
        pad_x = 12
        pad_y = 8
        row_metrics = []
        max_row_width = 0

        for label, value, value_color in rows:
            label_surface = stats_font.render(f"{label}:", True, (194, 202, 224))
            value_surface = stats_font.render(str(value), True, value_color)
            row_width = label_surface.get_width() + label_value_gap + value_surface.get_width()
            row_height = max(label_surface.get_height(), value_surface.get_height())
            row_metrics.append((label_surface, value_surface, row_width, row_height))
            max_row_width = max(max_row_width, row_width)

        panel_width = max_row_width + (pad_x * 2)
        panel_height = (
            sum(metric[3] for metric in row_metrics)
            + (row_gap * max(0, len(row_metrics) - 1))
            + (pad_y * 2)
        )

        if align_right:
            panel_x -= panel_width

        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(
            panel_surface,
            (9, 14, 31, 168),
            (0, 0, panel_width, panel_height),
            border_radius=10,
        )
        pygame.draw.rect(
            panel_surface,
            (accent_color[0], accent_color[1], accent_color[2], 112),
            (0, 0, panel_width, panel_height),
            1,
            border_radius=10,
        )

        text_y = pad_y
        for label_surface, value_surface, row_width, row_height in row_metrics:
            panel_surface.blit(label_surface, (pad_x, text_y))
            panel_surface.blit(
                value_surface,
                (panel_width - pad_x - value_surface.get_width(), text_y),
            )
            text_y += row_height + row_gap

        win.blit(panel_surface, (panel_x, panel_y))

    draw_stats_panel(
        10,
        10,
        [
            ("Gens", gen - 1, (245, 247, 255)),
            ("Alive", len(players), (245, 247, 255)),
            ("Profile", profile_visual["label"], profile_visual["ui_color"]),
        ],
        profile_visual["ui_color"],
    )
    draw_stats_panel(
        WIDTH - 10,
        10,
        [
            ("Level", level, (245, 247, 255)),
            ("Lives", lives_remaining, (245, 247, 255)),
        ],
        (196, 208, 255),
        align_right=True,
    )

    if boss is not None:
        boss_label = hud_font.render("Boss Fight", 1, (255, 150, 150))
        win.blit(boss_label, (WIDTH // 2 - boss_label.get_width() // 2, 12))

        bar_width = 260
        bar_height = 14
        bar_x = WIDTH // 2 - bar_width // 2
        bar_y = 46
        health_ratio = max(0.0, boss.health / boss.max_health)

        pygame.draw.rect(win, (65, 24, 24), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(win, (255, 82, 82), (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        pygame.draw.rect(win, (255, 220, 220), (bar_x, bar_y, bar_width, bar_height), 2)

    pygame.display.update()


def run_single_agent_episode(
    net,
    rng,
    reward_values=None,
    win=None,
    gen_label=0,
    draw_episode=False,
    max_frames=EVAL_MAX_FRAMES,
) -> EpisodeResult:
    fitness_delta = 0.0
    reward_totals = RewardTotals()
    event_totals = EventTotals()

    player = Player(300, 630)
    apply_profile_visuals_to_player(player, ACTIVE_REWARD_PROFILE_NAME)
    episode_state = EpisodeState(player=player)
    frame_count = 0

    while (
        episode_state.player.health > 0
        and episode_state.lives > 0
        and frame_count < max_frames
    ):
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        frame_result = step_frame(
            episode_state,
            net,
            rng,
            build_observation=build_observation,
            world_width=WIDTH,
            world_height=HEIGHT,
            reward_values=reward_values,
            event_totals=event_totals,
            reward_totals=reward_totals,
        )
        fitness_delta += frame_result.fitness_delta
        if frame_result.terminal:
            break

        if draw_episode and win is not None:
            draw_window(
                win,
                episode_state.enemies,
                [episode_state.player],
                gen_label,
                episode_state.level,
                episode_state.lives,
                frame_result.active_boss,
            )

    return EpisodeResult(
        fitness_delta=fitness_delta,
        frames=frame_count,
        lives_remaining=episode_state.lives,
        player_alive=int(episode_state.player.health > 0 and episode_state.lives > 0),
        event_totals=event_totals,
        reward_totals=reward_totals,
    )


def make_eval_genomes(win):
    generation_counter = {"value": 0}

    def eval_genomes(genomes, config):
        # Evaluate each genome in isolation over deterministic episodes.
        generation_counter["value"] += 1
        generation_label = generation_counter["value"]
        generation_index = generation_label - 1

        reward = ACTIVE_REWARD_PROFILE
        reward_totals = RewardTotals()
        event_totals = EventTotals()
        frame_count = 0
        survivors_at_end = 0
        average_lives_sum = 0.0
        fitness_values = []

        population_size = len(genomes)
        if population_size == 0:
            return

        for genome_position, (genome_id, genome) in enumerate(genomes):
            genome.fitness = 0.0
            net = neat.nn.RecurrentNetwork.create(genome, config)

            episode_fitness_values = []
            episode_lives_values = []
            last_episode_alive = 0

            for episode_index in range(EVAL_EPISODES_PER_GENOME):
                episode_seed = ((generation_index + 1) * 1_000_000) + (int(genome_id) * 1000) + episode_index
                rng = random.Random(episode_seed)
                draw_episode = genome_position == 0 and episode_index == 0
                episode_result = run_single_agent_episode(
                    net,
                    rng,
                    reward_values=reward,
                    win=win,
                    gen_label=generation_label,
                    draw_episode=draw_episode,
                    max_frames=EVAL_MAX_FRAMES,
                )

                episode_fitness_values.append(episode_result.fitness_delta)
                episode_lives_values.append(episode_result.lives_remaining)
                last_episode_alive = episode_result.player_alive
                frame_count += episode_result.frames

                event_totals.merge(episode_result.event_totals)
                reward_totals.merge(episode_result.reward_totals)

            genome.fitness = sum(episode_fitness_values) / EVAL_EPISODES_PER_GENOME
            fitness_values.append(genome.fitness)
            average_lives_sum += sum(episode_lives_values) / EVAL_EPISODES_PER_GENOME
            survivors_at_end += 1 if last_episode_alive else 0

        average_lives_remaining = int(round(average_lives_sum / population_size))
        avg_fitness = sum(fitness_values) / population_size
        best_fitness = max(fitness_values)
        worst_fitness = min(fitness_values)
        kill_per_shot = event_totals.kills / event_totals.shots_fired if event_totals.shots_fired > 0 else 0.0

        metrics_row = GenerationMetricsRow(
            generation=generation_index,
            frames=frame_count,
            population_size=population_size,
            survivors_at_end=survivors_at_end,
            lives_remaining=average_lives_remaining,
            avg_fitness=round(avg_fitness, 5),
            best_fitness=round(best_fitness, 5),
            worst_fitness=round(worst_fitness, 5),
            shots_fired=event_totals.shots_fired,
            kills=event_totals.kills,
            enemy_escapes=event_totals.enemy_escapes,
            player_deaths=event_totals.player_deaths,
            wave_clears=event_totals.wave_clears,
            level_failures=event_totals.level_failures,
            kill_per_shot=round(kill_per_shot, 5),
            survival_reward_total=round(reward_totals.survival_reward_total, 5),
            kill_reward_total=round(reward_totals.kill_reward_total, 5),
            wave_clear_reward_total=round(reward_totals.wave_clear_reward_total, 5),
            shot_penalty_total=round(reward_totals.shot_penalty_total, 5),
            death_penalty_total=round(reward_totals.death_penalty_total, 5),
            enemy_escape_penalty_total=round(reward_totals.enemy_escape_penalty_total, 5),
            level_fail_penalty_total=round(reward_totals.level_fail_penalty_total, 5),
        )
        append_training_metrics_row(metrics_row)

        print(
            f"[RewardLog][Profile {ACTIVE_REWARD_PROFILE_NAME}][Gen {generation_index}] "
            f"episodes/genome={EVAL_EPISODES_PER_GENOME} "
            f"shots={metrics_row.shots_fired} kills={metrics_row.kills} k/shot={metrics_row.kill_per_shot:.3f} "
            f"reward_totals(survival={metrics_row.survival_reward_total:.3f}, "
            f"kill={metrics_row.kill_reward_total:.3f}, wave={metrics_row.wave_clear_reward_total:.3f}, "
            f"shot={metrics_row.shot_penalty_total:.3f}, death={metrics_row.death_penalty_total:.3f}, "
            f"escape={metrics_row.enemy_escape_penalty_total:.3f}, fail={metrics_row.level_fail_penalty_total:.3f})"
        )

    return eval_genomes


def replay_saved_genome(config_file, genome_path):
    if not os.path.exists(genome_path):
        print(f"No saved genome found at: {genome_path}")
        print("Run a fresh training session first to create one.")
        return

    replay_profile = infer_profile_from_genome_path(genome_path)
    set_active_reward_profile(replay_profile)
    config = load_config(config_file)
    genome = load_best_genome(genome_path)
    net = neat.nn.RecurrentNetwork.create(genome, config)

    win = WIN
    clock = pygame.time.Clock()
    fps = 60
    player = Player(300, 630)
    apply_profile_visuals_to_player(player, ACTIVE_REWARD_PROFILE_NAME)
    episode_state = EpisodeState(player=player)

    print(f"Replaying best genome from: {genome_path}")
    run = True
    while run and episode_state.player.health > 0 and episode_state.lives > 0:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        if not run:
            break

        frame_result = step_frame(
            episode_state,
            net,
            random,
            build_observation=build_observation,
            world_width=WIDTH,
            world_height=HEIGHT,
            reward_values=None,
            event_totals=None,
            reward_totals=None,
        )

        draw_window(
            win,
            episode_state.enemies,
            [episode_state.player],
            0,
            episode_state.level,
            episode_state.lives,
            frame_result.active_boss,
        )

    if episode_state.player.health <= 0:
        print("Replay ended: player destroyed.")
    elif episode_state.lives <= 0:
        print("Replay ended: lives depleted.")


def evaluate_genome_episode(genome, config, seed, max_frames=3600):
    rng = random.Random(seed)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    episode_result = run_single_agent_episode(
        net,
        rng,
        reward_values=None,
        win=None,
        gen_label=0,
        draw_episode=False,
        max_frames=max_frames,
    )

    event_totals = episode_result.event_totals
    return GenomeEpisodeMetrics(
        seed=seed,
        frames_survived=episode_result.frames,
        kills=event_totals.kills,
        boss_kills=event_totals.boss_kills,
        wave_clears=event_totals.wave_clears,
        shots_fired=event_totals.shots_fired,
        laser_hits_taken=event_totals.laser_hits_taken,
        lives_remaining=episode_result.lives_remaining,
        player_alive=episode_result.player_alive,
    )


def benchmark_genome(genome, config, base_seed, episodes=3, max_frames=3600):
    episode_metrics = [
        evaluate_genome_episode(
            genome,
            config,
            seed=base_seed + (episode * 97),
            max_frames=max_frames,
        )
        for episode in range(episodes)
    ]

    avg_frames_survived = 0.0
    avg_kills = 0.0
    avg_boss_kills = 0.0
    avg_wave_clears = 0.0
    avg_shots_fired = 0.0
    avg_laser_hits_taken = 0.0
    avg_lives_remaining = 0.0
    total_shots = 0
    total_kills = 0
    for metrics in episode_metrics:
        avg_frames_survived += metrics.frames_survived
        avg_kills += metrics.kills
        avg_boss_kills += metrics.boss_kills
        avg_wave_clears += metrics.wave_clears
        avg_shots_fired += metrics.shots_fired
        avg_laser_hits_taken += metrics.laser_hits_taken
        avg_lives_remaining += metrics.lives_remaining
        total_shots += metrics.shots_fired
        total_kills += metrics.kills

    return BenchmarkMetrics(
        avg_frames_survived=round(avg_frames_survived / episodes, 4),
        avg_kills=round(avg_kills / episodes, 4),
        avg_boss_kills=round(avg_boss_kills / episodes, 4),
        avg_wave_clears=round(avg_wave_clears / episodes, 4),
        avg_shots_fired=round(avg_shots_fired / episodes, 4),
        avg_laser_hits_taken=round(avg_laser_hits_taken / episodes, 4),
        avg_lives_remaining=round(avg_lives_remaining / episodes, 4),
        kill_per_shot=round((total_kills / total_shots) if total_shots > 0 else 0.0, 4),
    )


def run_experiment(config_file, generations, base_seed=42, episodes=3, max_frames=3600):
    global ACTIVE_TRAINING_METRICS_PATH

    local_dir = os.path.dirname(__file__)
    results = []
    print(f"Starting reward-profile experiment ({len(REWARD_PROFILES)} profiles)...")

    for profile_index, profile_name in enumerate(REWARD_PROFILES):
        set_active_reward_profile(profile_name)
        ACTIVE_TRAINING_METRICS_PATH = os.path.join(local_dir, f"training_metrics_{profile_name}.csv")
        winner_path = os.path.join(local_dir, f"best_genome_{profile_name}.pkl")

        random.seed(base_seed + profile_index)
        print(f"\n=== Profile: {profile_name} ===")
        print(f"Metrics file: {ACTIVE_TRAINING_METRICS_PATH}")
        winner = run(
            config_file,
            best_genome_path=winner_path,
            generations=generations,
            reward_profile_name=profile_name,
            metrics_path=ACTIVE_TRAINING_METRICS_PATH,
        )
        config = load_config(config_file)
        benchmark = benchmark_genome(
            winner,
            config,
            base_seed=base_seed + (profile_index * 1000),
            episodes=episodes,
            max_frames=max_frames,
        )

        result = ExperimentResult(
            profile=profile_name,
            winner_fitness=round(winner.fitness, 5),
            winner_path=winner_path,
            benchmark=benchmark,
        )
        results.append(result)
        print(
            f"Benchmark {profile_name}: "
            f"avg_kills={result.benchmark.avg_kills}, avg_wave_clears={result.benchmark.avg_wave_clears}, "
            f"avg_frames={result.benchmark.avg_frames_survived}, kill_per_shot={result.benchmark.kill_per_shot}"
        )

    summary_fields = [
        "profile",
        "winner_fitness",
        "avg_frames_survived",
        "avg_kills",
        "avg_boss_kills",
        "avg_wave_clears",
        "avg_shots_fired",
        "avg_laser_hits_taken",
        "avg_lives_remaining",
        "kill_per_shot",
        "winner_path",
    ]
    with open(EXPERIMENT_SUMMARY_PATH, "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_summary_row())

    ranked = sorted(
        results,
        key=lambda item: item.ranking_key(),
        reverse=True,
    )
    print("\nExperiment ranking (best first):")
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank}. {result.profile} "
            f"(waves={result.benchmark.avg_wave_clears}, kills={result.benchmark.avg_kills}, "
            f"frames={result.benchmark.avg_frames_survived}, hits={result.benchmark.avg_laser_hits_taken})"
        )
    print(f"\nExperiment summary written to: {EXPERIMENT_SUMMARY_PATH}")

    set_active_reward_profile("kill_focus")
    ACTIVE_TRAINING_METRICS_PATH = TRAINING_METRICS_PATH
    return ranked


def run(config_file, best_genome_path=BEST_GENOME_PATH, generations=50, reward_profile_name=None, metrics_path=None):
    # runs the NEAT algorithm to train a neural network to play space invaders.
    # :param config_file: location of config file
    # :return: best genome

    global ACTIVE_TRAINING_METRICS_PATH
    if reward_profile_name is not None:
        set_active_reward_profile(reward_profile_name)
    ACTIVE_TRAINING_METRICS_PATH = metrics_path if metrics_path is not None else TRAINING_METRICS_PATH

    config = load_config(config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for the configured number of generations.
    winner = p.run(make_eval_genomes(WIN), generations)

    save_best_genome(winner, best_genome_path)
    print(f"\nSaved best genome to: {best_genome_path}")

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    return winner


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prompt", "fresh", "best"),
        default="prompt",
        help="Start mode: interactive prompt, fresh training, or replay saved best genome.",
    )
    parser.add_argument(
        "--genome-path",
        default=BEST_GENOME_PATH,
        help="Path for reading/writing best genome pickle.",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run all built-in reward profiles and benchmark which one produces the best genome.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Generation count for fresh mode (and per profile in experiment mode).",
    )
    parser.add_argument(
        "--experiment-episodes",
        type=int,
        default=3,
        help="Benchmark episodes per profile in experiment mode.",
    )
    parser.add_argument(
        "--experiment-max-frames",
        type=int,
        default=3600,
        help="Frame cap per benchmark episode in experiment mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for experiment profile runs and benchmarking.",
    )
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    if args.experiment:
        run_experiment(
            config_path,
            generations=args.generations,
            base_seed=args.seed,
            episodes=args.experiment_episodes,
            max_frames=args.experiment_max_frames,
        )
    else:
        start_mode = resolve_start_mode(args.mode)
        if start_mode == "best":
            replay_saved_genome(config_path, args.genome_path)
        else:
            run(
                config_path,
                best_genome_path=args.genome_path,
                generations=args.generations,
                reward_profile_name=ACTIVE_REWARD_PROFILE_NAME,
                metrics_path=TRAINING_METRICS_PATH,
            )
