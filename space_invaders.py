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

pygame.font.init()
main_font = pygame.font.SysFont("comicsans", 50)
hud_font = pygame.font.SysFont("comicsans", 30)
pygame.display.set_caption("Space Shooter Tutorial")

WIDTH, HEIGHT = 750, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
BG = pygame.transform.scale(pygame.image.load(
    os.path.join("assets", "background-black.png")), (WIDTH, HEIGHT))
TRAINING_METRICS_PATH = os.path.join(os.path.dirname(__file__), "training_metrics.csv")
BEST_GENOME_PATH = os.path.join(os.path.dirname(__file__), "best_genome.pkl")


def append_training_metrics_row(metrics_row):
    fieldnames = [
        "generation",
        "frames",
        "population_size",
        "survivors_at_end",
        "lives_remaining",
        "avg_fitness",
        "best_fitness",
        "worst_fitness",
        "shots_fired",
        "kills",
        "enemy_escapes",
        "player_deaths",
        "wave_clears",
        "level_failures",
        "kill_per_shot",
        "survival_reward_total",
        "kill_reward_total",
        "wave_clear_reward_total",
        "shot_penalty_total",
        "death_penalty_total",
        "enemy_escape_penalty_total",
        "level_fail_penalty_total",
    ]
    file_exists = os.path.exists(TRAINING_METRICS_PATH)
    with open(TRAINING_METRICS_PATH, "a", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_row)


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


def clamp_signed(value, scale):
    if scale == 0:
        return 0.0
    return max(-1.0, min(1.0, value / scale))


def build_observation(player, enemies):
    """Build richer, signed threat and self-state features for one player."""
    player_center_x = player.x + (player.get_width() / 2)
    player_center_y = player.y + (player.get_height() / 2)

    nearest_enemy_dist_sq = float("inf")
    nearest_enemy_dx = 0.0
    nearest_enemy_dy = 0.0

    nearest_laser_dist_sq = float("inf")
    nearest_laser_dx = 0.0
    nearest_laser_dy = HEIGHT

    nearest_own_laser_dy_abs = float("inf")
    nearest_own_laser_dy = 0.0
    for laser in player.lasers:
        laser_center_y = laser.y + (laser.img.get_height() / 2)
        own_laser_dy = laser_center_y - player_center_y
        if abs(own_laser_dy) < nearest_own_laser_dy_abs:
            nearest_own_laser_dy_abs = abs(own_laser_dy)
            nearest_own_laser_dy = own_laser_dy

    boss_present = 0.0
    for enemy in enemies:
        if isinstance(enemy, Boss):
            boss_present = 1.0

        enemy_center_x = enemy.x + (enemy.get_width() / 2)
        enemy_center_y = enemy.y + (enemy.get_height() / 2)
        enemy_dx = enemy_center_x - player_center_x
        enemy_dy = enemy_center_y - player_center_y
        enemy_dist_sq = (enemy_dx * enemy_dx) + (enemy_dy * enemy_dy)
        if enemy_dist_sq < nearest_enemy_dist_sq:
            nearest_enemy_dist_sq = enemy_dist_sq
            nearest_enemy_dx = enemy_dx
            nearest_enemy_dy = enemy_dy

        for laser in enemy.lasers:
            laser_center_x = laser.x + (laser.img.get_width() / 2)
            laser_center_y = laser.y + (laser.img.get_height() / 2)
            laser_dx = laser_center_x - player_center_x
            laser_dy = laser_center_y - player_center_y
            laser_dist_sq = (laser_dx * laser_dx) + (laser_dy * laser_dy)
            if laser_dist_sq < nearest_laser_dist_sq:
                nearest_laser_dist_sq = laser_dist_sq
                nearest_laser_dx = laser_dx
                nearest_laser_dy = laser_dy

    return (
        max(0.0, min(1.0, player_center_x / WIDTH)),
        max(0.0, min(1.0, player_center_y / HEIGHT)),
        clamp_signed(nearest_enemy_dx, WIDTH),
        clamp_signed(nearest_enemy_dy, HEIGHT),
        clamp_signed(nearest_laser_dx, WIDTH),
        clamp_signed(nearest_laser_dy, HEIGHT),
        min(1.0, len(enemies) / 20.0),
        boss_present,
        min(1.0, len(player.lasers) / 6.0),
        min(1.0, player.cool_down_counter / player.COOLDOWN),
        clamp_signed(nearest_own_laser_dy, HEIGHT),
    )


def draw_window(win, enemies, players, gen, level, boss=None):

    if gen == 0:
        gen = 1

    win.blit(BG, (0, 0))

    for enemy in enemies:
        enemy.draw(win)

    for player in players:
        player.draw(win)

    # generations label
    gen_label = main_font.render(f"Gens: {gen-1}", 1, (255, 255, 255))
    win.blit(gen_label, (10, 10))

    # alive players
    score_label = main_font.render(
        f"Alive: {len(players)}", 1, (255, 255, 255))
    win.blit(score_label, (10, 50))

    # level label
    level_label = main_font.render(f"Level: {level}", 1, (255, 255, 255))
    win.blit(level_label, (WIDTH - level_label.get_width() - 10, 10))

    # lives label
    lives_label = main_font.render(f"Lives: {lives}", 1, (255, 255, 255))
    WIN.blit(lives_label, (WIDTH - lives_label.get_width() - 10, 50))

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


def eval_genomes(genomes, config):
    # runs the simulation of the current population of
    # players and sets their fitness based on ...

    global WIN, gen, lives
    win = WIN
    gen += 1
    level = 0
    lives = 5
    clock = pygame.time.Clock()

    FPS = 60

    # Kill-focused shaping: strong upside for kills, strong downside for laser deaths/hits.
    SURVIVAL_REWARD = 0.005
    KILL_REWARD = 12.0
    BOSS_KILL_REWARD = 32.0
    WAVE_CLEAR_REWARD = 4.0
    SHOT_PENALTY = 0.005
    LASER_HIT_PENALTY = 10.0
    DEATH_PENALTY = 10.0
    ENEMY_ESCAPE_PENALTY = 1.0
    LEVEL_FAIL_PENALTY = 4.0

    reward_totals = {
        "survival_reward_total": 0.0,
        "kill_reward_total": 0.0,
        "wave_clear_reward_total": 0.0,
        "shot_penalty_total": 0.0,
        "death_penalty_total": 0.0,
        "enemy_escape_penalty_total": 0.0,
        "level_fail_penalty_total": 0.0,
    }
    event_totals = {
        "shots_fired": 0,
        "kills": 0,
        "enemy_escapes": 0,
        "player_deaths": 0,
        "wave_clears": 0,
        "level_failures": 0,
    }
    frame_count = 0

    def apply_reward(player_index, delta, reward_key):
        ge[player_index].fitness += delta
        reward_totals[reward_key] += delta

    enemies = []
    wave_length = 5
    boss_active = False

    def spawn_regular_wave():
        nonlocal level, wave_length, boss_active
        boss_active = False
        level += 1
        wave_length += 5
        invaders_to_spawn = max(1, int(round(wave_length * 0.8)))
        for _ in range(invaders_to_spawn):
            enemy = Enemy(
                random.randrange(50, WIDTH - 100),
                random.randrange(-1500, -100),
                random.choice(["red", "blue", "green"]),
            )
            enemies.append(enemy)

    def spawn_boss_wave():
        nonlocal boss_active
        boss_active = True
        boss_x = (WIDTH - Boss.BOSS_WIDTH) // 2
        enemies.append(Boss(boss_x, -Boss.BOSS_HEIGHT, level))

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # player object that uses that network to play
    nets = []
    players = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append(Player(300, 630))
        ge.append(genome)

    run = True
    while run and len(players) > 0:
        clock.tick(FPS)
        frame_count += 1

        # exit the game and end the run loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        # advance through phases: regular wave -> boss fight -> next regular wave
        if len(enemies) == 0:
            if boss_active:
                spawn_regular_wave()
            elif level > 0:
                event_totals["wave_clears"] += 1
                for i in range(len(ge)):
                    apply_reward(i, WAVE_CLEAR_REWARD, "wave_clear_reward_total")
                spawn_boss_wave()
            else:
                spawn_regular_wave()

        # the network decides movement/shooting from signed, normalized threat features
        for i, player in enumerate(players):
            # small survival reward each frame
            apply_reward(i, SURVIVAL_REWARD, "survival_reward_total")

            output = nets[i].activate(build_observation(player, enemies))

            # tanh outputs are in [-1, 1]
            if output[0] > 0.5 and player.x + player.PLAYER_VEL + player.get_width() < WIDTH:
                player.move_right()

            if output[1] > 0.5 and player.x - player.PLAYER_VEL > 0:
                player.move_left()

            if output[2] > 0.5:
                shots_before = len(player.lasers)
                player.shoot()
                if len(player.lasers) > shots_before:
                    event_totals["shots_fired"] += 1
                    apply_reward(i, -SHOT_PENALTY, "shot_penalty_total")

        # update player lasers once per frame so shooting/cooldown works during training
        for i, player in enumerate(players):
            player.cooldown()
            for laser in player.lasers[:]:
                laser.move(-1)

                if laser.off_screen(HEIGHT):
                    player.lasers.remove(laser)
                    continue

                for enemy in enemies[:]:
                    if laser.collision(enemy):
                        enemy.health -= 100
                        if laser in player.lasers:
                            player.lasers.remove(laser)
                        if enemy.health <= 0 and enemy in enemies:
                            enemies.remove(enemy)
                            event_totals["kills"] += 1
                            reward_delta = BOSS_KILL_REWARD if isinstance(enemy, Boss) else KILL_REWARD
                            apply_reward(i, reward_delta, "kill_reward_total")
                        break

        for enemy in enemies[:]:
            if isinstance(enemy, Boss):
                enemy.move(WIDTH)
                shoot_window = max(25, 50 - (level * 2))
            else:
                enemy.move()
                shoot_window = 2 * 60

            if random.randrange(0, shoot_window) == 1:
                enemy.shoot()

            enemy.cooldown()
            for laser in enemy.lasers[:]:
                laser.move(1)

                if laser.off_screen(HEIGHT):
                    enemy.lasers.remove(laser)
                    continue

                hit_player = False
                for i, player in enumerate(players):
                    if laser.collision(player):
                        player.health -= 100
                        apply_reward(i, -LASER_HIT_PENALTY, "death_penalty_total")
                        hit_player = True

                if hit_player and laser in enemy.lasers:
                    enemy.lasers.remove(laser)

            if enemy.y + enemy.get_height() > HEIGHT:
                lives_loss = 2 if isinstance(enemy, Boss) else 1
                lives -= lives_loss
                event_totals["enemy_escapes"] += 1
                for i in range(len(ge)):
                    apply_reward(i, -ENEMY_ESCAPE_PENALTY * lives_loss, "enemy_escape_penalty_total")
                enemies.remove(enemy)
                continue

            collided_players = [player for player in players if collide(enemy, player)]
            if collided_players:
                for player in collided_players:
                    player.health -= 100
                if not isinstance(enemy, Boss) and enemy in enemies:
                    enemies.remove(enemy)

        # remove dead players and associated genomes
        for i in range(len(players) - 1, -1, -1):
            if players[i].health <= 0:
                event_totals["player_deaths"] += 1
                apply_reward(i, -DEATH_PENALTY, "death_penalty_total")
                nets.pop(i)
                ge.pop(i)
                players.pop(i)

        if lives <= 0:
            event_totals["level_failures"] += 1
            for i in range(len(players) - 1, -1, -1):
                apply_reward(i, -LEVEL_FAIL_PENALTY, "level_fail_penalty_total")
                nets.pop(i)
                ge.pop(i)
                players.pop(i)

        active_boss = next((enemy for enemy in enemies if isinstance(enemy, Boss)), None)
        draw_window(win, enemies, players, gen, level, active_boss)

    generation_index = gen - 1
    population_size = len(genomes)
    fitness_values = [genome.fitness for _, genome in genomes]
    avg_fitness = sum(fitness_values) / population_size
    best_fitness = max(fitness_values)
    worst_fitness = min(fitness_values)
    kill_per_shot = event_totals["kills"] / event_totals["shots_fired"] if event_totals["shots_fired"] > 0 else 0.0

    metrics_row = {
        "generation": generation_index,
        "frames": frame_count,
        "population_size": population_size,
        "survivors_at_end": len(players),
        "lives_remaining": lives,
        "avg_fitness": round(avg_fitness, 5),
        "best_fitness": round(best_fitness, 5),
        "worst_fitness": round(worst_fitness, 5),
        "shots_fired": event_totals["shots_fired"],
        "kills": event_totals["kills"],
        "enemy_escapes": event_totals["enemy_escapes"],
        "player_deaths": event_totals["player_deaths"],
        "wave_clears": event_totals["wave_clears"],
        "level_failures": event_totals["level_failures"],
        "kill_per_shot": round(kill_per_shot, 5),
        "survival_reward_total": round(reward_totals["survival_reward_total"], 5),
        "kill_reward_total": round(reward_totals["kill_reward_total"], 5),
        "wave_clear_reward_total": round(reward_totals["wave_clear_reward_total"], 5),
        "shot_penalty_total": round(reward_totals["shot_penalty_total"], 5),
        "death_penalty_total": round(reward_totals["death_penalty_total"], 5),
        "enemy_escape_penalty_total": round(reward_totals["enemy_escape_penalty_total"], 5),
        "level_fail_penalty_total": round(reward_totals["level_fail_penalty_total"], 5),
    }
    append_training_metrics_row(metrics_row)

    print(
        f"[RewardLog][Gen {generation_index}] "
        f"shots={metrics_row['shots_fired']} kills={metrics_row['kills']} k/shot={metrics_row['kill_per_shot']:.3f} "
        f"reward_totals(survival={metrics_row['survival_reward_total']:.3f}, "
        f"kill={metrics_row['kill_reward_total']:.3f}, wave={metrics_row['wave_clear_reward_total']:.3f}, "
        f"shot={metrics_row['shot_penalty_total']:.3f}, death={metrics_row['death_penalty_total']:.3f}, "
        f"escape={metrics_row['enemy_escape_penalty_total']:.3f}, fail={metrics_row['level_fail_penalty_total']:.3f})"
    )


def replay_saved_genome(config_file, genome_path):
    global WIN, lives

    if not os.path.exists(genome_path):
        print(f"No saved genome found at: {genome_path}")
        print("Run a fresh training session first to create one.")
        return

    config = load_config(config_file)
    genome = load_best_genome(genome_path)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    win = WIN
    level = 0
    lives = 5
    clock = pygame.time.Clock()
    fps = 60
    player = Player(300, 630)

    enemies = []
    wave_length = 5
    boss_active = False

    def spawn_regular_wave():
        nonlocal level, wave_length, boss_active
        boss_active = False
        level += 1
        wave_length += 5
        invaders_to_spawn = max(1, int(round(wave_length * 0.8)))
        for _ in range(invaders_to_spawn):
            enemy = Enemy(
                random.randrange(50, WIDTH - 100),
                random.randrange(-1500, -100),
                random.choice(["red", "blue", "green"]),
            )
            enemies.append(enemy)

    def spawn_boss_wave():
        nonlocal boss_active
        boss_active = True
        boss_x = (WIDTH - Boss.BOSS_WIDTH) // 2
        enemies.append(Boss(boss_x, -Boss.BOSS_HEIGHT, level))

    print(f"Replaying best genome from: {genome_path}")
    run = True
    while run and player.health > 0 and lives > 0:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        if not run:
            break

        if len(enemies) == 0:
            if boss_active:
                spawn_regular_wave()
            elif level > 0:
                spawn_boss_wave()
            else:
                spawn_regular_wave()

        output = net.activate(build_observation(player, enemies))
        if output[0] > 0.5 and player.x + player.PLAYER_VEL + player.get_width() < WIDTH:
            player.move_right()
        if output[1] > 0.5 and player.x - player.PLAYER_VEL > 0:
            player.move_left()
        if output[2] > 0.5:
            player.shoot()

        player.cooldown()
        for laser in player.lasers[:]:
            laser.move(-1)

            if laser.off_screen(HEIGHT):
                player.lasers.remove(laser)
                continue

            for enemy in enemies[:]:
                if laser.collision(enemy):
                    enemy.health -= 100
                    if laser in player.lasers:
                        player.lasers.remove(laser)
                    if enemy.health <= 0 and enemy in enemies:
                        enemies.remove(enemy)
                    break

        for enemy in enemies[:]:
            if isinstance(enemy, Boss):
                enemy.move(WIDTH)
                shoot_window = max(25, 50 - (level * 2))
            else:
                enemy.move()
                shoot_window = 2 * 60

            if random.randrange(0, shoot_window) == 1:
                enemy.shoot()

            enemy.cooldown()
            for laser in enemy.lasers[:]:
                laser.move(1)

                if laser.off_screen(HEIGHT):
                    enemy.lasers.remove(laser)
                    continue

                if laser.collision(player):
                    player.health -= 100
                    if laser in enemy.lasers:
                        enemy.lasers.remove(laser)

            if enemy.y + enemy.get_height() > HEIGHT:
                lives -= 2 if isinstance(enemy, Boss) else 1
                enemies.remove(enemy)
                continue

            if collide(enemy, player):
                player.health -= 100
                if not isinstance(enemy, Boss) and enemy in enemies:
                    enemies.remove(enemy)

        active_boss = next((enemy for enemy in enemies if isinstance(enemy, Boss)), None)
        draw_window(win, enemies, [player], 0, level, active_boss)

    if player.health <= 0:
        print("Replay ended: player destroyed.")
    elif lives <= 0:
        print("Replay ended: lives depleted.")


def run(config_file, best_genome_path=BEST_GENOME_PATH):
    # runs the NEAT algorithm to train a neural network to play space invaders.
    # :param config_file: location of config file
    # :return: best genome

    global gen
    gen = 0

    config = load_config(config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

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
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    start_mode = resolve_start_mode(args.mode)
    if start_mode == "best":
        replay_saved_genome(config_path, args.genome_path)
    else:
        run(config_path, best_genome_path=args.genome_path)
