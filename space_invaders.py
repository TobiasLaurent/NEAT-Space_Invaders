# In this version of Space Invaders the player can only move left and right.
# The player loses when an enemy ship collides with him,
# the enemy hits the player with a laser, or when 5
# enemy ships successfully lands on earth,
# i.e. surpasses the space ship of the player.
import pygame
import os
import csv
# import time
import random
import neat

from Object import *

pygame.font.init()
main_font = pygame.font.SysFont("comicsans", 50)
pygame.display.set_caption("Space Shooter Tutorial")

WIDTH, HEIGHT = 750, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
BG = pygame.transform.scale(pygame.image.load(
    os.path.join("assets", "background-black.png")), (WIDTH, HEIGHT))
TRAINING_METRICS_PATH = os.path.join(os.path.dirname(__file__), "training_metrics.csv")


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


def draw_window(win, enemies, players, gen, level):

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

    # Reward shaping tuned to prioritize meaningful combat over passive stalling.
    SURVIVAL_REWARD = 0.02
    KILL_REWARD = 4.0
    WAVE_CLEAR_REWARD = 2.0
    SHOT_PENALTY = 0.01
    DEATH_PENALTY = 4.0
    ENEMY_ESCAPE_PENALTY = 0.75
    LEVEL_FAIL_PENALTY = 2.0

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

        # creates new wave of enemies once all enemies of the previos wave have been removed
        if len(enemies) == 0:
            if level > 0:
                event_totals["wave_clears"] += 1
                for i in range(len(ge)):
                    apply_reward(i, WAVE_CLEAR_REWARD, "wave_clear_reward_total")
            level += 1
            wave_length += 5
            for i in range(wave_length):
                enemy = Enemy(random.randrange(50, WIDTH - 100),
                              random.randrange(-1500, -100), random.choice(["red", "blue", "green"]))
                enemies.append(enemy)

        # TODO: does this really work????
        # determine:
        # 1. the closest enemy and set them as target index for the network
        # 2. the closest laser and save the distance to the player in y-direction into dist_laser
        target_ind = 0
        # avoid_ind = 0
        dist_target = 2200     # max dist between players and spawned enemies is 2130
        dist_laser = 2200       # max dist between players and spawned lasers is 2130
        if len(enemies) > 0:
            for enemy in enemies:
                if abs(630 - enemy.y) < dist_target:
                    dist_target = abs(630 - enemy.y)
                    target_ind = enemies.index(enemy)
                for laser in enemy.lasers:
                    if abs(630 - laser.y) < dist_laser:
                        dist_laser = abs(630 - laser.y)
                        # avoid_ind = enemies.index(enemy)

        # the network determines whether to go left, right or shoot based on the nearest enemy and laser
        for i, player in enumerate(players):
            # small survival reward each frame
            apply_reward(i, SURVIVAL_REWARD, "survival_reward_total")

            # send player location, nearest enemy x-distance and nearest enemy-laser y-distance
            output = nets[i].activate((player.y, abs(player.x - enemies[target_ind].x), dist_laser))

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
                            apply_reward(i, KILL_REWARD, "kill_reward_total")
                        break

        for enemy in enemies[:]:
            enemy.move()

            if random.randrange(0, 2 * 60) == 1:
                enemy.shoot()

            enemy.cooldown()
            for laser in enemy.lasers[:]:
                laser.move(1)

                if laser.off_screen(HEIGHT):
                    enemy.lasers.remove(laser)
                    continue

                hit_player = False
                for player in players:
                    if laser.collision(player):
                        player.health -= 100
                        hit_player = True

                if hit_player and laser in enemy.lasers:
                    enemy.lasers.remove(laser)

            if enemy.y + enemy.get_height() > HEIGHT:
                lives -= 1
                event_totals["enemy_escapes"] += 1
                for i in range(len(ge)):
                    apply_reward(i, -ENEMY_ESCAPE_PENALTY, "enemy_escape_penalty_total")
                enemies.remove(enemy)
                continue

            collided_players = [player for player in players if collide(enemy, player)]
            if collided_players:
                for player in collided_players:
                    player.health -= 100
                if enemy in enemies:
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

        draw_window(win, enemies, players, gen, level)

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


def run(config_file):
    # runs the NEAT algorithm to train a neural network to play space invaders.
    # :param config_file: location of config file
    # :return: None

    global gen
    gen = 0

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
