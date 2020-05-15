# In this version of Space Invaders the player can only move left and right.
# The player loses when an enemy ship collides with him,
# the enemy hits the player with a laser, or when 5 enemy ships successfully lands on earth,
# i.e. surpasses the space ship of the player.


from Object import *
from Constants import *

pygame.font.init()
main_font = pygame.font.SysFont("comicsans", 50)
pygame.display.set_caption("Space Shooter Tutorial")

gen = 0
######## lives won't be reset to 5 after one generation has failed and the next one is started #######
######## lives = 0 does not effect the game at all yet ######
lives = 5


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
    score_label = main_font.render(f"Alive: {len(players)}", 1, (255, 255, 255))
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
    clock = pygame.time.Clock()

    FPS = 60
    main_font = pygame.font.SysFont("comicsans", 50)

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        for player in players:  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[players.index(player)].fitness += 0.1
            # bird.move()

            # send player location, first enemy location and second enemy location and determine from network whether to go right or not
            #output = nets[players.index(player)].activate((player.y, abs(player.y - enemies[0].y), abs(player.y - enemies[1].y)))
            output = [0.4, 0]

            # we use a tanh activation function so result will be between -1 and 1. if over 0.5 go right
            if output[0] > 0.5 and player.x + player.PLAYER_VEL + player.get_width() < WIDTH:
                player.move_right()

        if len(enemies) == 0:
            level += 1
            wave_length += 5
            for i in range(wave_length):
                enemy = Enemy(random.randrange(50, WIDTH-100),
                              random.randrange(-1500, -100), random.choice(["red", "blue", "green"]))
                enemies.append(enemy)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        for enemy in enemies:
            enemy.move()

            for player in players:
                enemy.move_lasers(player)
                if player.health <= 0:
                    ge[players.index(player)].fitness -= 1
                    nets.pop(players.index(player))
                    ge.pop(players.index(player))
                    players.pop(players.index(player))

                # checks for collusion between current enemy with current player
                # each player that collide with the current enemy will be remove
                if collide(enemy, player):
                    ge[players.index(player)].fitness -= 1
                    nets.pop(players.index(player))
                    ge.pop(players.index(player))
                    players.pop(players.index(player))
                    # the enemy has to be removed after every player has been checked for collision
                    # try:
                    #    enemies.remove(enemy)
                    # except:
                    #    pass

            if enemy.y + enemy.get_height() > HEIGHT:
                lives -= 1
                enemies.remove(enemy)
                # if lives == 0:
                #     for player in players:
                #         ge[players.index(player)].fitness -= 1
                #         nets.pop(players.index(player))
                #         ge.pop(players.index(player))
                #         players.pop(players.index(player))

        # for player in players:

        #     for enemy in enemies:
        #         enemy.move()
        #         enemy.move_lasers(player)

        #         if player.health <= 0:
        #             ge[players.index(player)].fitness -= 1
        #             nets.pop(players.index(player))
        #             ge.pop(players.index(player))
        #             players.pop(players.index(player))

        #         if collide(enemy, player):
        #             #player.health -= 100 # kills player instantly
        #             ge[players.index(player)].fitness -= 1
        #             nets.pop(players.index(player))
        #             ge.pop(players.index(player))
        #             players.pop(players.index(player))
        #             #enemies.remove(enemy) # does not work since it iterates over players
        #         elif enemy.y + enemy.get_height() > HEIGHT:
        #             lives -= 1
        #             ge[players.index(player)].fitness -= 1
        #             nets.pop(players.index(player))
        #             ge.pop(players.index(player))
        #             players.pop(players.index(player))
        #             #enemies.remove(enemy) # does not work since it iterates over players

            if random.randrange(0, 2*60) == 1:
                enemy.shoot()

        draw_window(win, enemies, players, gen, level)

        player.move_lasers(enemies)


def run(config_file):
    # runs the NEAT algorithm to train a neural network to play space invaders.
    # :param config_file: location of config file
    # :return: None

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
