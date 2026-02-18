"""One-shot screenshot helper – run under xvfb-run."""
import os
import sys
import random
import pickle

# Must be set before pygame (and therefore space_invaders) is imported
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import pygame
import neat

# ── bootstrap pygame early so the module-level WIN is created cleanly ────────
pygame.init()
pygame.font.init()

# We import the game module *after* pygame.init() so its module-level
# pygame.display.set_mode() call works under the virtual framebuffer.
sys.path.insert(0, os.path.dirname(__file__))
from space_invaders import (
    WIN, WIDTH, HEIGHT,
    load_config, load_best_genome,
    draw_window, build_observation,
    apply_profile_visuals_to_player,
    infer_profile_from_genome_path,
    set_active_reward_profile,
    BEST_GENOME_PATH,
)
from Object import Player
from engine import EpisodeState, step_frame

SCREENSHOT_PATH = os.path.join(os.path.dirname(__file__), "assets", "screenshot.png")
CONFIG_PATH     = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
GENOME_PATH     = BEST_GENOME_PATH
MAX_FRAMES      = 800   # run at most this many frames


def count_visible_enemies(enemies, height):
    """Count enemies that are currently visible on screen."""
    return sum(1 for e in enemies if 0 <= e.y <= height)


def main():
    if not os.path.exists(GENOME_PATH):
        print(f"No saved genome at {GENOME_PATH}. Run training first.")
        sys.exit(1)

    replay_profile = infer_profile_from_genome_path(GENOME_PATH)
    set_active_reward_profile(replay_profile)

    config  = load_config(CONFIG_PATH)
    genome  = load_best_genome(GENOME_PATH)
    net     = neat.nn.FeedForwardNetwork.create(genome, config)

    player  = Player(300, 630)
    apply_profile_visuals_to_player(player, replay_profile)
    episode = EpisodeState(player=player)
    rng     = random.Random(42)

    clock = pygame.time.Clock()

    best_frame_surface = None
    best_enemy_count   = -1

    frame = 0
    while episode.player.health > 0 and episode.lives > 0 and frame < MAX_FRAMES:
        # drain the event queue so pygame doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        frame_result = step_frame(
            episode,
            net,
            rng,
            build_observation=build_observation,
            world_width=WIDTH,
            world_height=HEIGHT,
            reward_values=None,
            event_totals=None,
            reward_totals=None,
        )

        draw_window(
            WIN,
            episode.enemies,
            [episode.player],
            1,
            episode.level,
            episode.lives,
            frame_result.active_boss,
        )

        frame += 1

        # Track the frame with the most visible enemies (after a brief warm-up)
        if frame >= 60:
            visible = count_visible_enemies(episode.enemies, HEIGHT)
            if visible > best_enemy_count:
                best_enemy_count = visible
                best_frame_surface = WIN.copy()

        if frame_result.terminal:
            break

        clock.tick(120)   # run fast – we don't need real-time pacing

    if best_frame_surface is not None:
        pygame.image.save(best_frame_surface, SCREENSHOT_PATH)
        print(f"Screenshot saved to {SCREENSHOT_PATH} (frame with {best_enemy_count} visible enemies)")
    else:
        pygame.image.save(WIN, SCREENSHOT_PATH)
        print(f"Screenshot saved to {SCREENSHOT_PATH} (fallback)")

    pygame.quit()


if __name__ == "__main__":
    main()
