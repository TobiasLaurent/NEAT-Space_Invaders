import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

from Object import Boss
from engine import EpisodeState, StepResult, step_frame
from training_types import EventTotals, RewardProfile, RewardTotals


def make_reward_profile(**overrides):
    profile_dict = {
        "survival_reward": 0.0,
        "kill_reward": 0.0,
        "boss_kill_reward": 0.0,
        "wave_clear_reward": 0.0,
        "shot_penalty": 0.0,
        "laser_hit_penalty": 0.0,
        "death_penalty": 0.0,
        "enemy_escape_penalty": 0.0,
        "level_fail_penalty": 0.0,
    }
    profile_dict.update(overrides)
    return RewardProfile(**profile_dict)


def make_event_totals():
    return EventTotals()


def make_reward_totals():
    return RewardTotals()


def zero_observation(_player, _enemies):
    return (0.0,) * 11


class DummyNet:
    def __init__(self, output=(0.0, 0.0, 0.0)):
        self.output = output

    def activate(self, _observation):
        return self.output


class DummyLaser:
    def __init__(self, collision_target=None, off_screen=False):
        self.collision_target = collision_target
        self.off_screen_value = off_screen
        self.img = type("Img", (), {"get_height": lambda self: 1, "get_width": lambda self: 1})()
        self.y = 0
        self.x = 0

    def move(self, _direction):
        return None

    def off_screen(self, _height):
        return self.off_screen_value

    def collision(self, obj):
        return obj is self.collision_target


class DummyPlayer:
    PLAYER_VEL = 5
    COOLDOWN = 30

    def __init__(self):
        self.x = 100
        self.y = 100
        self.health = 100
        self.lasers = []
        self.cool_down_counter = 0
        self.mask = pygame.mask.Mask((1, 1), fill=False)

    def get_width(self):
        return 20

    def get_height(self):
        return 20

    def move_right(self):
        self.x += self.PLAYER_VEL

    def move_left(self):
        self.x -= self.PLAYER_VEL

    def shoot(self):
        if self.cool_down_counter == 0:
            self.lasers.append(DummyLaser())
            self.cool_down_counter = 1

    def cooldown(self):
        if self.cool_down_counter >= self.COOLDOWN:
            self.cool_down_counter = 0
        elif self.cool_down_counter > 0:
            self.cool_down_counter += 1


class DummyEnemy:
    def __init__(self, health=100, y=0, height=10, lasers=None):
        self.x = 200
        self.y = y
        self.health = health
        self._height = height
        self.lasers = list(lasers) if lasers else []
        self.mask = pygame.mask.Mask((1, 1), fill=False)

    def move(self):
        return None

    def shoot(self):
        return None

    def cooldown(self):
        return None

    def get_height(self):
        return self._height

    def get_width(self):
        return 20


class StepFrameTests(unittest.TestCase):
    def setUp(self):
        self.width = 750
        self.height = 750

    def test_spawns_regular_wave_from_empty_state(self):
        state = EpisodeState(player=DummyPlayer())
        result = step_frame(
            state,
            DummyNet(),
            random.Random(7),
            build_observation=zero_observation,
            world_width=self.width,
            world_height=self.height,
            reward_values=None,
            event_totals=None,
            reward_totals=None,
        )

        self.assertIsInstance(result, StepResult)
        self.assertEqual(state.level, 1)
        self.assertEqual(state.wave_length, 10)
        self.assertGreater(len(state.enemies), 0)

    def test_spawns_boss_wave_and_awards_wave_clear(self):
        state = EpisodeState(player=DummyPlayer(), level=1, wave_length=10, boss_active=False, lives=5)
        event_totals = make_event_totals()
        reward_totals = make_reward_totals()

        result = step_frame(
            state,
            DummyNet(),
            random.Random(11),
            build_observation=zero_observation,
            world_width=self.width,
            world_height=self.height,
            reward_values=make_reward_profile(wave_clear_reward=3.5),
            event_totals=event_totals,
            reward_totals=reward_totals,
        )

        self.assertFalse(result.terminal)
        self.assertTrue(state.boss_active)
        self.assertEqual(event_totals.wave_clears, 1)
        self.assertEqual(len(state.enemies), 1)
        self.assertIsInstance(state.enemies[0], Boss)
        self.assertAlmostEqual(reward_totals.wave_clear_reward_total, 3.5)

    def test_player_laser_kill_updates_counts_and_rewards(self):
        player = DummyPlayer()
        enemy = DummyEnemy(health=100)
        player.lasers = [DummyLaser(collision_target=enemy)]
        state = EpisodeState(player=player, enemies=[enemy], level=1, wave_length=10, lives=5)
        event_totals = make_event_totals()
        reward_totals = make_reward_totals()

        result = step_frame(
            state,
            DummyNet(),
            random.Random(3),
            build_observation=zero_observation,
            world_width=self.width,
            world_height=self.height,
            reward_values=make_reward_profile(kill_reward=7.0),
            event_totals=event_totals,
            reward_totals=reward_totals,
        )

        self.assertFalse(result.terminal)
        self.assertEqual(event_totals.kills, 1)
        self.assertEqual(len(state.enemies), 0)
        self.assertAlmostEqual(reward_totals.kill_reward_total, 7.0)

    def test_enemy_laser_hit_can_kill_player_and_end_episode(self):
        player = DummyPlayer()
        enemy = DummyEnemy(health=100, lasers=[DummyLaser(collision_target=player)])
        state = EpisodeState(player=player, enemies=[enemy], level=1, wave_length=10, lives=5)
        event_totals = make_event_totals()
        reward_totals = make_reward_totals()

        result = step_frame(
            state,
            DummyNet(),
            random.Random(5),
            build_observation=zero_observation,
            world_width=self.width,
            world_height=self.height,
            reward_values=make_reward_profile(laser_hit_penalty=2.0, death_penalty=9.0),
            event_totals=event_totals,
            reward_totals=reward_totals,
        )

        self.assertTrue(result.terminal)
        self.assertEqual(player.health, 0)
        self.assertEqual(event_totals.laser_hits_taken, 1)
        self.assertEqual(event_totals.player_deaths, 1)
        self.assertAlmostEqual(reward_totals.death_penalty_total, -11.0)

    def test_enemy_escape_can_trigger_level_failure(self):
        player = DummyPlayer()
        enemy = DummyEnemy(y=745, height=10)
        state = EpisodeState(player=player, enemies=[enemy], level=1, wave_length=10, lives=1)
        event_totals = make_event_totals()
        reward_totals = make_reward_totals()

        result = step_frame(
            state,
            DummyNet(),
            random.Random(13),
            build_observation=zero_observation,
            world_width=self.width,
            world_height=self.height,
            reward_values=make_reward_profile(enemy_escape_penalty=1.5, level_fail_penalty=4.0),
            event_totals=event_totals,
            reward_totals=reward_totals,
        )

        self.assertTrue(result.terminal)
        self.assertEqual(state.lives, 0)
        self.assertEqual(event_totals.enemy_escapes, 1)
        self.assertEqual(event_totals.level_failures, 1)
        self.assertAlmostEqual(reward_totals.enemy_escape_penalty_total, -1.5)
        self.assertAlmostEqual(reward_totals.level_fail_penalty_total, -4.0)


if __name__ == "__main__":
    unittest.main()
