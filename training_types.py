from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardProfile:
    survival_reward: float
    kill_reward: float
    boss_kill_reward: float
    wave_clear_reward: float
    shot_penalty: float
    laser_hit_penalty: float
    death_penalty: float
    enemy_escape_penalty: float
    level_fail_penalty: float


@dataclass
class EventTotals:
    shots_fired: int = 0
    kills: int = 0
    boss_kills: int = 0
    enemy_escapes: int = 0
    player_deaths: int = 0
    wave_clears: int = 0
    level_failures: int = 0
    laser_hits_taken: int = 0

    def merge(self, other):
        self.shots_fired += other.shots_fired
        self.kills += other.kills
        self.boss_kills += other.boss_kills
        self.enemy_escapes += other.enemy_escapes
        self.player_deaths += other.player_deaths
        self.wave_clears += other.wave_clears
        self.level_failures += other.level_failures
        self.laser_hits_taken += other.laser_hits_taken

    def as_dict(self):
        return asdict(self)


@dataclass
class RewardTotals:
    survival_reward_total: float = 0.0
    kill_reward_total: float = 0.0
    wave_clear_reward_total: float = 0.0
    shot_penalty_total: float = 0.0
    death_penalty_total: float = 0.0
    enemy_escape_penalty_total: float = 0.0
    level_fail_penalty_total: float = 0.0

    def merge(self, other):
        self.survival_reward_total += other.survival_reward_total
        self.kill_reward_total += other.kill_reward_total
        self.wave_clear_reward_total += other.wave_clear_reward_total
        self.shot_penalty_total += other.shot_penalty_total
        self.death_penalty_total += other.death_penalty_total
        self.enemy_escape_penalty_total += other.enemy_escape_penalty_total
        self.level_fail_penalty_total += other.level_fail_penalty_total

    def as_dict(self):
        return asdict(self)


@dataclass
class EpisodeResult:
    fitness_delta: float
    frames: int
    lives_remaining: int
    player_alive: int
    event_totals: EventTotals
    reward_totals: RewardTotals


@dataclass(frozen=True)
class GenerationMetricsRow:
    generation: int
    frames: int
    population_size: int
    survivors_at_end: int
    lives_remaining: int
    avg_fitness: float
    best_fitness: float
    worst_fitness: float
    shots_fired: int
    kills: int
    enemy_escapes: int
    player_deaths: int
    wave_clears: int
    level_failures: int
    kill_per_shot: float
    survival_reward_total: float
    kill_reward_total: float
    wave_clear_reward_total: float
    shot_penalty_total: float
    death_penalty_total: float
    enemy_escape_penalty_total: float
    level_fail_penalty_total: float

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def fieldnames():
        return [
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


@dataclass(frozen=True)
class GenomeEpisodeMetrics:
    seed: int
    frames_survived: int
    kills: int
    boss_kills: int
    wave_clears: int
    shots_fired: int
    laser_hits_taken: int
    lives_remaining: int
    player_alive: int


@dataclass(frozen=True)
class BenchmarkMetrics:
    avg_frames_survived: float
    avg_kills: float
    avg_boss_kills: float
    avg_wave_clears: float
    avg_shots_fired: float
    avg_laser_hits_taken: float
    avg_lives_remaining: float
    kill_per_shot: float

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    profile: str
    winner_fitness: float
    winner_path: str
    benchmark: BenchmarkMetrics

    def as_summary_row(self):
        return {
            "profile": self.profile,
            "winner_fitness": self.winner_fitness,
            **self.benchmark.as_dict(),
            "winner_path": self.winner_path,
        }

    def ranking_key(self):
        return (
            self.benchmark.avg_wave_clears,
            self.benchmark.avg_kills,
            self.benchmark.avg_frames_survived,
            -self.benchmark.avg_laser_hits_taken,
            self.benchmark.kill_per_shot,
        )
