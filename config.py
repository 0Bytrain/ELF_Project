from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

MODULATION_SCHEMES: List[str] = ["BPSK", "QPSK", "MSK"]
CODE_RATES: List[float] = [0.25, 0.50, 0.75, 8.0 / 9.0]
SPREADING_FACTORS: List[int] = [2, 4, 8]
POWER_LEVELS_DBM: List[int] = list(range(1, 17))








REFERENCE_DISTANCE_M: float = 10.0
REFERENCE_TX_POWER_DBM: float = 10.0
REFERENCE_RX_POWER_DBM: float = -84.0
REFERENCE_PATH_LOSS_DB: float = REFERENCE_TX_POWER_DBM - REFERENCE_RX_POWER_DBM

REFERENCE_PARAMETER_INDEX = {
    "modulation_idx": 1,
    "code_idx": 1,
    "spread_idx": 1,
    "power_idx": 6,
}








COMMUNICATION_MODES: Dict[str, Tuple[float, float, float]] = {
    "模式一_高速传输": (0.70, 0.20, 0.10),
    "模式二_高可靠": (0.10, 0.76, 0.14),
    "模式三_低功耗": (0.12, 0.10, 0.78),
    "模式四_均衡鲁棒": (0.42, 0.33, 0.25),
}













DEFAULT_SCENARIOS: List[Dict[str, float]] = [
    {"name": "场景A_近距浅海_3Hz", "frequency_hz": 3.0, "source_depth_m": 10.0, "receiver_depth_m": 120.0, "water_depth_m": 500.0, "ice_thickness_m": 2.0, "distance_m": 30.0, "noise_floor_dbm": -98.0},
    {"name": "场景B_中距浅海_3Hz", "frequency_hz": 3.0, "source_depth_m": 10.0, "receiver_depth_m": 120.0, "water_depth_m": 500.0, "ice_thickness_m": 2.0, "distance_m": 50.0, "noise_floor_dbm": -97.0},
    {"name": "场景C_远距浅海_3Hz", "frequency_hz": 3.0, "source_depth_m": 10.0, "receiver_depth_m": 120.0, "water_depth_m": 500.0, "ice_thickness_m": 2.0, "distance_m": 150.0, "noise_floor_dbm": -95.0},
    {"name": "场景D_中距较深海_5Hz", "frequency_hz": 5.0, "source_depth_m": 15.0, "receiver_depth_m": 135.0, "water_depth_m": 800.0, "ice_thickness_m": 2.0, "distance_m": 50.0, "noise_floor_dbm": -96.0},
    {"name": "场景E_中距浅层_2Hz", "frequency_hz": 2.0, "source_depth_m": 8.0, "receiver_depth_m": 120.0, "water_depth_m": 300.0, "ice_thickness_m": 2.0, "distance_m": 50.0, "noise_floor_dbm": -100.0},
    {"name": "场景F_远距深海_4Hz", "frequency_hz": 4.0, "source_depth_m": 18.0, "receiver_depth_m": 400.0, "water_depth_m": 1000.0, "ice_thickness_m": 2.0, "distance_m": 200.0, "noise_floor_dbm": -94.0},
]

EPS0 = 8.85e-12
MU0 = 4.0 * 3.141592653589793 * 1e-7
AIR_PARAMS = {"eps": EPS0, "mu": MU0, "sigma": 0.0}
SEA_PARAMS = {"eps": 81.0 * EPS0, "mu": MU0, "sigma": 4.0}
SEABED_PARAMS = {"eps": 8.0 * EPS0, "mu": MU0, "sigma": 0.4}


@dataclass
class TrainConfig:

    episodes_per_mode: int = 500


    max_steps_per_episode: int = 12

    batch_size: int = 192

    memory_size: int = 80000

    warmup_size: int = 1024

    updates_per_step: int = 1

    gamma: float = 0.98

    lr: float = 4.0e-5


    target_update_step: int = 1


    soft_tau: float = 0.01

    epsilon_start: float = 1.00

    epsilon_end: float = 0.02


    epsilon_decay_ratio: float = 0.82


    train_fixed_start_ratio: float = 0.55

    seed: int = 11

    evaluation_episodes: int = 48

    validation_interval: int = 10


    rolling_window: int = 25


    early_stop_patience: int = 10

    min_episodes_before_early_stop: int = 120


    checkpoint_metric_alpha: float = 0.30


    convergence_ratio: float = 0.985

    convergence_patience: int = 5


    convergence_std_tol: float = 0.008


    convergence_trend_tol: float = 0.002


    train_distance_jitter_ratio: float = 0.0

    train_frequency_jitter_ratio: float = 0.0


    train_depth_jitter_ratio: float = 0.0

    train_noise_jitter_db: float = 0.0


    eval_distance_jitter_ratio: float = 0.0

    eval_frequency_jitter_ratio: float = 0.0

    eval_depth_jitter_ratio: float = 0.0

    eval_noise_jitter_db: float = 0.0


    use_power_control: bool = True

    output_dir: str = "outputs_air_ice"

    decision_plot_mode_name: str = "模式二_高可靠"


    decision_plot_scenario_index: int = 1
    @property
    def model_dir(self) -> str:
        return os.path.join(self.output_dir, "models")

    @property
    def figure_dir(self) -> str:
        return os.path.join(self.output_dir, "figures")

    @property
    def history_dir(self) -> str:
        return os.path.join(self.output_dir, "history")














@dataclass
class EnvConfig:


    base_symbol_rate: float = 1.0



    system_margin_db: float = 0.0



    ber_floor: float = 1e-9



    target_ber: float = 1e-5



    local_window_m: float = 80.0



    local_num_points: int = 7



    max_distance_m: float = 200.0



    max_frequency_hz: float = 10.0



    max_depth_m: float = 1500.0



    max_path_loss_db: float = 180.0



    max_steps_per_episode: int = 12



    power_levels_dbm: List[int] = field(default_factory=lambda: POWER_LEVELS_DBM)



    use_power_control: bool = True



    reward_utility_weight: float = 0.80



    reward_improvement_weight: float = 0.24



    switch_penalty_weight: float = 0.025



    invalid_action_penalty: float = 0.04



    snr_violation_penalty_weight: float = 0.22



    ber_violation_penalty_weight: float = 0.22



    stability_bonus_weight: float = 0.025



    best_progress_bonus_weight: float = 0.10



    terminal_best_bonus_weight: float = 0.06



    power_ramp_penalty_weight: float = 0.05



    repeated_power_ramp_penalty_weight: float = 0.03



    reference_modulation_idx: int = REFERENCE_PARAMETER_INDEX["modulation_idx"]



    reference_code_idx: int = REFERENCE_PARAMETER_INDEX["code_idx"]



    reference_spread_idx: int = REFERENCE_PARAMETER_INDEX["spread_idx"]



    reference_power_idx: int = REFERENCE_PARAMETER_INDEX["power_idx"]






@dataclass
class TestConfig:
    output_dir: str = "outputs"
    save_decision_csv: bool = True
    evaluation_scenarios: List[Dict[str, float]] = field(default_factory=lambda: DEFAULT_SCENARIOS)








def ensure_output_dirs(train_cfg: TrainConfig) -> None:
    for path in [train_cfg.output_dir, train_cfg.model_dir, train_cfg.figure_dir, train_cfg.history_dir]:
        os.makedirs(path, exist_ok=True)
