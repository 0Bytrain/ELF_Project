from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import random
import numpy as np
from config import (
    CODE_RATES,
    COMMUNICATION_MODES,
    DEFAULT_SCENARIOS,
    MODULATION_SCHEMES,
    POWER_LEVELS_DBM,
    SPREADING_FACTORS,
    EnvConfig,
)

from air_ice_marine import get_local_channel_snapshot
from performance import compute_communication_performance
@dataclass(frozen=True)
class ParameterState:




    modulation_idx: int           
    code_idx: int                
    spread_idx: int               
    power_idx: int                


@dataclass(frozen=True)
class LocalAction:




    name: str
    modulation_delta: int = 0
    code_delta: int = 0
    spread_delta: int = 0
    power_delta: int = 0


class MarineWaveformEnv:


    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        scenarios: Optional[List[Dict[str, float]]] = None
    ):

        self.cfg = env_cfg or EnvConfig()


        self.scenarios = scenarios or DEFAULT_SCENARIOS


        self.mode_names = list(COMMUNICATION_MODES.keys())


        self.modulations = MODULATION_SCHEMES
        self.code_rates = CODE_RATES
        self.spreading_factors = SPREADING_FACTORS



        self.power_levels_dbm = POWER_LEVELS_DBM if self.cfg.use_power_control else [10]



        self.reference_param_state = ParameterState(
            modulation_idx=min(self.cfg.reference_modulation_idx, len(self.modulations) - 1),
            code_idx=min(self.cfg.reference_code_idx, len(self.code_rates) - 1),
            spread_idx=min(self.cfg.reference_spread_idx, len(self.spreading_factors) - 1),
            power_idx=min(self.cfg.reference_power_idx, len(self.power_levels_dbm) - 1),
        )


        self.actions: List[LocalAction] = self._build_action_table()


        self.action_dim = len(self.actions)


        self.hold_action_id = 0


        self.state_dim = 29


        self.rng = random.Random(1234)


        self.current_mode_name: str = self.mode_names[0]


        self.current_mode_weights = COMMUNICATION_MODES[self.current_mode_name]


        self.current_scenario: Dict[str, float] = copy.deepcopy(self.scenarios[0])


        self.current_snapshot: Dict[str, float] = {}


        self.current_param_state = self.reference_param_state


        self.current_metrics: Dict[str, float] = {}


        self.step_count = 0


        self.episode_best_c = 0.0


        self.previous_action_id = self.hold_action_id


        self.consecutive_power_up_count = 0

    def seed(self, seed_value: int) -> None:



        self.rng.seed(seed_value)

    def _build_action_table(self) -> List[LocalAction]:







        return [
            LocalAction("保持"),
            LocalAction("调制降一档", modulation_delta=-1),
            LocalAction("调制升一档", modulation_delta=1),
            LocalAction("编码降一档", code_delta=-1),
            LocalAction("编码升一档", code_delta=1),
            LocalAction("扩频降一档", spread_delta=-1),
            LocalAction("扩频升一档", spread_delta=1),
            LocalAction("功率降一档", power_delta=-1),
            LocalAction("功率升一档", power_delta=1),
        ]

    def _parameter_dict(self, param_state: Optional[ParameterState] = None) -> Dict[str, float]:







        state = param_state or self.current_param_state
        return {
            "modulation": self.modulations[state.modulation_idx],
            "code_rate": self.code_rates[state.code_idx],
            "spread_factor": self.spreading_factors[state.spread_idx],
            "tx_power_dbm": self.power_levels_dbm[state.power_idx],
        }

    def _apply_action(self, param_state: ParameterState, action: LocalAction) -> ParameterState:





        return ParameterState(
            modulation_idx=int(np.clip(
                param_state.modulation_idx + action.modulation_delta,
                0,
                len(self.modulations) - 1
            )),
            code_idx=int(np.clip(
                param_state.code_idx + action.code_delta,
                0,
                len(self.code_rates) - 1
            )),
            spread_idx=int(np.clip(
                param_state.spread_idx + action.spread_delta,
                0,
                len(self.spreading_factors) - 1
            )),
            power_idx=int(np.clip(
                param_state.power_idx + action.power_delta,
                0,
                len(self.power_levels_dbm) - 1
            )),
        )

    def _normalized_switch_distance(self, prev_state: ParameterState, next_state: ParameterState) -> float:











        diffs = [
            abs(next_state.modulation_idx - prev_state.modulation_idx) / max(1, len(self.modulations) - 1),
            abs(next_state.code_idx - prev_state.code_idx) / max(1, len(self.code_rates) - 1),
            abs(next_state.spread_idx - prev_state.spread_idx) / max(1, len(self.spreading_factors) - 1),
            abs(next_state.power_idx - prev_state.power_idx) / max(1, len(self.power_levels_dbm) - 1),
        ]
        return float(np.mean(diffs))

    def _evaluate_param_state(self, param_state: ParameterState) -> Dict[str, float]:








        params = self._parameter_dict(param_state)
        return compute_communication_performance(
            snapshot=self.current_snapshot,
            weights=self.current_mode_weights,
            modulation=params["modulation"],
            code_rate=params["code_rate"],
            spread_factor=params["spread_factor"],
            tx_power_dbm=params["tx_power_dbm"],
            system_margin_db=self.cfg.system_margin_db,
            ber_floor=self.cfg.ber_floor,
            target_ber=self.cfg.target_ber,
            channel_window_m=self.cfg.local_window_m,
        )

    def _build_state_vector(self) -> np.ndarray:

















        def squash_signed(x: float, center: float = 0.0, scale: float = 10.0) -> float:






            return float(0.5 * (1.0 + np.tanh((x - center) / max(scale, 1e-6))))

        def squash_nonnegative(x: float, scale: float = 10.0) -> float:




            x = max(0.0, float(x))
            return float(x / (x + max(scale, 1e-6)))

        def ratio01(x: float, upper: float) -> float:




            return float(np.clip(x / max(upper, 1e-9), 0.0, 1.0))

        snapshot = self.current_snapshot
        metrics = self.current_metrics
        w1, w2, w3 = self.current_mode_weights




        ber_log = -np.log10(max(metrics["ber"], 1e-12))
        ber_feature = squash_nonnegative(ber_log, scale=3.0)

        state = np.array([




            float(w1),
            float(w2),
            float(w3),





            ratio01(self.current_scenario["frequency_hz"], self.cfg.max_frequency_hz),
            ratio01(self.current_scenario["distance_m"], self.cfg.max_distance_m),
            ratio01(self.current_scenario["source_depth_m"], self.cfg.max_depth_m),
            ratio01(self.current_scenario["receiver_depth_m"], self.cfg.max_depth_m),
            ratio01(self.current_scenario["water_depth_m"], self.cfg.max_depth_m),


            squash_signed(self.current_scenario["noise_floor_dbm"], center=-100.0, scale=10.0),






            ratio01(snapshot["path_loss_db"], self.cfg.max_path_loss_db),
            ratio01(snapshot["path_loss_mean_db"], self.cfg.max_path_loss_db),


            squash_nonnegative(snapshot["path_loss_std_db"], scale=10.0),


            squash_signed(snapshot["slope_db_per_km"], center=0.0, scale=20.0),


            squash_signed(snapshot["noise_floor_dbm"], center=-100.0, scale=10.0),


            squash_signed(snapshot["direct_db"], center=-20.0, scale=10.0),
            squash_signed(snapshot["lateral_db"], center=-20.0, scale=10.0),
            squash_signed(snapshot["reflected_db"], center=-20.0, scale=10.0),





            float(np.clip(metrics["f_rate"], 0.0, 1.0)),
            float(np.clip(metrics["f_ber"], 0.0, 1.0)),
            float(np.clip(metrics["f_power"], 0.0, 1.0)),
            float(np.clip(metrics["C"], 0.0, 1.0)),


            squash_signed(metrics["snr_db"], center=5.0, scale=8.0),
            squash_signed(metrics["snr_margin_db"], center=0.0, scale=5.0),
            ber_feature,





            ratio01(self.current_param_state.modulation_idx, max(1, len(self.modulations) - 1)),
            ratio01(self.current_param_state.code_idx, max(1, len(self.code_rates) - 1)),
            ratio01(self.current_param_state.spread_idx, max(1, len(self.spreading_factors) - 1)),
            ratio01(self.current_param_state.power_idx, max(1, len(self.power_levels_dbm) - 1)),





            ratio01(self.step_count, max(1, self.cfg.max_steps_per_episode - 1)),
        ], dtype=np.float32)

        return state

    def sample_augmented_scenario(
            self,
            base_scenario: Dict[str, float],
            rng: random.Random,
            distance_jitter_ratio: float,
            frequency_jitter_ratio: float,
            depth_jitter_ratio: float,
            noise_jitter_db: float,
    ) -> Dict[str, float]:







        scenario = copy.deepcopy(base_scenario)



        scenario["distance_m"] = float(np.clip(
            scenario["distance_m"] * (1.0 + rng.uniform(-distance_jitter_ratio, distance_jitter_ratio)),
            10.0,
            self.cfg.max_distance_m,
        ))


        scenario["frequency_hz"] = float(np.clip(
            scenario["frequency_hz"] * (1.0 + rng.uniform(-frequency_jitter_ratio, frequency_jitter_ratio)),
            1.0,
            self.cfg.max_frequency_hz,
        ))


        scenario["source_depth_m"] = float(np.clip(
            scenario["source_depth_m"] * (1.0 + rng.uniform(-depth_jitter_ratio, depth_jitter_ratio)),
            2.0,
            self.cfg.max_depth_m,
        ))


        scenario["receiver_depth_m"] = float(np.clip(
            scenario["receiver_depth_m"] * (1.0 + rng.uniform(-depth_jitter_ratio, depth_jitter_ratio)),
            2.0,
            self.cfg.max_depth_m,
        ))


        scenario["water_depth_m"] = float(np.clip(
            scenario["water_depth_m"] * (1.0 + rng.uniform(-depth_jitter_ratio, depth_jitter_ratio)),
            scenario["receiver_depth_m"] + 50.0,
            self.cfg.max_depth_m,
        ))


        scenario["noise_floor_dbm"] = float(np.clip(
            scenario["noise_floor_dbm"] + rng.uniform(-noise_jitter_db, noise_jitter_db),
            -110.0,
            -85.0,
        ))


        scenario["name"] = scenario["name"] + "_aug"
        return scenario

    def set_scenario(self, scenario: Dict[str, float], recompute_metrics: bool = True) -> None:





        self.current_scenario = copy.deepcopy(scenario)


        self.current_snapshot = get_local_channel_snapshot(
            scenario=self.current_scenario,
            local_window_m=self.cfg.local_window_m,
            local_num_points=self.cfg.local_num_points,
        )


        if recompute_metrics:
            self.current_metrics = self._evaluate_param_state(self.current_param_state)

    def reset(
        self,
        mode_name: Optional[str] = None,
        scenario: Optional[Dict[str, float]] = None,
        fixed_start: bool = True,
        initial_param_state: Optional[ParameterState] = None,
    ):














        if mode_name is None:
            mode_name = self.rng.choice(self.mode_names)

        self.current_mode_name = mode_name
        self.current_mode_weights = COMMUNICATION_MODES[mode_name]


        if scenario is None:
            scenario = copy.deepcopy(self.rng.choice(self.scenarios))


        self.set_scenario(scenario, recompute_metrics=False)


        if initial_param_state is not None:
            self.current_param_state = initial_param_state
        elif fixed_start:
            self.current_param_state = self.reference_param_state
        else:

            self.current_param_state = ParameterState(
                modulation_idx=self.rng.randrange(len(self.modulations)),
                code_idx=self.rng.randrange(len(self.code_rates)),
                spread_idx=self.rng.randrange(len(self.spreading_factors)),
                power_idx=self.rng.randrange(len(self.power_levels_dbm)),
            )


        self.current_metrics = self._evaluate_param_state(self.current_param_state)


        self.step_count = 0
        self.episode_best_c = float(self.current_metrics["C"])
        self.previous_action_id = self.hold_action_id
        self.consecutive_power_up_count = 0

        return self._build_state_vector(), self._info_dict()

    def _info_from_state(self, param_state: ParameterState, metrics: Dict[str, float]) -> Dict[str, float]:









        params = self._parameter_dict(param_state)

        info = {
            "mode_name": self.current_mode_name,
            "scenario_name": self.current_scenario["name"],
            "distance_m": self.current_scenario["distance_m"],
            "frequency_hz": self.current_scenario["frequency_hz"],
            "modulation": params["modulation"],
            "code_rate": params["code_rate"],
            "spread_factor": params["spread_factor"],
            "tx_power_dbm": params["tx_power_dbm"],
            "modulation_idx": param_state.modulation_idx,
            "code_idx": param_state.code_idx,
            "spread_idx": param_state.spread_idx,
            "power_idx": param_state.power_idx,
        }


        info.update(metrics)


        info.update(self.current_snapshot)
        return info

    def _info_dict(self) -> Dict[str, float]:



        return self._info_from_state(self.current_param_state, self.current_metrics)

    def get_valid_action_mask(self, param_state: Optional[ParameterState] = None) -> np.ndarray:








        state = param_state or self.current_param_state
        mask = np.zeros(self.action_dim, dtype=bool)


        mask[self.hold_action_id] = True

        for action_id, action in enumerate(self.actions[1:], start=1):
            next_state = self._apply_action(state, action)
            mask[action_id] = next_state != state

        return mask

    def evaluate_action(self, action_id: int) -> Dict[str, object]:













        action = self.actions[action_id]
        prev_state = self.current_param_state
        prev_metrics = self.current_metrics


        next_state = self._apply_action(prev_state, action)


        next_metrics = self._evaluate_param_state(next_state)


        switch_distance = self._normalized_switch_distance(prev_state, next_state)


        invalid_action = bool(next_state == prev_state and action_id != self.hold_action_id)


        delta_c = float(next_metrics["C"] - prev_metrics["C"])




        switch_penalty = self.cfg.switch_penalty_weight * switch_distance




        power_weight = float(self.current_mode_weights[2])
        constraint_scale = 1.0 + 1.5 * max(0.0, power_weight - 0.5)
        snr_penalty = (
            self.cfg.snr_violation_penalty_weight
            * constraint_scale
            * next_metrics["snr_violation"]
        )
        ber_penalty = (
            self.cfg.ber_violation_penalty_weight
            * constraint_scale
            * next_metrics["ber_violation"]
        )


        stability_bonus = 0.0
        if action_id == self.hold_action_id and next_metrics["C"] >= prev_metrics["C"] - 1e-6:
            stability_bonus = self.cfg.stability_bonus_weight * next_metrics["C"]

        invalid_penalty = self.cfg.invalid_action_penalty if invalid_action else 0.0


        best_progress_bonus = self.cfg.best_progress_bonus_weight * max(
            0.0,
            next_metrics["C"] - self.episode_best_c
        )




        power_up_ratio = max(0.0, next_state.power_idx - prev_state.power_idx) / max(
            1,
            len(self.power_levels_dbm) - 1
        )

        repeated_power_up_count = (
            self.consecutive_power_up_count + 1 if action.power_delta > 0 else 0
        )




        ramp_scale = 0.4 + 0.8 * power_weight
        if power_weight >= 0.60:
            ramp_scale *= 0.65
        power_ramp_penalty = self.cfg.power_ramp_penalty_weight * power_up_ratio * ramp_scale


        repeated_power_penalty = 0.0
        if action.power_delta > 0 and self.consecutive_power_up_count > 0:
            repeated_power_penalty = (
                self.cfg.repeated_power_ramp_penalty_weight
                * repeated_power_up_count
                * ramp_scale
            )




        reward = (
            self.cfg.reward_utility_weight * next_metrics["C"]
            + self.cfg.reward_improvement_weight * delta_c
            - switch_penalty
            - snr_penalty
            - ber_penalty
            - power_ramp_penalty
            - repeated_power_penalty
            + stability_bonus
            + best_progress_bonus
            - invalid_penalty
        )


        info = self._info_from_state(next_state, next_metrics)
        info.update({
            "action_id": int(action_id),
            "action_name": action.name,
            "delta_C": float(delta_c),
            "switch_distance": float(switch_distance),
            "switch_penalty": float(switch_penalty),
            "snr_penalty": float(snr_penalty),
            "ber_penalty": float(ber_penalty),
            "stability_bonus": float(stability_bonus),
            "best_progress_bonus": float(best_progress_bonus),
            "power_ramp_penalty": float(power_ramp_penalty),
            "repeated_power_penalty": float(repeated_power_penalty),
            "invalid_action": bool(invalid_action),
            "invalid_penalty": float(invalid_penalty),
            "reward": float(reward),
        })

        return {
            "action_id": int(action_id),
            "action": action,
            "param_state": next_state,
            "metrics": next_metrics,
            "reward": float(reward),
            "info": info,
        }

    def step(self, action_id: int):









        outcome = self.evaluate_action(action_id)


        self.current_param_state = outcome["param_state"]
        self.current_metrics = outcome["metrics"]
        self.step_count += 1

        reward = float(outcome["reward"])


        self.episode_best_c = max(self.episode_best_c, float(self.current_metrics["C"]))

        action = outcome["action"]


        if action.power_delta > 0:
            self.consecutive_power_up_count += 1
        else:
            self.consecutive_power_up_count = 0

        self.previous_action_id = int(outcome["action_id"])


        done = self.step_count >= self.cfg.max_steps_per_episode


        if done:
            reward += self.cfg.terminal_best_bonus_weight * self.episode_best_c


        info = dict(outcome["info"])
        info["episode_best_C"] = float(self.episode_best_c)
        info["reward"] = float(reward)

        next_state = self._build_state_vector()
        return next_state, reward, done, info

    def enumerate_action_outcomes(self, valid_only: bool = True) -> List[Dict[str, object]]:







        mask = self.get_valid_action_mask()
        outcomes: List[Dict[str, object]] = []

        for action_id in range(self.action_dim):
            if valid_only and not mask[action_id]:
                continue
            outcomes.append(self.evaluate_action(action_id))

        return outcomes

    def greedy_best_action(self) -> Tuple[int, Dict[str, object]]:






        outcomes = self.enumerate_action_outcomes(valid_only=True)
        best = max(outcomes, key=lambda x: x["reward"])
        return int(best["action_id"]), best

    def get_neighbor_states(
        self,
        param_state: Optional[ParameterState] = None,
        step: int = 1
    ) -> List[ParameterState]:







        state = param_state or self.current_param_state

        ranges = [
            len(self.modulations),
            len(self.code_rates),
            len(self.spreading_factors),
            len(self.power_levels_dbm),
        ]
        values = [state.modulation_idx, state.code_idx, state.spread_idx, state.power_idx]

        neighbors = set()

        for dim, upper in enumerate(ranges):
            for delta in (-step, step):
                candidate = list(values)
                candidate[dim] = int(np.clip(candidate[dim] + delta, 0, upper - 1))
                neighbors.add(tuple(candidate))


        neighbors.discard(tuple(values))

        return [ParameterState(*candidate) for candidate in sorted(neighbors)]

    def evaluate_state(self, param_state: ParameterState) -> Dict[str, float]:





        metrics = self._evaluate_param_state(param_state)
        return self._info_from_state(param_state, metrics)

    def parameter_state_from_info(self, info: Dict[str, float]) -> ParameterState:








        return ParameterState(
            modulation_idx=int(info["modulation_idx"]),
            code_idx=int(info["code_idx"]),
            spread_idx=int(info["spread_idx"]),
            power_idx=int(info["power_idx"]),
        )
