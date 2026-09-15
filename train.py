from __future__ import annotations
import copy
import json
import os
import random
from typing import Any, Dict, List, Sequence
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    COMMUNICATION_MODES,
    DEFAULT_SCENARIOS,
    EnvConfig,
    TrainConfig,
    ensure_output_dirs,
)
from dqn_agent import AgentConfig, DQNAgent
from environment import MarineWaveformEnv, ParameterState

try:

    from tqdm import tqdm
except ImportError:
    tqdm = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def _json_default(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)



def english_mode_name(mode_name: str) -> str:
    if "高速传输" in mode_name:
        return "Mode 1: High-Speed Transmission"
    if "高可靠" in mode_name:
        return "Mode 2: High Reliability"
    if "低功耗" in mode_name:
        return "Mode 3: Low Power Consumption"
    if "均衡鲁棒" in mode_name:
        return "Mode 4: Balanced Robustness"
    return mode_name



def english_mode_filename(mode_name: str) -> str:
    safe_name = english_mode_name(mode_name)
    safe_name = safe_name.replace(":", "").replace("/", "_").replace(" ", "_")
    return safe_name


def set_plot_font_sizes() -> None:

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "figure.titlesize": 20,

        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def beautify_axes(ax) -> None:

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", direction="out", width=1.0, length=4, labelsize=17)


def add_panel_label(ax, label: str, fontsize: int = 18) -> None:






    ax.text(
        0.5,
        -0.30,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontfamily="Times New Roman",
        fontweight="bold",
        clip_on=False,
    )


def english_mode_short_name(mode_name: str) -> str:

    if "高速传输" in mode_name:
        return "Mode 1: High-Speed"
    if "高可靠" in mode_name:
        return "Mode 2: High Reliability"
    if "低功耗" in mode_name:
        return "Mode 3: Low Power"
    if "均衡鲁棒" in mode_name:
        return "Mode 4: Balanced Robustness"
    return mode_name



def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    smooth = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, smooth[0])
    return np.concatenate([prefix, smooth])



def exponential_moving_average(values: Sequence[float], alpha: float = 0.30) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for idx in range(1, len(arr)):
        ema[idx] = alpha * arr[idx] + (1.0 - alpha) * ema[idx - 1]
    return ema



def monotone_envelope(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    return np.maximum.accumulate(arr)



def first_convergence_index(
    values: Sequence[float],
    ratio: float = 0.985,
    patience: int = 5,
    std_tol: float = 0.008,
    trend_tol: float = 0.002,
):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return None
    if len(arr) < patience + 1:
        return len(arr)
    smooth = exponential_moving_average(arr, alpha=0.35)
    plateau = float(np.mean(smooth[-patience:]))
    target = ratio * plateau
    for idx in range(0, len(smooth) - patience + 1):
        window = smooth[idx: idx + patience]
        if float(np.mean(window)) < target:
            continue
        if float(np.std(window)) > std_tol:
            continue
        if abs(float(window[-1] - window[0])) > trend_tol:
            continue
        return idx + 1
    return len(arr)



def model_filename(mode_name: str) -> str:
    mode_names = list(COMMUNICATION_MODES.keys())
    return f"dqn_mode_{mode_names.index(mode_name) + 1}.pth"



def make_agent(state_dim: int, action_dim: int, train_cfg: TrainConfig) -> DQNAgent:
    return DQNAgent(
        AgentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=train_cfg.gamma,
            lr=train_cfg.lr,
            target_update_step=train_cfg.target_update_step,
            tau=train_cfg.soft_tau,
            memory_size=train_cfg.memory_size,
            batch_size=train_cfg.batch_size,
        )
    )



def epsilon_by_episode(episode_idx: int, train_cfg: TrainConfig) -> float:
    decay_episodes = max(1, int(train_cfg.episodes_per_mode * train_cfg.epsilon_decay_ratio))
    progress = min(1.0, (episode_idx - 1) / decay_episodes)
    epsilon = train_cfg.epsilon_start + progress * (train_cfg.epsilon_end - train_cfg.epsilon_start)
    return float(max(train_cfg.epsilon_end, epsilon))



def build_scenario_schedule(
    env: MarineWaveformEnv,
    scenarios: Sequence[Dict[str, float]],
    episodes: int,
    seed: int,
    distance_jitter_ratio: float,
    frequency_jitter_ratio: float,
    depth_jitter_ratio: float,
    noise_jitter_db: float,
) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    schedule: List[Dict[str, float]] = []
    while len(schedule) < episodes:
        chunk = list(scenarios)
        rng.shuffle(chunk)
        for item in chunk:
            schedule.append(
                env.sample_augmented_scenario(
                    base_scenario=item,
                    rng=rng,
                    distance_jitter_ratio=distance_jitter_ratio,
                    frequency_jitter_ratio=frequency_jitter_ratio,
                    depth_jitter_ratio=depth_jitter_ratio,
                    noise_jitter_db=noise_jitter_db,
                )
            )
            if len(schedule) >= episodes:
                break
    return schedule[:episodes]



def make_trace_scenario(mode_idx: int, train_cfg: TrainConfig) -> Dict[str, float]:
    base = DEFAULT_SCENARIOS[train_cfg.decision_plot_scenario_index % len(DEFAULT_SCENARIOS)]
    scenario = dict(base)
    scenario["name"] = f"{base['name']}_trace_mode_{mode_idx + 1}"
    return scenario



def make_trace_initial_state(mode_name: str, env: MarineWaveformEnv) -> ParameterState:









    mid_mod = len(env.modulations) // 2
    mid_code = len(env.code_rates) // 2
    mid_spread = len(env.spreading_factors) // 2
    mid_power = len(env.power_levels_dbm) // 2

    low_power = max(0, len(env.power_levels_dbm) // 4)
    high_power = min(len(env.power_levels_dbm) - 1, (3 * len(env.power_levels_dbm)) // 4)

    low_code = 0
    high_code = len(env.code_rates) - 1

    low_spread = 0
    high_spread = len(env.spreading_factors) - 1

    if "高速传输" in mode_name:
        return ParameterState(
            modulation_idx=min(mid_mod + 1, len(env.modulations) - 1),
            code_idx=max(0, high_code - 1),
            spread_idx=low_spread,
            power_idx=high_power,
        )

    if "高可靠" in mode_name:
        return ParameterState(
            modulation_idx=max(0, mid_mod - 1),
            code_idx=low_code,
            spread_idx=high_spread,
            power_idx=high_power,
        )

    if "低功耗" in mode_name:
        return ParameterState(
            modulation_idx=mid_mod,
            code_idx=mid_code,
            spread_idx=mid_spread,
            power_idx=low_power,
        )

    return ParameterState(
        modulation_idx=mid_mod,
        code_idx=mid_code,
        spread_idx=mid_spread,
        power_idx=mid_power,
    )



def run_dqn_episode(
    agent: DQNAgent,
    env: MarineWaveformEnv,
    mode_name: str,
    scenario: Dict[str, float],
    train_cfg: TrainConfig,
    epsilon: float = 0.0,
    training: bool = False,
    fixed_start: bool = True,
    initial_param_state: ParameterState | None = None,
    gamma: float = 1.0,
) -> Dict[str, Any]:
    state, info = env.reset(
        mode_name=mode_name,
        scenario=scenario,
        fixed_start=fixed_start,
        initial_param_state=initial_param_state,
    )
    c_trace: List[float] = []
    reward_trace: List[float] = []
    action_trace: List[int] = []
    info_trace: List[Dict[str, Any]] = []
    td_losses: List[float] = []

    done = False
    while not done:
        action_mask = env.get_valid_action_mask()
        if training:
            action = agent.select_action(state, epsilon, action_mask)
        else:
            action = agent.greedy_action(state, action_mask)

        next_state, reward, done, info = env.step(action)
        next_action_mask = env.get_valid_action_mask()

        if training:
            agent.memory.add(state, next_state, float(reward), int(action), done, next_action_mask)
            if agent.memory.count >= train_cfg.warmup_size:
                for _ in range(train_cfg.updates_per_step):
                    td_loss = agent.learn()
                    if td_loss is not None:
                        td_losses.append(td_loss)

        state = next_state
        c_trace.append(float(info["C"]))
        reward_trace.append(float(reward))
        action_trace.append(int(action))
        info_trace.append(dict(info))

    final_info = dict(info_trace[-1])
    result = {
        "final_info": final_info,
        "episode_avg_C": float(np.mean(c_trace)),
        "episode_final_C": float(c_trace[-1]),
        "episode_best_C": float(np.max(c_trace)),
        "episode_total_reward": float(np.sum(reward_trace)),
        "eval_count": int(len(action_trace)),
        "action_trace": action_trace,
        "c_trace": c_trace,
        "info_trace": info_trace,
        "td_losses": td_losses,
    }
    return result



def make_shared_initial_state(
    env: MarineWaveformEnv,
    rng: random.Random,
    use_fixed_start: bool,
) -> ParameterState:

    if use_fixed_start:
        return env.reference_param_state

    return ParameterState(
        modulation_idx=rng.randrange(len(env.modulations)),
        code_idx=rng.randrange(len(env.code_rates)),
        spread_idx=rng.randrange(len(env.spreading_factors)),
        power_idx=rng.randrange(len(env.power_levels_dbm)),
    )



def evaluate_agent_on_schedule(
    agent: DQNAgent,
    env: MarineWaveformEnv,
    mode_name: str,
    scenario_schedule: Sequence[Dict[str, float]],
    train_cfg: TrainConfig,
) -> Dict[str, float]:
    final_c_values: List[float] = []
    avg_c_values: List[float] = []
    reward_values: List[float] = []
    for scenario in scenario_schedule:
        result = run_dqn_episode(agent, env, mode_name, scenario, train_cfg, epsilon=0.0, training=False, fixed_start=True)
        final_c_values.append(float(result["episode_final_C"]))
        avg_c_values.append(float(result["episode_avg_C"]))
        reward_values.append(float(result["episode_total_reward"]))
    return {
        "avg_final_C": float(np.mean(final_c_values)),
        "best_final_C": float(np.max(final_c_values)),
        "avg_episode_C": float(np.mean(avg_c_values)),
        "avg_total_reward": float(np.mean(reward_values)),
    }



def plot_convergence_curve(
    histories_by_mode: Dict[str, Dict[str, Any]],
    save_path: str,
    smooth_window: int = 20,
) -> None:

    set_plot_font_sizes()

    mode_names = list(histories_by_mode.keys())
    num_modes = len(mode_names)
    if num_modes == 0:
        return

    ncols = 2
    nrows = int(np.ceil(num_modes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.4 * nrows))
    axes = np.array(axes).reshape(-1)

    panel_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]

    for idx_ax, (ax, mode_name) in enumerate(zip(axes, mode_names)):
        history = histories_by_mode[mode_name]

        train_raw = np.asarray(history.get("train_episode_avg_C", []), dtype=float)
        if len(train_raw) > 0:
            x_train = np.arange(1, len(train_raw) + 1)
            train_smooth = moving_average(train_raw, smooth_window)
            ax.plot(
                x_train,
                train_smooth,
                linewidth=2.0,
                alpha=0.75,
                label="DQN Training",
            )

        dev_episodes = np.asarray(history.get("dev_episode", []), dtype=float)
        dqn_dev_raw = np.asarray(history.get("dev_episode_avg_C", []), dtype=float)

        if len(dev_episodes) > 0 and len(dqn_dev_raw) == len(dev_episodes):
            dqn_dev_smooth = exponential_moving_average(dqn_dev_raw, alpha=0.35)
            ax.plot(
                dev_episodes,
                dqn_dev_smooth,
                marker="o",
                markersize=5,
                linewidth=2.5,
                label="DQN Dev",
            )

        add_panel_label(
            ax,
            f"{panel_letters[idx_ax]}) {english_mode_short_name(mode_name)}",
            fontsize=18,
        )

        ax.set_xlabel("Training Episodes", fontsize=20)
        ax.set_ylabel("Communication Performance C", fontsize=20)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.legend(frameon=False, fontsize=15)
        beautify_axes(ax)

    for ax in axes[num_modes:]:
        ax.axis("off")

    fig.tight_layout(h_pad=4.0, w_pad=2.0)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_option_labels(option_values: Sequence[Any], decimals: int = 2) -> List[str]:
    labels: List[str] = []
    for idx, value in enumerate(option_values, start=1):
        if isinstance(value, (float, np.floating)):
            labels.append(f"{idx}({float(value):.{decimals}f})")
        else:
            labels.append(f"{idx}({value})")
    return labels





def _plot_discrete_trace(
    ax,
    x: np.ndarray,
    y_values: Sequence[int],
    option_labels: Sequence[str],
    panel_label: str,
    ylabel: str,
    x_label: str = "Single Decision Step",
) -> None:
    if len(x) != len(y_values):
        raise ValueError("x and y_values must have the same length.")
    if len(option_labels) == 0:
        raise ValueError("option_labels must not be empty.")
    if len(y_values) == 0:
        return

    y = np.asarray(y_values, dtype=int)

    ax.step(
        x,
        y,
        where="post",
        linewidth=2.6,
    )
    ax.plot(
        x,
        y,
        linestyle="None",
        marker="o",
        markersize=5.5,
    )

    tick_positions = np.arange(1, len(option_labels) + 1)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(option_labels, fontsize=17)
    ax.set_ylim(0.5, len(option_labels) + 0.5)

    add_panel_label(ax, panel_label, fontsize=20)
    ax.set_xlabel(x_label, fontsize=21)
    ax.set_ylabel(ylabel, fontsize=21, labelpad=12)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)

    ax.tick_params(axis="x", labelsize=17)
    ax.tick_params(axis="y", labelsize=17)

    beautify_axes(ax)


def build_trace_payload(mode_name: str, trace_result: Dict[str, Any]) -> Dict[str, Any]:
    info_trace = trace_result["info_trace"]
    action_trace = trace_result["action_trace"]

    if len(info_trace) != len(action_trace):
        raise ValueError("info_trace and action_trace must have the same length.")

    trace = {
        "mode_name": mode_name,
        "step": [],
        "modulation_idx": [],
        "code_idx": [],
        "spread_idx": [],
        "power_idx": [],
        "modulation_option": [],
        "code_option": [],
        "spread_option": [],
        "power_option": [],
        "action_id": [],
        "action_name": [],
        "C": [],
        "reward": [],
        "delta_C": [],
    }

    for step_id, (info_item, action_id) in enumerate(zip(info_trace, action_trace), start=1):
        modulation_idx = int(info_item["modulation_idx"])
        code_idx = int(info_item["code_idx"])
        spread_idx = int(info_item["spread_idx"])
        power_idx = int(info_item["power_idx"])

        trace["step"].append(step_id)
        trace["modulation_idx"].append(modulation_idx)
        trace["code_idx"].append(code_idx)
        trace["spread_idx"].append(spread_idx)
        trace["power_idx"].append(power_idx)

        trace["modulation_option"].append(modulation_idx + 1)
        trace["code_option"].append(code_idx + 1)
        trace["spread_option"].append(spread_idx + 1)
        trace["power_option"].append(power_idx + 1)

        trace["action_id"].append(int(action_id) + 1)
        trace["action_name"].append(str(info_item.get("action_name", "")))
        trace["C"].append(float(info_item["C"]))
        trace["reward"].append(float(info_item.get("reward", 0.0)))
        trace["delta_C"].append(float(info_item.get("delta_C", 0.0)))

    return trace



def init_decision_evolution(mode_name: str) -> Dict[str, Any]:
    return {
        "mode_name": mode_name,
        "episode": [],
        "modulation_option": [],
        "code_option": [],
        "spread_option": [],
        "power_option": [],
        "final_C": [],
        "final_ber": [],
        "scenario_name": "",
    }



def append_decision_evolution(history: Dict[str, Any], episode: int, final_info: Dict[str, Any]) -> None:
    required_keys = ["modulation_idx", "code_idx", "spread_idx", "power_idx", "C", "ber"]
    missing = [key for key in required_keys if key not in final_info]
    if missing:
        raise KeyError(f"final_info is missing required keys: {missing}")

    history["episode"].append(int(episode))
    history["modulation_option"].append(int(final_info["modulation_idx"]) + 1)
    history["code_option"].append(int(final_info["code_idx"]) + 1)
    history["spread_option"].append(int(final_info["spread_idx"]) + 1)
    history["power_option"].append(int(final_info["power_idx"]) + 1)
    history["final_C"].append(float(final_info["C"]))
    history["final_ber"].append(float(final_info["ber"]))
    history["scenario_name"] = str(final_info.get("scenario_name", history.get("scenario_name", "")))




def plot_decision_evolution(
    mode_name: str,
    decision_history: Dict[str, Any],
    env: MarineWaveformEnv,
    save_path: str,
) -> None:
    set_plot_font_sizes()

    x = np.asarray(decision_history["episode"], dtype=int)
    if len(x) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.2))
    axes = axes.flatten()

    modulation_labels = _format_option_labels(env.modulations, decimals=0)
    code_labels = _format_option_labels(env.code_rates, decimals=2)
    spread_labels = _format_option_labels(env.spreading_factors, decimals=0)
    power_labels = _format_option_labels(env.power_levels_dbm, decimals=0)

    _plot_discrete_trace(
        axes[0],
        x,
        decision_history["modulation_option"],
        modulation_labels,
        "a) Modulation Scheme",
        "Modulation Scheme",
        x_label="Training Episodes",
    )

    _plot_discrete_trace(
        axes[1],
        x,
        decision_history["code_option"],
        code_labels,
        "b) Code Rate",
        "Code Rate",
        x_label="Training Episodes",
    )

    _plot_discrete_trace(
        axes[2],
        x,
        decision_history["spread_option"],
        spread_labels,
        "c) Spreading Factor",
        "Spreading Factor",
        x_label="Training Episodes",
    )

    _plot_discrete_trace(
        axes[3],
        x,
        decision_history["power_option"],
        power_labels,
        "d) Power Level",
        "Power Level",
        x_label="Training Episodes",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.97,
        hspace=0.72,
        wspace=0.32,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)



def plot_all_modes_final_c_evolution(
    decision_histories_by_mode: Dict[str, Dict[str, Any]],
    save_path: str,
    smooth_alpha: float = 0.30,
) -> None:
    set_plot_font_sizes()

    mode_names = list(decision_histories_by_mode.keys())
    if len(mode_names) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.8))
    axes = np.array(axes).reshape(-1)

    panel_letters = ["a", "b", "c", "d", "e", "f"]

    for idx, (ax, mode_name) in enumerate(zip(axes, mode_names)):
        history = decision_histories_by_mode[mode_name]
        episodes = np.asarray(history.get("episode", []), dtype=int)
        final_c = np.asarray(history.get("final_C", []), dtype=float)

        add_panel_label(
            ax,
            f"{panel_letters[idx]}) {english_mode_short_name(mode_name)}",
            fontsize=20,
        )

        if len(episodes) == 0 or len(final_c) == 0:
            ax.set_xlabel("Training Episodes", fontsize=22)
            ax.set_ylabel("Final Communication Performance", fontsize=22, labelpad=14)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
            ax.tick_params(axis="both", labelsize=18)
            beautify_axes(ax)
            continue

        final_c_smooth = exponential_moving_average(final_c, alpha=smooth_alpha)

        ax.plot(
            episodes,
            final_c_smooth,
            linewidth=2.8,
        )

        ax.set_xlabel("Training Episodes", fontsize=22)
        ax.set_ylabel("Final Communication Performance", fontsize=22, labelpad=14)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)




        ax.tick_params(axis="both", labelsize=18)
        beautify_axes(ax)

    for ax in axes[len(mode_names):]:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.10,
        top=0.97,
        hspace=0.72,
        wspace=0.38,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_all_modes_final_ber_evolution(
    decision_histories_by_mode: Dict[str, Dict[str, Any]],
    save_path: str,
    smooth_alpha: float = 0.30,
) -> None:

    set_plot_font_sizes()

    mode_names = list(decision_histories_by_mode.keys())
    if len(mode_names) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.8))
    axes = np.array(axes).reshape(-1)
    panel_letters = ["a", "b", "c", "d", "e", "f"]

    for idx, (ax, mode_name) in enumerate(zip(axes, mode_names)):
        history = decision_histories_by_mode[mode_name]
        episodes = np.asarray(history.get("episode", []), dtype=int)
        final_ber = np.asarray(history.get("final_ber", []), dtype=float)

        add_panel_label(
            ax,
            f"{panel_letters[idx]}) {english_mode_short_name(mode_name)}",
            fontsize=20,
        )

        valid = np.isfinite(final_ber) & (final_ber > 0.0)
        if len(episodes) > 0 and len(final_ber) == len(episodes) and np.any(valid):
            x = episodes[valid]
            y = final_ber[valid]
            y_smooth = exponential_moving_average(y, alpha=smooth_alpha)
            ax.semilogy(x, y, linewidth=1.0, alpha=0.25, color="#1f77b4")
            ax.semilogy(x, y_smooth, linewidth=2.8, color="#1f77b4", label="DQN Policy BER (EMA)")
            ax.legend(frameon=False, fontsize=15)

        ax.set_xlabel("Training Episodes", fontsize=22)
        ax.set_ylabel("Final BER", fontsize=22, labelpad=14)
        ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=18)
        beautify_axes(ax)

    for ax in axes[len(mode_names):]:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.10,
        top=0.97,
        hspace=0.72,
        wspace=0.38,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)



def plot_all_modes_decision_evolution(
    decision_histories_by_mode: Dict[str, Dict[str, Any]],
    env: MarineWaveformEnv,
    save_path: str,
) -> None:
    set_plot_font_sizes()

    mode_names = list(decision_histories_by_mode.keys())
    if not mode_names:
        return

    fig, axes = plt.subplots(
        len(mode_names),
        4,
        figsize=(24, 5.6 * len(mode_names)),
        squeeze=False,
    )

    modulation_labels = _format_option_labels(env.modulations, decimals=0)
    code_labels = _format_option_labels(env.code_rates, decimals=2)
    spread_labels = _format_option_labels(env.spreading_factors, decimals=0)
    power_labels = _format_option_labels(env.power_levels_dbm, decimals=0)

    metric_names = [
        "Modulation",
        "Code Rate",
        "Spreading Factor",
        "Power Level",
    ]

    panel_idx = 0

    for row_idx, mode_name in enumerate(mode_names):
        decision_history = decision_histories_by_mode[mode_name]
        display_mode = english_mode_short_name(mode_name)
        x = np.asarray(decision_history["episode"], dtype=int)

        y_data_list = [
            decision_history["modulation_option"],
            decision_history["code_option"],
            decision_history["spread_option"],
            decision_history["power_option"],
        ]

        label_list = [
            modulation_labels,
            code_labels,
            spread_labels,
            power_labels,
        ]

        ylabel_list = [
            "Modulation",
            "Code Rate",
            "Spreading Factor",
            "Power Level",
        ]

        for col_idx in range(4):
            letter = chr(ord("a") + panel_idx)
            panel_text = f"{letter}) {display_mode} - {metric_names[col_idx]}"

            _plot_discrete_trace(
                axes[row_idx, col_idx],
                x,
                y_data_list[col_idx],
                label_list[col_idx],
                panel_text,
                ylabel_list[col_idx],
                x_label="Training Episodes",
            )

            panel_idx += 1

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.05,
        top=0.98,
        hspace=0.85,
        wspace=0.42,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def train_single_mode(
    mode_name: str,
    train_cfg: TrainConfig,
    env: MarineWaveformEnv,
    mode_idx: int,
) -> Dict[str, Any]:
    agent = make_agent(env.state_dim, env.action_dim, train_cfg)
    ensure_dir(train_cfg.model_dir)
    model_path = os.path.join(train_cfg.model_dir, model_filename(mode_name))

    train_schedule = build_scenario_schedule(
        env,
        DEFAULT_SCENARIOS,
        train_cfg.episodes_per_mode,
        train_cfg.seed + 1000 + mode_idx * 101,
        train_cfg.train_distance_jitter_ratio,
        train_cfg.train_frequency_jitter_ratio,
        train_cfg.train_depth_jitter_ratio,
        train_cfg.train_noise_jitter_db,
    )
    dev_schedule = build_scenario_schedule(
        env,
        DEFAULT_SCENARIOS,
        train_cfg.evaluation_episodes,
        train_cfg.seed + 2000 + mode_idx * 101,
        train_cfg.eval_distance_jitter_ratio,
        train_cfg.eval_frequency_jitter_ratio,
        train_cfg.eval_depth_jitter_ratio,
        train_cfg.eval_noise_jitter_db,
    )
    test_schedule = build_scenario_schedule(
        env,
        DEFAULT_SCENARIOS,
        train_cfg.evaluation_episodes,
        train_cfg.seed + 3000 + mode_idx * 101,
        train_cfg.eval_distance_jitter_ratio,
        train_cfg.eval_frequency_jitter_ratio,
        train_cfg.eval_depth_jitter_ratio,
        train_cfg.eval_noise_jitter_db,
    )

    trace_scenario = make_trace_scenario(mode_idx, train_cfg)
    trace_initial_state = make_trace_initial_state(mode_name, env)

    decision_trace: Dict[str, Any] = {
        "mode_name": mode_name,
        "step": [],
        "modulation_idx": [],
        "code_idx": [],
        "spread_idx": [],
        "power_idx": [],
        "modulation_option": [],
        "code_option": [],
        "spread_option": [],
        "power_option": [],
        "action_id": [],
        "C": [],
    }
    decision_evolution: Dict[str, Any] = init_decision_evolution(mode_name)

    best_trace_score = -1e9
    best_dev_metric = -1e9
    best_eval_state = copy.deepcopy(agent.eval_net.state_dict())
    best_target_state = copy.deepcopy(agent.target_net.state_dict())
    no_improve_validations = 0

    history: Dict[str, List[float] | List[int]] = {
        "train_episode_final_C": [],
        "train_episode_avg_C": [],
        "dqn_eval_count": [],
        "dqn_td_loss": [],
        "dev_episode": [],
        "dev_episode_final_C": [],
        "dev_episode_avg_C": [],
        "dev_ema_C": [],
        "dev_best_so_far_C": [],
    }
    episode_iter = range(1, train_cfg.episodes_per_mode + 1)
    if tqdm is not None:
        episode_iter = tqdm(episode_iter, desc=f"训练 {mode_name}", ncols=120)

    for episode in episode_iter:
        scenario = train_schedule[episode - 1]
        fixed_start = episode <= int(train_cfg.episodes_per_mode * train_cfg.train_fixed_start_ratio)
        init_rng = random.Random(train_cfg.seed + 50000 + mode_idx * 1000 + episode)
        shared_initial_state = make_shared_initial_state(env, init_rng, fixed_start)

        dqn_result = run_dqn_episode(
            agent,
            env,
            mode_name,
            scenario,
            train_cfg,
            epsilon=epsilon_by_episode(episode, train_cfg),
            training=True,
            fixed_start=fixed_start,
            initial_param_state=shared_initial_state,
        )
        history["train_episode_final_C"].append(float(dqn_result["episode_final_C"]))
        history["train_episode_avg_C"].append(float(dqn_result["episode_avg_C"]))
        history["dqn_eval_count"].append(int(dqn_result["eval_count"]))
        history["dqn_td_loss"].append(float(np.mean(dqn_result["td_losses"])) if dqn_result["td_losses"] else 0.0)

        policy_probe = run_dqn_episode(
            agent,
            env,
            mode_name,
            trace_scenario,
            train_cfg,
            epsilon=0.0,
            training=False,
            fixed_start=True,
            initial_param_state=trace_initial_state,
        )
        append_decision_evolution(decision_evolution, episode, policy_probe["final_info"])

        if episode % train_cfg.validation_interval == 0 or episode == train_cfg.episodes_per_mode:
            history["dev_episode"].append(int(episode))
            dqn_dev = evaluate_agent_on_schedule(agent, env, mode_name, dev_schedule, train_cfg)
            history["dev_episode_final_C"].append(float(dqn_dev["avg_final_C"]))
            history["dev_episode_avg_C"].append(float(dqn_dev["avg_episode_C"]))

            dev_ema = exponential_moving_average(history["dev_episode_avg_C"], alpha=train_cfg.checkpoint_metric_alpha)
            history["dev_ema_C"] = dev_ema.tolist()
            history["dev_best_so_far_C"] = monotone_envelope(dev_ema).tolist()

            current_metric = float(dev_ema[-1])
            if current_metric > best_dev_metric + 1e-6:
                best_dev_metric = current_metric
                best_eval_state = copy.deepcopy(agent.eval_net.state_dict())
                best_target_state = copy.deepcopy(agent.target_net.state_dict())
                no_improve_validations = 0
            else:
                no_improve_validations += 1

            if current_metric >= best_trace_score:
                best_trace_score = current_metric
                trace_result = run_dqn_episode(
                    agent,
                    env,
                    mode_name,
                    trace_scenario,
                    train_cfg,
                    epsilon=0.0,
                    training=False,
                    fixed_start=True,
                    initial_param_state=trace_initial_state,
                )
                decision_trace = build_trace_payload(mode_name, trace_result)

            if episode >= train_cfg.min_episodes_before_early_stop and no_improve_validations >= train_cfg.early_stop_patience:
                break

    agent.eval_net.load_state_dict(best_eval_state)
    agent.target_net.load_state_dict(best_target_state)

    restored_dev = evaluate_agent_on_schedule(agent, env, mode_name, dev_schedule, train_cfg)
    final_test = evaluate_agent_on_schedule(agent, env, mode_name, test_schedule, train_cfg)
    agent.save_model(model_path)

    trace_result = run_dqn_episode(
        agent,
        env,
        mode_name,
        trace_scenario,
        train_cfg,
        epsilon=0.0,
        training=False,
        fixed_start=True,
        initial_param_state=trace_initial_state,
    )
    decision_trace = build_trace_payload(mode_name, trace_result)

    convergence_index = first_convergence_index(
        history["dev_episode_avg_C"],
        ratio=train_cfg.convergence_ratio,
        patience=min(train_cfg.convergence_patience, max(2, len(history["dev_episode_avg_C"]) or 2)),
        std_tol=train_cfg.convergence_std_tol,
        trend_tol=train_cfg.convergence_trend_tol,
    )

    result = {
        "agent": agent,
        "history": history,
        "decision_trace": decision_trace,
        "decision_evolution": decision_evolution,
        "convergence_index": convergence_index,
        "final_dev_episode_avg_C": float(restored_dev["avg_episode_C"]),
        "final_dev_final_C": float(restored_dev["avg_final_C"]),
        "final_test_episode_avg_C": float(final_test["avg_episode_C"]),
        "final_test_final_C": float(final_test["avg_final_C"]),
        "stopped_episode": int(history["dev_episode"][-1] if history["dev_episode"] else 0),
    }
    return result




def main() -> None:
    train_cfg = TrainConfig()

    set_global_seed(train_cfg.seed)
    ensure_output_dirs(train_cfg)
    set_plot_font_sizes()

    env = MarineWaveformEnv(
        env_cfg=EnvConfig(
            max_steps_per_episode=train_cfg.max_steps_per_episode,
            use_power_control=train_cfg.use_power_control,
        )
    )
    env.seed(train_cfg.seed)

    print("=" * 88)
    print("开始训练：海洋低频电磁波形决策系统（DQN）")
    print("=" * 88)

    histories_by_mode: Dict[str, Dict[str, Any]] = {}
    convergence_summary: Dict[str, Any] = {}
    decision_traces_by_mode: Dict[str, Dict[str, Any]] = {}
    decision_evolutions_by_mode: Dict[str, Dict[str, Any]] = {}

    for mode_idx, mode_name in enumerate(COMMUNICATION_MODES):
        print(f"\n>>> 当前训练模式：{mode_name}")
        result = train_single_mode(
            mode_name=mode_name,
            train_cfg=train_cfg,
            env=env,
            mode_idx=mode_idx,
        )

        histories_by_mode[mode_name] = result["history"]
        convergence_summary[mode_name] = {
            "convergence_index": int(result["convergence_index"]),
            "final_dev_episode_avg_C": float(result["final_dev_episode_avg_C"]),
            "final_dev_final_C": float(result["final_dev_final_C"]),
            "final_test_episode_avg_C": float(result["final_test_episode_avg_C"]),
            "final_test_final_C": float(result["final_test_final_C"]),
            "stopped_episode": int(result["stopped_episode"]),
        }

        save_json(result["history"], os.path.join(train_cfg.history_dir, f"mode_{mode_idx + 1}_history.json"))
        decision_traces_by_mode[mode_name] = result["decision_trace"]
        decision_evolutions_by_mode[mode_name] = result["decision_evolution"]

        print(
            f"  DQN Dev平均episode-C: {result['final_dev_episode_avg_C']:.4f} | "
            f"DQN Test平均episode-C: {result['final_test_episode_avg_C']:.4f} | "
            f"收敛轮数: {result['convergence_index']} | "
            f"停止轮数: {result['stopped_episode']}"
        )

    plot_convergence_curve(
        histories_by_mode,
        os.path.join(train_cfg.figure_dir, "Figure1_DQN_Convergence_Across_Four_Modes.png"),
        smooth_window=train_cfg.rolling_window,
    )

    save_json({"decision_traces": decision_traces_by_mode}, os.path.join(train_cfg.history_dir, "decision_traces_all_modes.json"))
    save_json({"decision_evolutions": decision_evolutions_by_mode}, os.path.join(train_cfg.history_dir, "decision_evolutions_all_modes.json"))

    for mode_name, evolution in decision_evolutions_by_mode.items():
        safe_name = english_mode_filename(mode_name)
        if evolution.get("episode"):
            plot_decision_evolution(
                mode_name,
                evolution,
                env,
                os.path.join(train_cfg.figure_dir, f"Figure5_{safe_name}_Decision_Convergence_Over_Training_Episodes.png"),
            )

    if decision_evolutions_by_mode:
        plot_all_modes_decision_evolution(
            decision_evolutions_by_mode,
            env,
            os.path.join(train_cfg.figure_dir, "Figure5_Overview_Decision_Convergence_Over_Training_Episodes_Across_Four_Modes.png"),
        )
        plot_all_modes_final_c_evolution(
            decision_evolutions_by_mode,
            os.path.join(train_cfg.figure_dir, "Figure6_Overview_Final_C_Evolution_Across_Four_Modes.png"),
            smooth_alpha=0.30,
        )
        plot_all_modes_final_ber_evolution(
            decision_evolutions_by_mode,
            os.path.join(train_cfg.figure_dir, "Figure7_Overview_Final_BER_Evolution_Across_Four_Modes.png"),
            smooth_alpha=0.30,
        )

    save_json(convergence_summary, os.path.join(train_cfg.history_dir, "convergence_summary.json"))

    print("\n训练完成。")
    print("输出目录:", os.path.abspath(train_cfg.output_dir))


if __name__ == "__main__":
    main()

