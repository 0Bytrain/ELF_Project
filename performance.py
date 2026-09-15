from __future__ import annotations
from math import log10
from typing import Dict, Tuple
import numpy as np
from config import CODE_RATES, MODULATION_SCHEMES, POWER_LEVELS_DBM, SPREADING_FACTORS


CHANNEL_STD_PENALTY_FACTOR = 0.35
CHANNEL_SLOPE_PENALTY_FACTOR = 0.08
DEFAULT_CHANNEL_WINDOW_M = 80.0






MODULATION_PROFILES: Dict[str, Dict[str, float]] = {
    "BPSK": {
        "bits_per_symbol": 1.0,
        "snr_offset_db": 0.0,
        "power_efficiency": 0.92,
        "required_snr_base_db": 5.0,                            
    },
    "QPSK": {
        "bits_per_symbol": 2.0,
        "snr_offset_db": -0.5,
        "power_efficiency": 0.86,
        "required_snr_base_db": 7.0,
    },
    "MSK": {
        "bits_per_symbol": 1.0,             
        "snr_offset_db": 0.3,
        "power_efficiency": 1.00,
        "required_snr_base_db": 5.0,
    }
}




CODING_GAIN_DB = {
    0.25: 4.5,
    0.50: 3.0,
    0.75: 1.5,
    round(8.0 / 9.0, 6): 0.5,
}



CODE_THRESHOLD_SHIFT_DB = {
    0.25: -3.0,
    0.50: -1.8,
    0.75: 0.8,
    round(8.0 / 9.0, 6): 0.0,
}



SPREAD_THRESHOLD_SHIFT_DB = {
    2: 0.4,
    4: 0.1,
    8: -0.2,
    32: -0.5,
}


def _safe_clip01(x: float) -> float:




    return float(np.clip(x, 0.0, 1.0))


def _sigmoid(x: float) -> float:





    x = float(np.clip(x, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _coding_gain_db(code_rate: float) -> float:




    for key, value in CODING_GAIN_DB.items():
        if abs(code_rate - key) < 1e-6:
            return value
    return 0.0


def _code_threshold_shift_db(code_rate: float) -> float:




    for key, value in CODE_THRESHOLD_SHIFT_DB.items():
        if abs(code_rate - key) < 1e-6:
            return value
    return 0.0


def compute_effective_snr_db(
    snapshot: Dict[str, float],
    modulation: str,
    code_rate: float,
    spread_factor: int,
    tx_power_dbm: float,
    system_margin_db: float,
    channel_window_m: float = DEFAULT_CHANNEL_WINDOW_M,
) -> float:

























    channel_shape_penalty = (
        CHANNEL_STD_PENALTY_FACTOR * max(snapshot["path_loss_std_db"], 0.0)
        + CHANNEL_SLOPE_PENALTY_FACTOR
        * abs(snapshot["slope_db_per_km"])
        * max(float(channel_window_m), 0.0)
        / 1000.0
    )

    rx_power_dbm = tx_power_dbm - snapshot["path_loss_db"]

    rx_snr_db = (
        rx_power_dbm
        - snapshot["noise_floor_dbm"]
        - system_margin_db
        - channel_shape_penalty
    )

    return float(rx_snr_db)


def compute_required_snr_db(modulation: str, code_rate: float, spread_factor: int) -> float:











    return float(
        MODULATION_PROFILES[modulation]["required_snr_base_db"]
        + _code_threshold_shift_db(code_rate)
        + SPREAD_THRESHOLD_SHIFT_DB[spread_factor]
    )


def compute_ber(
    snapshot: Dict[str, float],
    modulation: str,
    snr_margin_db: float,
    target_ber: float = 1e-5,
    ber_floor: float = 1e-9,
) -> float:

































    dominant_echo_db = max(snapshot["lateral_db"], snapshot["reflected_db"])
    multipath_excess_db = max(0.0, dominant_echo_db - snapshot["direct_db"])





    multipath_penalty_db = 0.60 * multipath_excess_db



    modulation_penalty_db = {
        "BPSK": 0.00,
        "MSK": 0.10,
        "QPSK": 0.25,
    }.get(modulation, 0.0)


    effective_margin_db = (
        snr_margin_db
        - multipath_penalty_db
        - modulation_penalty_db
    )





    log10_ber = np.log10(max(target_ber, 1e-12)) - 0.85 * effective_margin_db
    ber = 10 ** log10_ber

    return float(np.clip(ber, ber_floor, 0.5))

def compute_feasibility_metric(snr_margin_db: float, ber: float, target_ber: float) -> float:














    snr_gate = _sigmoid((snr_margin_db + 1.0) / 1.8)

    ber_log_ratio = np.log10(max(target_ber, 1e-12) / max(ber, 1e-12))

    ber_gate = _sigmoid((ber_log_ratio + 0.15) / 0.65)

    return _safe_clip01(0.55 * snr_gate + 0.45 * ber_gate)


def compute_rate_metric(modulation: str, code_rate: float, spread_factor: int, ber: float, snr_margin_db: float) -> float:










    raw_rate = MODULATION_PROFILES[modulation]["bits_per_symbol"] * code_rate / spread_factor





    success_factor = _sigmoid((snr_margin_db + 1.2) / 1.8) * np.exp(-12.0 * ber)

    effective_rate = raw_rate * success_factor

    max_raw_rate = (
        max(MODULATION_PROFILES[m]["bits_per_symbol"] for m in MODULATION_SCHEMES)
        * max(CODE_RATES)
        / min(SPREADING_FACTORS)
    )

    return _safe_clip01(effective_rate / max_raw_rate)


def compute_reliability_metric(ber: float, snr_margin_db: float, target_ber: float) -> float:













    ber_score = 1.0 / (1.0 + (max(ber, 1e-12) / target_ber) ** 0.50)


    margin_score = _sigmoid((snr_margin_db + 0.8) / 1.8)

    return _safe_clip01(0.65 * ber_score + 0.35 * margin_score)


def compute_power_metric(modulation: str, code_rate: float, spread_factor: int, tx_power_dbm: float) -> float:











    profile = MODULATION_PROFILES[modulation]


    tx_power_mw = 10 ** (tx_power_dbm / 10.0)





    energy_like = tx_power_mw * np.sqrt(spread_factor) / max(
        profile["bits_per_symbol"] * code_rate * profile["power_efficiency"], 1e-6
    )


    min_energy_like = (
        (10 ** (min(POWER_LEVELS_DBM) / 10.0))
        * np.sqrt(min(SPREADING_FACTORS))
        / max(
            MODULATION_PROFILES[m]["bits_per_symbol"] * max(CODE_RATES) * MODULATION_PROFILES[m]["power_efficiency"]
            for m in MODULATION_SCHEMES
        )
    )


    max_energy_like = (
        (10 ** (max(POWER_LEVELS_DBM) / 10.0))
        * np.sqrt(max(SPREADING_FACTORS))
        / min(
            MODULATION_PROFILES[m]["bits_per_symbol"] * min(CODE_RATES) * MODULATION_PROFILES[m]["power_efficiency"]
            for m in MODULATION_SCHEMES
        )
    )


    numerator = np.log10(max(energy_like, 1e-12)) - np.log10(min_energy_like)
    denominator = np.log10(max_energy_like) - np.log10(min_energy_like)


    metric = 1.0 - numerator / max(denominator, 1e-9)


    if modulation == "MSK":
        metric += 0.015

    return _safe_clip01(metric)


def compute_strategy_bonus(
    weights: Tuple[float, float, float],
    code_rate: float,
    spread_factor: int,
    tx_power_dbm: float,
) -> float:













    w_rate, w_ber, w_power = weights


    code_norm = (code_rate - min(CODE_RATES)) / max(max(CODE_RATES) - min(CODE_RATES), 1e-9)
    spread_norm = (np.log2(spread_factor) - np.log2(min(SPREADING_FACTORS))) / max(
        np.log2(max(SPREADING_FACTORS)) - np.log2(min(SPREADING_FACTORS)), 1e-9
    )
    power_norm = (tx_power_dbm - min(POWER_LEVELS_DBM)) / max(
        max(POWER_LEVELS_DBM) - min(POWER_LEVELS_DBM), 1e-9
    )




    rate_pref = 0.55 * code_norm + 0.45 * (1.0 - spread_norm)




    reliability_pref = 0.60 * spread_norm + 0.40 * (1.0 - code_norm)


    power_pref = 1.0 - power_norm


    compatibility = w_rate * rate_pref + w_ber * reliability_pref + w_power * power_pref


    return float(np.clip(0.015 * (compatibility - 0.5), -0.008, 0.008))


def compute_communication_performance(
    snapshot: Dict[str, float],
    weights: Tuple[float, float, float],
    modulation: str,
    code_rate: float,
    spread_factor: int,
    tx_power_dbm: float,
    system_margin_db: float = 50.0,
    ber_floor: float = 1e-9,
    target_ber: float = 1e-5,
    channel_window_m: float = DEFAULT_CHANNEL_WINDOW_M,
) -> Dict[str, float]:














    w_rate, w_ber, w_power = weights













    channel_shape_penalty = (
        CHANNEL_STD_PENALTY_FACTOR * max(snapshot["path_loss_std_db"], 0.0)
        + CHANNEL_SLOPE_PENALTY_FACTOR
        * abs(snapshot["slope_db_per_km"])
        * max(float(channel_window_m), 0.0)
        / 1000.0
    )

    rx_power_dbm = float(tx_power_dbm - snapshot["path_loss_db"])

    snr_db = float(
        rx_power_dbm
        - snapshot["noise_floor_dbm"]
        - system_margin_db
        - channel_shape_penalty
    )





    required_snr_db = compute_required_snr_db(
        modulation=modulation,
        code_rate=code_rate,
        spread_factor=spread_factor
    )




    snr_margin_db = float(snr_db - required_snr_db)













    dominant_echo_db = max(snapshot["lateral_db"], snapshot["reflected_db"])
    multipath_excess_db = max(0.0, dominant_echo_db - snapshot["direct_db"])

    multipath_penalty_db = 0.60 * multipath_excess_db

    modulation_penalty_db = {
        "BPSK": 0.00,
        "MSK": 0.10,
        "QPSK": 0.25,
    }.get(modulation, 0.0)

    effective_margin_db = float(
        snr_margin_db
        - multipath_penalty_db
        - modulation_penalty_db
    )



    log10_ber = np.log10(max(target_ber, 1e-12)) - 0.85 * effective_margin_db
    ber = float(np.clip(10 ** log10_ber, ber_floor, 0.5))









    f_rate = compute_rate_metric(modulation, code_rate, spread_factor, ber, snr_margin_db)

    f_reliability = compute_reliability_metric(
        ber=ber,
        snr_margin_db=snr_margin_db,
        target_ber=target_ber
    )

    f_power = compute_power_metric(
        modulation=modulation,
        code_rate=code_rate,
        spread_factor=spread_factor,
        tx_power_dbm=tx_power_dbm
    )




    strategy_bonus = compute_strategy_bonus(weights, code_rate, spread_factor, tx_power_dbm)





    feasibility = compute_feasibility_metric(
        snr_margin_db=snr_margin_db,
        ber=ber,
        target_ber=target_ber
    )





    snr_violation = float(max(0.0, -snr_margin_db) / 10.0)
    ber_violation = (
        float(max(0.0, log10(max(ber, 1e-12) / target_ber)) / 5.0)
        if ber > target_ber else 0.0
    )




    utility_core = float(
        w_rate * f_rate
        + w_ber * f_reliability
        + w_power * f_power
        + strategy_bonus
    )









    c_value = float(
        utility_core * (0.25 + 0.75 * feasibility)
        - 0.08 * snr_violation
        - 0.08 * ber_violation
    )


    c_value = _safe_clip01(c_value)




    return {
        "rx_power_dbm": float(rx_power_dbm),
        "path_loss_db": float(snapshot["path_loss_db"]),
        "snr_db": float(snr_db),
        "required_snr_db": float(required_snr_db),
        "snr_margin_db": float(snr_margin_db),
        "snr_violation": float(snr_violation),
        "ber": float(ber),
        "ber_violation": float(ber_violation),
        "feasibility": float(feasibility),
        "f_rate": float(f_rate),
        "f_ber": float(f_reliability),
        "f_reliability": float(f_reliability),
        "f_power": float(f_power),
        "strategy_bonus": float(strategy_bonus),
        "C": float(c_value),
    }
