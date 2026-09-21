












from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np

from config import (
    REFERENCE_DISTANCE_M,
    REFERENCE_PATH_LOSS_DB,
)







EPS0 = 8.854_187_812_8e-12  # 真空介电常数 ε0 (F/m)
MU0 = 1.256_637_062_12e-6   # 真空磁导率 μ0 (H/m)


DB_PER_NEPER = 20.0 / np.log(10.0)  # ≈ 8.6859  ← 注意这里是 np.log


AIR_PARAMS = {
    "name": "空气",
    "eps_r": 1.0,
    "mu_r": 1.0,
    "sigma_s_m": 0.0,
}


ICE_PARAMS = {
    "name": "一年冰",
    "eps_r": 5.0,
    "mu_r": 1.0,
    "sigma_s_m": 0.03,
}


SEA_PARAMS = {
    "name": "极地海水",
    "eps_r": 81.0,
    "mu_r": 1.0,
    "sigma_s_m": 3.0,
}


SCENARIO_PARAMS = {
    "frequency_hz": 30.0,
    "ice_thickness_m": 2.0,
    "incidence_deg": 0.0,
    "polarization": "TE",
    "incident_field_v_m": 1.0,
    "min_water_depth_m": 0.0,
    "max_water_depth_m": 200.0,
    "num_points": 201,
}


PLOT_PARAMS = {
    "output_filename": "polar_three_layer_attenuation.png",
    "dpi": 180,
    "show_figure": False,
}



DEFAULT_ICE_THICKNESS_M = 2.0
DEFAULT_POLARIZATION = "TE"



UNRESOLVED_MULTIPATH_GAP_DB = 120.0






def set_chinese_font() -> None:


    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimSun",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    available = [name for name in candidate_fonts if name in installed]
    plt.rcParams["font.sans-serif"] = available + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False






def _decaying_sqrt(value: complex) -> complex:


    root = complex(np.sqrt(complex(value)))
    tolerance = 1e-14 * max(1.0, abs(root))
    if root.real < -tolerance or (
        abs(root.real) <= tolerance and root.imag < 0.0
    ):
        root = -root
    return root


def calculate_wave_parameters(
    frequency_hz: float,
    eps_r: float,
    mu_r: float,
    sigma_s_m: float,
) -> Dict[str, complex | float]:


    if frequency_hz <= 0.0:
        raise ValueError("频率必须大于 0 Hz。")
    if eps_r <= 0.0 or mu_r <= 0.0 or sigma_s_m < 0.0:
        raise ValueError("介质参数无效。")

    omega = 2.0 * np.pi * frequency_hz
    epsilon = EPS0 * eps_r
    mu = MU0 * mu_r


    admittance = sigma_s_m + 1j * omega * epsilon


    gamma = _decaying_sqrt(1j * omega * mu * admittance)
    impedance = _decaying_sqrt(1j * omega * mu / admittance)
    if impedance.real < 0.0:
        impedance = -impedance

    alpha = float(gamma.real)
    return {
        "omega": omega,
        "epsilon": epsilon,
        "mu": mu,
        "admittance": admittance,
        "gamma": gamma,
        "alpha": alpha,
        "beta": float(gamma.imag),
        "impedance": impedance,
        "skin_depth": np.inf if alpha <= 1e-15 else 1.0 / alpha,
    }


def _normal_gamma(gamma: complex, transverse_wavenumber: float) -> complex:


    return _decaying_sqrt(gamma * gamma + transverse_wavenumber**2)


def _wave_impedance(
    wave_params: Dict[str, complex | float],
    normal_gamma: complex,
    polarization: str,
) -> complex:


    omega = float(wave_params["omega"])
    mu = float(wave_params["mu"])
    admittance = complex(wave_params["admittance"])
    if polarization == "TE":
        return 1j * omega * mu / normal_gamma
    return normal_gamma / admittance


def amplitude_to_db(amplitude: np.ndarray | complex | float) -> np.ndarray:


    magnitude = np.maximum(np.abs(amplitude), np.finfo(float).tiny)
    return 20.0 * np.log10(magnitude)






def calculate_ice_layer_component(
    frequency_hz: float,
    ice_thickness_m: float,
    incidence_deg: float,
    polarization: str,
    incident_field_v_m: complex,
    air_params: Dict[str, float | str],
    ice_params: Dict[str, float | str],
    sea_params: Dict[str, float | str],
) -> Dict[str, complex | float]:


    if ice_thickness_m < 0.0:
        raise ValueError("海冰厚度不能为负数。")
    if not 0.0 <= incidence_deg < 89.9:
        raise ValueError("入射角必须位于 [0, 89.9) 度。")
    polarization = polarization.upper()
    if polarization not in {"TE", "TM"}:
        raise ValueError("极化方式必须是 TE 或 TM。")


    air_wave = calculate_wave_parameters(
        frequency_hz,
        float(air_params["eps_r"]),
        float(air_params["mu_r"]),
        float(air_params["sigma_s_m"]),
    )
    ice_wave = calculate_wave_parameters(
        frequency_hz,
        float(ice_params["eps_r"]),
        float(ice_params["mu_r"]),
        float(ice_params["sigma_s_m"]),
    )
    sea_wave = calculate_wave_parameters(
        frequency_hz,
        float(sea_params["eps_r"]),
        float(sea_params["mu_r"]),
        float(sea_params["sigma_s_m"]),
    )


    theta0 = np.deg2rad(incidence_deg)
    transverse_wavenumber = abs(float(air_wave["beta"])) * np.sin(theta0)
    gamma_air_z = _normal_gamma(
        complex(air_wave["gamma"]), transverse_wavenumber
    )
    gamma_ice_z = _normal_gamma(
        complex(ice_wave["gamma"]), transverse_wavenumber
    )
    gamma_sea_z = _normal_gamma(
        complex(sea_wave["gamma"]), transverse_wavenumber
    )

    z_air = _wave_impedance(air_wave, gamma_air_z, polarization)
    z_ice = _wave_impedance(ice_wave, gamma_ice_z, polarization)
    z_sea = _wave_impedance(sea_wave, gamma_sea_z, polarization)



    electrical_thickness = gamma_ice_z * ice_thickness_m
    matrix_a = np.cosh(electrical_thickness)
    matrix_b = z_ice * np.sinh(electrical_thickness)
    matrix_c = np.sinh(electrical_thickness) / z_ice
    matrix_d = matrix_a


    denominator = (
        matrix_a
        + matrix_b / z_sea
        + z_air * (matrix_c + matrix_d / z_sea)
    )
    transmission = complex(2.0 / denominator)
    reflection = complex(
        (
            matrix_a
            + matrix_b / z_sea
            - z_air * (matrix_c + matrix_d / z_sea)
        )
        / denominator
    )

    field_at_ice_bottom = complex(incident_field_v_m) * transmission


    incident_admittance = max(float(np.real(1.0 / z_air)), 0.0)
    sea_admittance = max(float(np.real(1.0 / z_sea)), 0.0)
    power_fraction_at_ice_bottom = (
        0.0
        if incident_admittance <= 0.0
        else sea_admittance
        / incident_admittance
        * abs(transmission) ** 2
    )

    slab_field_loss_db = -20.0 * np.log10(
        max(abs(transmission), np.finfo(float).tiny)
    )
    ice_bulk_field_loss_db = (
        DB_PER_NEPER * gamma_ice_z.real * ice_thickness_m
    )

    return {
        "reflection_coefficient": reflection,
        "transmission_coefficient": transmission,
        "reflected_field_v_m": complex(incident_field_v_m) * reflection,
        "field_at_ice_bottom_v_m": field_at_ice_bottom,
        "water_normal_gamma_per_m": gamma_sea_z,
        "water_wave_impedance_ohm": z_sea,
        "reflected_power_fraction": abs(reflection) ** 2,
        "power_fraction_at_ice_bottom": power_fraction_at_ice_bottom,
        "slab_field_loss_db": float(slab_field_loss_db),
        "ice_bulk_field_loss_db": float(ice_bulk_field_loss_db),
        "boundary_loss_db": float(
            slab_field_loss_db - ice_bulk_field_loss_db
        ),
        "air_wave_params": air_wave,
        "ice_wave_params": ice_wave,
        "sea_wave_params": sea_wave,
    }






def calculate_water_propagation_component(
    field_at_water_surface_v_m: complex,
    water_depth_values: np.ndarray,
    water_normal_gamma_per_m: complex,
    water_wave_impedance_ohm: complex,
    sea_mu_r: float,
) -> Dict[str, np.ndarray]:


    propagation_factor = np.exp(
        -water_normal_gamma_per_m * water_depth_values
    )
    receiver_field = field_at_water_surface_v_m * propagation_factor
    receiver_magnetic_field = receiver_field / water_wave_impedance_ohm
    receiver_magnetic_flux = MU0 * sea_mu_r * receiver_magnetic_field
    water_bulk_loss_db = (
        DB_PER_NEPER
        * water_normal_gamma_per_m.real
        * water_depth_values
    )
    return {
        "propagation_factor": propagation_factor,
        "receiver_field_v_m": receiver_field,
        "receiver_magnetic_field_a_m": receiver_magnetic_field,
        "receiver_magnetic_flux_t": receiver_magnetic_flux,
        "water_bulk_field_loss_db": water_bulk_loss_db,
    }






def compute_field_components(
    water_depth_values: np.ndarray,
    frequency_hz: float,
    ice_thickness_m: float,
    incidence_deg: float,
    polarization: str,
    incident_field_v_m: complex,
    air_params: Dict[str, float | str] = AIR_PARAMS,
    ice_params: Dict[str, float | str] = ICE_PARAMS,
    sea_params: Dict[str, float | str] = SEA_PARAMS,
) -> Dict[str, np.ndarray | Dict[str, complex | float]]:


    water_depth_values = np.asarray(water_depth_values, dtype=float)
    if water_depth_values.ndim != 1 or len(water_depth_values) == 0:
        raise ValueError("water_depth_values 必须是一维非空数组。")
    if np.any(water_depth_values < 0.0):
        raise ValueError("水下接收深度不能为负数。")
    if abs(incident_field_v_m) == 0.0:
        raise ValueError("入射电场不能为零。")


    ice_component = calculate_ice_layer_component(
        frequency_hz,
        ice_thickness_m,
        incidence_deg,
        polarization,
        incident_field_v_m,
        air_params,
        ice_params,
        sea_params,
    )


    water_component = calculate_water_propagation_component(
        complex(ice_component["field_at_ice_bottom_v_m"]),
        water_depth_values,
        complex(ice_component["water_normal_gamma_per_m"]),
        complex(ice_component["water_wave_impedance_ohm"]),
        float(sea_params["mu_r"]),
    )

    receiver_field = np.asarray(
        water_component["receiver_field_v_m"], dtype=complex
    )
    propagation_factor = np.asarray(
        water_component["propagation_factor"], dtype=complex
    )
    incident_magnitude = abs(incident_field_v_m)


    total_field_loss_db = -amplitude_to_db(
        receiver_field / incident_magnitude
    )


    total_power_fraction = float(
        ice_component["power_fraction_at_ice_bottom"]
    ) * abs(propagation_factor) ** 2
    total_power_loss_db = -10.0 * np.log10(
        np.maximum(total_power_fraction, np.finfo(float).tiny)
    )

    num_points = len(water_depth_values)
    return {
        "receiver_field_v_m": receiver_field,
        "receiver_field_db_v_m": amplitude_to_db(receiver_field),
        "receiver_magnetic_field_a_m": np.asarray(
            water_component["receiver_magnetic_field_a_m"], dtype=complex
        ),
        "receiver_magnetic_flux_t": np.asarray(
            water_component["receiver_magnetic_flux_t"], dtype=complex
        ),
        "total_field_loss_db": total_field_loss_db,
        "total_power_loss_db": total_power_loss_db,
        "ice_layer_field_loss_db": np.full(
            num_points, float(ice_component["slab_field_loss_db"])
        ),
        "ice_bulk_field_loss_db": np.full(
            num_points, float(ice_component["ice_bulk_field_loss_db"])
        ),
        "boundary_loss_db": np.full(
            num_points, float(ice_component["boundary_loss_db"])
        ),
        "water_bulk_field_loss_db": np.asarray(
            water_component["water_bulk_field_loss_db"], dtype=float
        ),
        "ice_component": ice_component,
    }






def _trace_air_ice_water_ray(
    horizontal_distance_m: float,
    transmitter_height_m: float,
    ice_thickness_m: float,
    receiver_depth_m: float,
    frequency_hz: float,
    air_params: Dict[str, float | str],
    ice_params: Dict[str, float | str],
    sea_params: Dict[str, float | str],
) -> Dict[str, float]:


    if horizontal_distance_m <= 0.0:
        raise ValueError("水平通信距离必须大于 0 m。")
    if transmitter_height_m <= 0.0:
        raise ValueError("空气发射端高度必须大于 0 m。")
    if ice_thickness_m < 0.0 or receiver_depth_m < 0.0:
        raise ValueError("冰层厚度和水下接收深度不能为负数。")

    air_wave = calculate_wave_parameters(
        frequency_hz,
        float(air_params["eps_r"]),
        float(air_params["mu_r"]),
        float(air_params["sigma_s_m"]),
    )
    ice_wave = calculate_wave_parameters(
        frequency_hz,
        float(ice_params["eps_r"]),
        float(ice_params["mu_r"]),
        float(ice_params["sigma_s_m"]),
    )
    sea_wave = calculate_wave_parameters(
        frequency_hz,
        float(sea_params["eps_r"]),
        float(sea_params["mu_r"]),
        float(sea_params["sigma_s_m"]),
    )

    beta_air = abs(float(air_wave["beta"]))
    beta_ice = abs(float(ice_wave["beta"]))
    beta_sea = abs(float(sea_wave["beta"]))
    if min(beta_air, beta_ice, beta_sea) <= 1e-15:
        raise ValueError("介质相位常数异常，无法计算分层折射路径。")

    def segment_angles(theta_air: float) -> Tuple[float, float]:
        transverse_wavenumber = beta_air * np.sin(theta_air)
        theta_ice = np.arcsin(np.clip(transverse_wavenumber / beta_ice, 0.0, 1.0))
        theta_sea = np.arcsin(np.clip(transverse_wavenumber / beta_sea, 0.0, 1.0))
        return float(theta_ice), float(theta_sea)

    def horizontal_offset(theta_air: float) -> float:
        theta_ice, theta_sea = segment_angles(theta_air)
        return float(
            transmitter_height_m * np.tan(theta_air)
            + ice_thickness_m * np.tan(theta_ice)
            + receiver_depth_m * np.tan(theta_sea)
        )


    lower_angle = 0.0
    upper_angle = np.deg2rad(89.8)
    if horizontal_offset(upper_angle) < horizontal_distance_m:
        raise ValueError("给定高度和距离无法形成有效的空气-冰层-海水传播路径。")

    for _ in range(80):
        middle_angle = 0.5 * (lower_angle + upper_angle)
        if horizontal_offset(middle_angle) < horizontal_distance_m:
            lower_angle = middle_angle
        else:
            upper_angle = middle_angle

    theta_air = 0.5 * (lower_angle + upper_angle)
    theta_ice, theta_sea = segment_angles(theta_air)
    air_path_m = transmitter_height_m / np.cos(theta_air)
    ice_path_m = ice_thickness_m / np.cos(theta_ice)
    water_path_m = receiver_depth_m / np.cos(theta_sea)

    return {
        "incidence_deg": float(np.rad2deg(theta_air)),
        "air_path_m": float(air_path_m),
        "ice_path_m": float(ice_path_m),
        "water_path_m": float(water_path_m),
        "total_path_m": float(air_path_m + ice_path_m + water_path_m),
    }


def _evaluate_direct_path(
    horizontal_distance_m: float,
    transmitter_height_m: float,
    receiver_depth_m: float,
    ice_thickness_m: float,
    frequency_hz: float,
    polarization: str,
) -> Tuple[float, float]:


    ray = _trace_air_ice_water_ray(
        horizontal_distance_m=horizontal_distance_m,
        transmitter_height_m=transmitter_height_m,
        ice_thickness_m=ice_thickness_m,
        receiver_depth_m=receiver_depth_m,
        frequency_hz=frequency_hz,
        air_params=AIR_PARAMS,
        ice_params=ICE_PARAMS,
        sea_params=SEA_PARAMS,
    )
    components = compute_field_components(
        water_depth_values=np.array([receiver_depth_m], dtype=float),
        frequency_hz=frequency_hz,
        ice_thickness_m=ice_thickness_m,
        incidence_deg=ray["incidence_deg"],
        polarization=polarization,
        incident_field_v_m=1.0,
    )
    receiver_field = complex(components["receiver_field_v_m"][0])




    normalized_amplitude = abs(receiver_field) / max(ray["total_path_m"], 1e-9)
    field_db = float(amplitude_to_db(normalized_amplitude))
    return float(normalized_amplitude), field_db


def get_local_channel_snapshot(
    scenario: Dict[str, float],
    local_window_m: float = 1200.0,
    local_num_points: int = 7,
) -> Dict[str, float]:







    frequency_hz = float(scenario["frequency_hz"])
    distance_m = float(scenario["distance_m"])
    transmitter_height_m = float(scenario["source_depth_m"])
    receiver_depth_m = float(scenario["receiver_depth_m"])
    water_depth_m = float(scenario["water_depth_m"])
    noise_floor_dbm = float(scenario["noise_floor_dbm"])
    ice_thickness_m = float(scenario.get("ice_thickness_m", DEFAULT_ICE_THICKNESS_M))
    polarization = str(scenario.get("polarization", DEFAULT_POLARIZATION)).upper()

    if receiver_depth_m > water_depth_m:
        raise ValueError("水下接收深度不能大于水深。")
    if local_window_m <= 0.0 or local_num_points < 2:
        raise ValueError("局部窗口必须为正，且采样点数至少为 2。")


    offsets = np.linspace(-0.5 * local_window_m, 0.5 * local_window_m, int(local_num_points))
    rho_values = np.maximum(1.0, distance_m + offsets)

    reference_amplitude, _ = _evaluate_direct_path(
        horizontal_distance_m=REFERENCE_DISTANCE_M,
        transmitter_height_m=transmitter_height_m,
        receiver_depth_m=receiver_depth_m,
        ice_thickness_m=ice_thickness_m,
        frequency_hz=frequency_hz,
        polarization=polarization,
    )

    path_loss_curve_db = np.empty(len(rho_values), dtype=float)
    direct_curve_db = np.empty(len(rho_values), dtype=float)
    for index, rho in enumerate(rho_values):
        amplitude, direct_db = _evaluate_direct_path(
            horizontal_distance_m=float(rho),
            transmitter_height_m=transmitter_height_m,
            receiver_depth_m=receiver_depth_m,
            ice_thickness_m=ice_thickness_m,
            frequency_hz=frequency_hz,
            polarization=polarization,
        )
        incremental_loss_db = 20.0 * np.log10(
            max(reference_amplitude, np.finfo(float).tiny)
            / max(amplitude, np.finfo(float).tiny)
        )
        path_loss_curve_db[index] = REFERENCE_PATH_LOSS_DB + incremental_loss_db
        direct_curve_db[index] = direct_db

    center_index = len(rho_values) // 2
    unique_rho, unique_indices = np.unique(rho_values, return_index=True)
    if len(unique_rho) >= 2:
        slope_db_per_km = float(np.polyfit(
            unique_rho / 1000.0,
            path_loss_curve_db[unique_indices],
            1,
        )[0])
    else:
        slope_db_per_km = 0.0

    direct_db = float(direct_curve_db[center_index])
    unresolved_component_db = direct_db - UNRESOLVED_MULTIPATH_GAP_DB
    return {
        "path_loss_db": float(path_loss_curve_db[center_index]),
        "path_loss_mean_db": float(np.mean(path_loss_curve_db)),
        "path_loss_std_db": float(np.std(path_loss_curve_db)),
        "slope_db_per_km": slope_db_per_km,
        "direct_db": direct_db,
        "lateral_db": float(unresolved_component_db),
        "reflected_db": float(unresolved_component_db),
        "noise_floor_dbm": noise_floor_dbm,
    }






def apply_matlab_like_axes(axis: plt.Axes) -> None:


    axis.set_facecolor((0.96, 0.96, 0.96))
    axis.grid(True, color="0.75", linewidth=0.8, alpha=0.65)
    axis.tick_params(direction="in", top=True, right=True, length=6)
    axis.tick_params(
        which="minor", direction="in", top=True, right=True, length=3
    )
    axis.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)


def plot_field_components(
    water_depth_values: np.ndarray,
    field_components: Dict[str, np.ndarray | Dict[str, complex | float]],
    title_text: str,
    save_path: str | Path,
    dpi: int = 180,
    show_figure: bool = False,
) -> Path:


    set_chinese_font()
    save_path = Path(save_path).resolve()
    figure, axis = plt.subplots(figsize=(10.0, 7.0))

    axis.plot(
        water_depth_values,
        field_components["total_field_loss_db"],
        "k-",
        linewidth=2.3,
        label="总电场损耗",
    )
    axis.plot(
        water_depth_values,
        field_components["total_power_loss_db"],
        color="#2563A6",
        linestyle="--",
        linewidth=2.0,
        label="总功率流损耗",
    )
    axis.plot(
        water_depth_values,
        field_components["ice_layer_field_loss_db"],
        color="#E17C05",
        linestyle="-.",
        linewidth=1.8,
        label="海冰层整体损耗（含界面）",
    )
    axis.plot(
        water_depth_values,
        field_components["water_bulk_field_loss_db"],
        color="#B62D2D",
        linestyle=":",
        linewidth=2.1,
        label="海水体损耗",
    )

    axis.set_xlabel("水下接收深度 / m", fontsize=13)
    axis.set_ylabel("传播损耗 / dB", fontsize=13)
    axis.set_title(title_text, fontsize=15)
    axis.set_xlim(
        float(np.min(water_depth_values)), float(np.max(water_depth_values))
    )
    axis.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    apply_matlab_like_axes(axis)
    figure.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close(figure)
    return save_path






def main() -> None:



    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    frequency_hz = float(SCENARIO_PARAMS["frequency_hz"])
    ice_thickness_m = float(SCENARIO_PARAMS["ice_thickness_m"])
    incidence_deg = float(SCENARIO_PARAMS["incidence_deg"])
    polarization = str(SCENARIO_PARAMS["polarization"])
    incident_field_v_m = complex(SCENARIO_PARAMS["incident_field_v_m"])

    water_depth_values = np.linspace(
        float(SCENARIO_PARAMS["min_water_depth_m"]),
        float(SCENARIO_PARAMS["max_water_depth_m"]),
        int(SCENARIO_PARAMS["num_points"]),
    )

    field_components = compute_field_components(
        water_depth_values,
        frequency_hz,
        ice_thickness_m,
        incidence_deg,
        polarization,
        incident_field_v_m,
    )

    title_text = (
        f"空气－{ICE_PARAMS['name']}－海水三层传播衰减 "
        f"(f={frequency_hz:g} Hz, 冰厚={ice_thickness_m:g} m, "
        f"{polarization})"
    )
    output_path = Path(__file__).resolve().with_name(
        str(PLOT_PARAMS["output_filename"])
    )
    plot_field_components(
        water_depth_values,
        field_components,
        title_text,
        output_path,
        dpi=int(PLOT_PARAMS["dpi"]),
        show_figure=bool(PLOT_PARAMS["show_figure"]),
    )

    last_index = -1
    final_depth = float(water_depth_values[last_index])
    final_e = complex(field_components["receiver_field_v_m"][last_index])
    final_b = complex(field_components["receiver_magnetic_flux_t"][last_index])
    final_field_loss = float(
        field_components["total_field_loss_db"][last_index]
    )
    final_power_loss = float(
        field_components["total_power_loss_db"][last_index]
    )

    print(f"传播衰减图已生成：{output_path}")
    print(f"水下接收深度：{final_depth:.1f} m")
    print(f"接收电场幅度：{abs(final_e):.6e} V/m")
    print(f"接收磁通密度：{abs(final_b):.6e} T")
    print(f"总电场损耗：{final_field_loss:.3f} dB")
    print(f"总功率流损耗：{final_power_loss:.3f} dB")


if __name__ == "__main__":
    main()
