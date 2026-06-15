"""
Main.py  -  Water/Water PCHE 열교환기 1D 해석 [Task 5]

역할 (과제 명세):
    1) 시뮬레이션 실행 (함수호출)         : Optimizer.optimize 로 L 도출 + 노드별 분포
    2) 데이터 저장 (csv)                   : 노드 / 셀 / summary 3종
    3) 데이터 로드 및 시각화 (matplotlib)  : csv → DataFrame → 2×3 subplot

전체 모듈 구성:
    Data_model.py     : 고정조건 dataclass + CoolProp 물성치 래퍼
    Physics_engine.py : 노드 단면의 (h, U, Re, Nu, f, v, ρ) 도출
    Nodal_solver.py   : 향류 single-node 1-pass (에너지 보존 + 변수 업데이트)
    Optimizer.py      : 공간 전진 sweep + cold 분포 수렴 + L 이분법
    Main.py           : ↑ 통합 실행 + csv 저장 + 시각화 (이 파일)

문제 조건:
    Hot  (Water) : T_in 600 K, P_in 15.0 MPa, inlet velocity 2 m/s
    Cold (Water): T_in 530 K, P_in 6.0 MPa, same mdot as hot side
    Geometry      : Dh 2 mm, t_wall 1 mm, k_wall 20 W/m·K
    Target        : cold-side outlet temperature = 550 K → 필요한 길이 L 도출 (대향류)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Data_model import Geometry, Target, Numeric, make_water_inlets
from Optimizer import optimize


# =================================================================
# 파일 경로 (Main.py 와 같은 폴더에 저장)
# =================================================================

HERE         = Path(__file__).parent
NODES_CSV    = HERE / "exam_nodes.csv"
CELLS_CSV    = HERE / "exam_cells.csv"
SUMMARY_CSV  = HERE / "exam_summary.csv"


# =================================================================
# 1) 시뮬레이션 실행 + CSV 저장
# =================================================================

def run_simulation():
    """Data_model 입력 → Optimizer.optimize → csv 3종 저장."""
    geo    = Geometry()
    target = Target()
    hot, cold = make_water_inlets(geo, hot_velocity=2.0, target=target)
    num    = Numeric()

    print("=" * 64)
    print("Water/Water PCHE Counter-current HX  -  Sizing Run")
    print("=" * 64)
    print(f"  Hot  : {hot.fluid:7s}  T_in={hot.T_in:7.2f} K ({hot.T_in-273.15:6.1f} °C), "
          f"P_in={hot.P_in/1e6:5.2f} MPa, mdot={hot.mdot:.2f} kg/s")
    print(f"  Cold : {cold.fluid:7s}  T_in={cold.T_in:7.2f} K ({cold.T_in-273.15:6.1f} °C), "
          f"P_in={cold.P_in/1e6:5.2f} MPa, mdot={cold.mdot:.2f} kg/s")
    print(f"  Geom : Dh={geo.Dh*1e3:.1f} mm, t_wall={geo.t_wall*1e3:.1f} mm, "
          f"k_wall={geo.k_wall:.1f} W/m·K, N_channels={geo.N_channels}, "
          f"boiling={geo.boiling_correlation}")
    print(f"  Target T_cold_out = {target.T_cold_out:.2f} K ({target.T_cold_out-273.15:.2f} °C) "
          f"(N={num.n_nodes} nodes, tol={num.tol})")
    print("-" * 64)

    L, result, x = optimize(geo, hot, cold, target, num)
    x_mid = 0.5 * (x[:-1] + x[1:])
    dx = L / num.n_nodes

    # Cell evaluation-point fluid and wall temperatures.
    # q_node is based on one channel, so the corresponding heat-transfer
    # area is also the one-channel wetted area. Use the same fluid states
    # used by the solver to keep the convection and wall-conduction
    # temperature drops consistent with q_node.
    T_h_bulk = result["T_h"][:-1]
    T_c_bulk = result["T_c"][:-1]
    A_ht_cell = np.pi * geo.Dh * dx
    q_flux_cell_actual = result["q_node"] / A_ht_cell
    q_flux_wall = result["U"] * np.maximum(T_h_bulk - T_c_bulk, 0.0)
    T_wall_hot = T_h_bulk - q_flux_wall / np.maximum(result["h_hot"], 1.0e-12)
    T_wall_cold = T_c_bulk + q_flux_wall / np.maximum(result["h_cold"], 1.0e-12)
    wall_conduction_delta_T = T_wall_hot - T_wall_cold

    # ---- 노드 경계 (N+1 행) ----
    df_nodes = pd.DataFrame({
        "x_m":      x,
        "T_h_K":    result["T_h"],
        "T_c_K":    result["T_c"],
        "T_h_C":    result["T_h"] - 273.15,
        "T_c_C":    result["T_c"] - 273.15,
        "P_h_Pa":   result["P_h"],
        "P_c_Pa":   result["P_c"],
        "P_h_MPa":  result["P_h"] / 1.0e6,
        "P_c_MPa":  result["P_c"] / 1.0e6,
        "H_h_Jkg":  result["H_h"],
        "H_c_Jkg":  result["H_c"],
    })

    # ---- 셀 (N 행) ----
    df_cells = pd.DataFrame({
        "x_mid_m":  x_mid,
        "T_h_bulk_K": T_h_bulk,
        "T_c_bulk_K": T_c_bulk,
        "T_wall_hot_K": T_wall_hot,
        "T_wall_cold_K": T_wall_cold,
        "T_h_bulk_C": T_h_bulk - 273.15,
        "T_c_bulk_C": T_c_bulk - 273.15,
        "T_wall_hot_C": T_wall_hot - 273.15,
        "T_wall_cold_C": T_wall_cold - 273.15,
        "wall_delta_T_K": wall_conduction_delta_T,
        "q_flux_wall_Wm2": q_flux_wall,
        "q_flux_cell_actual_Wm2": q_flux_cell_actual,
        "q_flux_to_atm_CHF_ref": q_flux_wall / 1.0e6,
        "exceeds_atm_CHF_ref": q_flux_wall >= 1.0e6,
        "h_hot":    result["h_hot"],
        "h_cold":   result["h_cold"],
        "U":        result["U"],
        "Re_hot":   result["Re_hot"],
        "Re_cold":  result["Re_cold"],
        "Nu_hot":   result["Nu_hot"],
        "Nu_cold":  result["Nu_cold"],
        "f_hot":    result["f_hot"],
        "f_cold":   result["f_cold"],
        "v_hot":    result["v_hot"],
        "v_cold":   result["v_cold"],
        "q_node":   result["q_node"],
        "q_cum":    np.cumsum(result["q_node"]),
        "dP_h":     result["dP_h"],
        "dP_c":     result["dP_c"],
        "x_cold":  result.get("x_cold", np.full_like(x_mid, -1.0)),
        "x_di_del_col": result.get("x_di", np.full_like(x_mid, np.nan)),
        "x_over_x_di": np.divide(
            result.get("x_cold", np.full_like(x_mid, -1.0)),
            result.get("x_di", np.full_like(x_mid, np.nan)),
            out=np.full_like(x_mid, np.nan),
            where=result.get("x_di", np.zeros_like(x_mid)) > 0.0,
        ),
        "R_LL": result.get("R_LL", np.full_like(x_mid, np.nan)),
        "dryout": result.get("dryout", np.zeros_like(x_mid, dtype=bool)),
        "active_correlation": result.get("active_correlation", np.full(len(x_mid), "")),
        "phase_cold": result.get("phase_cold", np.full(len(x_mid), "")),
        "subcooled_boiling": result.get("subcooled_boiling", np.zeros_like(x_mid, dtype=bool)),
        "T_sat_cold_K": result.get("T_sat_cold", np.full_like(x_mid, np.nan)),
        "T_sat_cold_C": result.get("T_sat_cold", np.full_like(x_mid, np.nan)) - 273.15,
        "T_wall_cold_est_K": result.get("T_wall_cold_est", np.full_like(x_mid, np.nan)),
        "T_wall_cold_est_C": result.get("T_wall_cold_est", np.full_like(x_mid, np.nan)) - 273.15,
        "wall_superheat_sat_K": result.get("wall_superheat_sat", np.full_like(x_mid, np.nan)),
        "delta_T_onb_K": result.get("delta_T_onb", np.full_like(x_mid, np.nan)),
        "Ja_star": result.get("Ja_star", np.full_like(x_mid, np.nan)),
        "psi_subcooled": result.get("psi_subcooled", np.full_like(x_mid, np.nan)),
        "psi_subcooled_raw": result.get("psi_subcooled_raw", np.full_like(x_mid, np.nan)),
        "h_sp_l_subcooled": result.get("h_sp_l_subcooled", np.full_like(x_mid, np.nan)),
        "h_cold_pre_dryout": result.get("h_cold_pre_dryout", np.full_like(x_mid, np.nan)),
        "h_cold_post_dryout": result.get("h_cold_post_dryout", np.full_like(x_mid, np.nan)),
        "dryout_weight": result.get("dryout_weight", np.zeros_like(x_mid)),
        "q_flux_est_Wm2": result.get("q_flux_est", np.full_like(x_mid, np.nan)),
    })

    # ---- Summary (에너지 보존 cross-check 포함, 전체 기준) ----
    # sweep 은 단위 채널 기준 → q_node 도 단위 채널. 전체 환산은 × N_channels.
    Q_total = float(np.sum(result["q_node"])) * geo.N_channels
    Q_hot   = hot.mdot  * (result["H_h"][0]  - result["H_h"][-1])   # 전체 (mdot × ΔH)
    Q_cold  = cold.mdot * (result["H_c"][0]  - result["H_c"][-1])   # 전체

    summary = {
        "L_m":             L,
        "N_nodes":         num.n_nodes,
        "N_channels":      geo.N_channels,
        "cold_mdot_ratio": cold.mdot / hot.mdot,
        "T_h_in_C":        hot.T_in - 273.15,
        "T_h_out_C":       result["T_h"][-1] - 273.15,
        "T_c_in_C":        cold.T_in - 273.15,
        "T_c_out_C":       result["T_c"][0]  - 273.15,
        "T_c_target_C":    target.T_cold_out - 273.15,
        "P_h_in_MPa":      hot.P_in  / 1.0e6,
        "P_h_out_MPa":     result["P_h"][-1] / 1.0e6,
        "P_c_in_MPa":      result["P_c"][-1] / 1.0e6,
        "P_c_out_MPa":     result["P_c"][0]  / 1.0e6,
        "P_c_target_MPa":   target.P_cold_out / 1.0e6,
        "dP_h_kPa":        (hot.P_in  - result["P_h"][-1]) / 1.0e3,
        "dP_c_kPa":        (result["P_c"][-1] - result["P_c"][0])  / 1.0e3,
        "Q_total_W":       Q_total,
        "Q_hot_W":         float(Q_hot),
        "Q_cold_W":        float(Q_cold),
        "U_avg_Wm2K":      float(np.mean(result["U"])),
        "T_wall_hot_min_C": float(np.min(T_wall_hot) - 273.15),
        "T_wall_hot_max_C": float(np.max(T_wall_hot) - 273.15),
        "T_wall_cold_min_C": float(np.min(T_wall_cold) - 273.15),
        "T_wall_cold_max_C": float(np.max(T_wall_cold) - 273.15),
        "boiling_correlation": geo.boiling_correlation,
        "max_cold_quality": float(np.nanmax(result.get("x_cold", [-1.0]))),
        "min_dryout_quality_x_di": float(np.nanmin(result.get("x_di", [np.nan]))),
        "dryout_cell_count": int(np.count_nonzero(result.get("dryout", []))),
        "subcooled_boiling_cell_count": int(np.count_nonzero(result.get("subcooled_boiling", []))),
        "max_q_flux_wall_Wm2": float(np.max(q_flux_wall)),
        "max_q_flux_to_atm_CHF_ref": float(np.max(q_flux_wall) / 1.0e6),
        "optimizer_converged": bool(result.get("converged", False)),
        "target_residual_K": float(result.get("target_residual_K", result["T_c"][0] - target.T_cold_out)),
    }
    df_summary = pd.DataFrame([summary])

    df_nodes  .to_csv(NODES_CSV,   index=False, encoding="utf-8-sig")
    df_cells  .to_csv(CELLS_CSV,   index=False, encoding="utf-8-sig")
    df_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    if summary["optimizer_converged"]:
        print(f"  L_required        = {L:10.6f}  m")
    else:
        print(f"  L_best_checked    = {L:10.6f}  m  (target not reached)")
    print(f"  T_h : in/out      = {summary['T_h_in_C']:7.2f} / {summary['T_h_out_C']:7.2f}  °C")
    print(f"  T_c : in/out      = {summary['T_c_in_C']:7.2f} / {summary['T_c_out_C']:7.2f}  °C  "
          f"(target {summary['T_c_target_C']:.1f}, residual {summary['target_residual_K']:+.3f} K)")
    print(f"  mdot cold/hot     = {summary['cold_mdot_ratio']:10.4f}")
    print(f"  ΔP : hot / cold   = {summary['dP_h_kPa']:7.2f} / {summary['dP_c_kPa']:7.2f}  kPa")
    print(f"  Q  : total / hot / cold = "
          f"{Q_total/1e3:8.2f} / {Q_hot/1e3:8.2f} / {Q_cold/1e3:8.2f}  kW")
    print(f"  U_avg             = {summary['U_avg_Wm2K']:10.2f}  W/m²K")
    print(f"  Wall T hot min/max  = {summary['T_wall_hot_min_C']:7.2f} / "
          f"{summary['T_wall_hot_max_C']:7.2f}  °C")
    print(f"  Wall T cold min/max = {summary['T_wall_cold_min_C']:7.2f} / "
          f"{summary['T_wall_cold_max_C']:7.2f}  °C")
    print(f"  Dryout cells       = {summary['dryout_cell_count']:10d}  "
          f"(min x_di={summary['min_dryout_quality_x_di']:.4f})")
    print(f"  Subcooled boiling  = {summary['subcooled_boiling_cell_count']:10d}  cells")
    print(f"  Max wall q''       = {summary['max_q_flux_wall_Wm2']/1.0e6:10.4f}  MW/m² "
          f"({summary['max_q_flux_to_atm_CHF_ref']:.3f} × 1 MW/m² atm reference)")
    print("-" * 64)
    print(f"  CSV saved:")
    print(f"    {NODES_CSV.name}")
    print(f"    {CELLS_CSV.name}")
    print(f"    {SUMMARY_CSV.name}")
    print("=" * 64)


# =================================================================
# 2) CSV 로드 + 시각화
# =================================================================

def load_and_plot():
    """csv 다시 읽어 2×3 subplot으로 시각화."""
    df_nodes   = pd.read_csv(NODES_CSV)
    df_cells   = pd.read_csv(CELLS_CSV)
    df_summary = pd.read_csv(SUMMARY_CSV)

    L_val    = df_summary["L_m"].iloc[0]
    target_C = df_summary["T_c_target_C"].iloc[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Water/Water PCHE Counter-current HX  (L = {L_val:.4f} m)",
        fontsize=14, fontweight="bold",
    )

    # (0,0) 온도 분포
    ax = axes[0, 0]
    ax.plot(df_nodes["x_m"], df_nodes["T_h_C"], "r-",  lw=1.6, label="Hot Water")
    ax.plot(df_nodes["x_m"], df_nodes["T_c_C"], "b-",  lw=1.6, label="Cold Water")
    ax.plot(df_cells["x_mid_m"], df_cells["T_wall_hot_C"], color="darkorange",
            linestyle="--", lw=1.4, label="Hot-side Wall")
    ax.plot(df_cells["x_mid_m"], df_cells["T_wall_cold_C"], color="teal",
            linestyle="--", lw=1.4, label="Cold-side Wall")
    valid_tsat = df_cells["T_sat_cold_C"].notna()
    ax.plot(df_cells.loc[valid_tsat, "x_mid_m"], df_cells.loc[valid_tsat, "T_sat_cold_C"],
            color="gray", linestyle=":", lw=1.3, label="Cold Saturation")
    subcooled_cells = df_cells[
        df_cells["subcooled_boiling"].astype(str).str.lower() == "true"
    ]
    if not subcooled_cells.empty:
        ax.scatter(subcooled_cells["x_mid_m"], subcooled_cells["T_wall_cold_C"],
                   color="magenta", marker="^", s=32, zorder=4,
                   label="Subcooled Boiling")
    ax.axhline(target_C, color="k", linestyle=":", lw=0.9,
               label=f"Target {target_C:.0f} °C")
    ax.set_xlabel("Axial position x [m]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Temperature Profile")
    ax.grid(alpha=0.3); ax.legend()

    # (0,1) 압력 분포
    ax = axes[0, 1]
    ax.plot(df_nodes["x_m"], df_nodes["P_h_MPa"], "r-", lw=1.6, label="Hot Water")
    ax.plot(df_nodes["x_m"], df_nodes["P_c_MPa"], "b-", lw=1.6, label="Cold Water")
    ax.set_xlabel("Axial position x [m]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Pressure Profile")
    ax.grid(alpha=0.3); ax.legend()

    # (0,2) 열전달계수 h, U
    ax = axes[0, 2]
    ax.plot(df_cells["x_mid_m"], df_cells["h_hot"],  "r-",  lw=1.4, label="h_hot")
    ax.plot(df_cells["x_mid_m"], df_cells["h_cold"], "b-",  lw=1.4, label="h_cold")
    ax.plot(df_cells["x_mid_m"], df_cells["U"],      "k--", lw=1.4, label="U (overall)")
    ax.set_xlabel("Axial position x [m]")
    ax.set_ylabel("HTC [W/m²K]")
    ax.set_title("Heat Transfer Coefficients")
    ax.grid(alpha=0.3); ax.legend()

    # (1,0) Reynolds
    ax = axes[1, 0]
    ax.plot(df_cells["x_mid_m"], df_cells["Re_hot"],  "r-", lw=1.4, label="Re_hot")
    ax.plot(df_cells["x_mid_m"], df_cells["Re_cold"], "b-", lw=1.4, label="Re_cold")
    ax.set_xlabel("Axial position x [m]")
    ax.set_ylabel("Reynolds Number")
    ax.set_title("Reynolds Number")
    ax.grid(alpha=0.3); ax.legend()

    # (1,1) vapor quality and Del Col dryout-inception quality
    ax = axes[1, 1]
    two_phase_cells = df_cells[
        (df_cells["x_cold"] > 0.0)
        & (df_cells["x_cold"] < 1.0)
        & (df_cells["x_di_del_col"] > 0.0)
    ]
    ax.plot(two_phase_cells["x_mid_m"], two_phase_cells["x_cold"],
            "b-", marker="o", markersize=3, lw=1.5, label="x")
    ax.plot(two_phase_cells["x_mid_m"], two_phase_cells["x_di_del_col"],
            "k--", marker="s", markersize=3, lw=1.5,
            label="x_di (Del Col)")
    dryout_cells = two_phase_cells[
        two_phase_cells["dryout"].astype(str).str.lower() == "true"
    ]
    if not dryout_cells.empty:
        ax.scatter(dryout_cells["x_mid_m"], dryout_cells["x_cold"],
                   color="red", s=24, zorder=3, label="Dougall-Rohsenow")
    ax.set_xlabel("Axial position x [m]")
    ax.set_ylabel("Vapor quality [-]")
    ax.set_title("Dryout Criterion")
    ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend()

    # (1,2) 누적 열량
    ax = axes[1, 2]
    ax.plot(df_cells["x_mid_m"], df_cells["q_cum"] / 1.0e3,
            "g-", lw=1.6, label="cumulative q")
    ax.set_xlabel("Axial position x [m]")
    ax.set_ylabel("Cumulative Q [kW]")
    ax.set_title("Cumulative Heat Transfer")
    ax.grid(alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.show()


# =================================================================
# Entry point
# =================================================================

if __name__ == "__main__":
    run_simulation()
    load_and_plot()
