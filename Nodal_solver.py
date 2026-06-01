"""
Nodal_solver.py  -  Water/Water PCHE 열교환기 1D 해석 [Task 3]

역할:
    대향류(Counter-current) 단일 노드(single-cell) 1-pass 해석기.
    한 노드 양 끝의 입구 상태를 받아 출구 상태를 1회만 도출한다.

규칙(과제 명세):
    - 단일 노드 (single node)        : 내부 수렴 루프 없음 (for/while 사용 X).
    - 별도 물리계산 사용 금지         : h, U, f, Re, Nu, v, ρ 도출은
                                       Physics_engine.compute_node 만 호출.
    - 에너지 보존 + 변수 업데이트 만 :
        q = U·A·ΔT,
        H_out = H_in ± q/ṁ,
        ΔP = f·(dx/Dh)·½ρv²,
        P_out = P_in − ΔP,
        T_out = T(P_out, H_out, fluid).
    - 공간 전진 / 출구압력 가정·수렴 루프 / cell-average 수렴은 Optimizer 가 담당.

향류 노드 구조 (노드 길이 dx) :

           좌단 (x_i)                              우단 (x_{i+1})
   hot   :  T_h_in   --hot 흐름→        T_h_out
   cold  :  T_c_out  ←cold 흐름--       T_c_in

    한 노드 입력 = (좌단 hot inlet, 우단 cold inlet)
    한 노드 출력 = (우단 hot outlet, 좌단 cold outlet)
"""

import numpy as np

from Data_model import Geometry, T_from_PH, H_from_PT, MIN_PRESSURE_PA
from Physics_engine import compute_node


# =================================================================
# Single-node counter-current solver  (1-pass, no internal loop)
# =================================================================

def solve_node(state_h_in, state_c_in, dx, geo: Geometry):
    """
    한 노드의 향류 해석을 1회 평가한다.

    Parameters
    ----------
    state_h_in : dict  좌단의 hot 입구 상태.
        keys = "T"[K], "P"[Pa], "mdot"[kg/s], "fluid"[str]
    state_c_in : dict  우단의 cold 입구 상태 (향류이므로 cold 흐름 방향의 입구).
    dx : float         노드 길이 [m].
    geo : Geometry     Data_model.Geometry 인스턴스.

    Returns
    -------
    dict  (출구 상태 + 노드 부수 변수)
    """
    # --- 입구 enthalpy (변수 업데이트용) -------------------------
    # dict.get(default)는 default 식을 먼저 계산하므로 CoolProp saturation 오류가 날 수 있음.
    # 따라서 H가 이미 있으면 H_from_PT를 호출하지 않는다.
    if "H" in state_h_in and state_h_in["H"] is not None:
        H_h_in = state_h_in["H"]
    else:
        H_h_in = H_from_PT(state_h_in["P"], state_h_in["T"], state_h_in["fluid"])

    if "H" in state_c_in and state_c_in["H"] is not None:
        H_c_in = state_c_in["H"]
    else:
        H_c_in = H_from_PT(state_c_in["P"], state_c_in["T"], state_c_in["fluid"])

    # --- Physics_engine : 입구 상태로 1회 평가 (h, U, f, v, ρ) ---
    info = compute_node(state_h_in, state_c_in, geo)

    # --- 에너지 보존 : 노드 양 끝 입구의 ΔT 사용 ----------------
    A_ht   = np.pi * geo.Dh * dx
    dT_in  = state_h_in["T"] - state_c_in["T"]

    # 물-물 비등 문제에서는 cold 초기 추정/반복 중 일부 노드에서
    # cold 쪽 온도가 hot보다 높게 추정될 수 있다. 이때 q가 음수가 되면
    # cold 엔탈피가 비정상적으로 감소하여 CoolProp P-H flash 오류가 발생한다.
    # 따라서 이 과제의 열교환기 방향(Hot -> Cold)에 맞게 음의 열전달은 0으로 제한한다.
    q_raw = info["U"] * A_ht * dT_in
    q_node = max(0.0, q_raw)

    # Keep a coarse cell from overshooting the local thermodynamic bounds.
    # Without this limiter, a long trial length in the optimizer can remove
    # more enthalpy from the hot stream than it physically contains, which
    # creates artificial end-point temperature jumps.
    if q_node > 0.0:
        dT_guard = 1.0e-6
        H_h_floor = H_from_PT(
            state_h_in["P"],
            min(state_h_in["T"], state_c_in["T"] + dT_guard),
            state_h_in["fluid"],
        )
        H_c_ceiling = H_from_PT(
            state_c_in["P"],
            max(state_c_in["T"], state_h_in["T"] - dT_guard),
            state_c_in["fluid"],
        )
        q_hot_limit = state_h_in["mdot"] * max(H_h_in - H_h_floor, 0.0)
        q_cold_limit = state_c_in["mdot"] * max(H_c_ceiling - H_c_in, 0.0)
        q_cold_step_limit = state_c_in["mdot"] * max(
            float(state_c_in.get("max_dH", np.inf)), 0.0
        )
        q_node = min(q_node, q_hot_limit, q_cold_limit, q_cold_step_limit)

    H_h_out = H_h_in - q_node / state_h_in["mdot"]
    H_c_out = H_c_in + q_node / state_c_in["mdot"]

    # --- 압력 강하 (각 유체의 자기 흐름 방향 in→out) -------------
    rho_h = info["props_hot" ]["rho"]
    rho_c = info["props_cold"]["rho"]
    dP_h_raw = info["f_hot" ] * (dx / geo.Dh) * 0.5 * rho_h * info["v_hot" ] ** 2
    dP_c_raw = info["f_cold"] * (dx / geo.Dh) * 0.5 * rho_c * info["v_cold"] ** 2

    # 수치 안정화:
    # 비등 구간에서 혼합밀도/유속이 급변하면 단일 노드 압력강하가
    # 입구압력보다 크게 계산되어 P_out이 음수가 될 수 있다.
    # CoolProp는 음압에서 P-H flash를 수행할 수 없으므로 각 노드의
    # 압력강하는 현재 압력의 20% 이하, 그리고 MIN_PRESSURE_PA 이상을
    # 남기는 범위로 제한한다.
    dP_h = min(dP_h_raw, 0.20 * state_h_in["P"], max(state_h_in["P"] - MIN_PRESSURE_PA, 0.0))
    dP_c = min(dP_c_raw, 0.20 * state_c_in["P"], max(state_c_in["P"] - MIN_PRESSURE_PA, 0.0))

    P_h_out = max(state_h_in["P"] - dP_h, MIN_PRESSURE_PA)
    P_c_out = max(state_c_in["P"] - dP_c, MIN_PRESSURE_PA)

    # --- 출구 온도 갱신 (T from P,H) ------------------------------
    T_h_out = T_from_PH(P_h_out, H_h_out, state_h_in["fluid"])
    T_c_out = T_from_PH(P_c_out, H_c_out, state_c_in["fluid"])

    # --- 결과 ----------------------------------------------------
    return {
        # 출구 상태
        "T_h_out": T_h_out, "P_h_out": P_h_out, "H_h_out": H_h_out,
        "T_c_out": T_c_out, "P_c_out": P_c_out, "H_c_out": H_c_out,
        # 노드 열량 / 압력강하
        "q_node": q_node, "dP_h": dP_h, "dP_c": dP_c,
        # Physics_engine 부수 결과
        "h_hot":   info["h_hot"],   "h_cold":  info["h_cold"],   "U":      info["U"],
        "Re_hot":  info["Re_hot"],  "Re_cold": info["Re_cold"],
        "Nu_hot":  info["Nu_hot"],  "Nu_cold": info["Nu_cold"],
        "f_hot":   info["f_hot"],   "f_cold":  info["f_cold"],
        "v_hot":   info["v_hot"],   "v_cold":  info["v_cold"],
        "x_cold": info.get("x_cold", -1.0),
        "phase_cold": info.get("phase_cold", "single"),
        "boiling_correlation": info.get("boiling_correlation", "Dittus-Boelter"),
        "q_flux_est": info.get("q_flux_est", np.nan),
    }
