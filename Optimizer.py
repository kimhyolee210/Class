"""
Optimizer.py  -  Water/Water PCHE 열교환기 1D 해석 [Task 4]

역할:
    cold water 출구 온도가 목표값이 되도록 열교환기 길이 L을 찾는다.
    동시에 노드별 데이터(온도/압력/엔탈피/h/U/Re/Nu/f/v/q/ΔP)를 배열로 저장한다.

규칙(과제 명세):
    - 공간 전진 루프      : 노드 i = 0..N-1 forward sweep
    - 출구압력 가정 + 수렴: cold side 분포(T,P)를 초기 가정 후 sweep 반복으로 수렴
    - 노드별 데이터 배열  : numpy ndarray 로 저장
    - 최종 길이 L 반환    : 외부 이분법(bisection) 으로 root-finding
    - 별도 물리계산 금지  : solve_node 호출만 수행 (h/U/f/v/ρ 직접 계산 X)

루프 구조 (명세 매핑):
    [외부]  L 이분법                     : (T_c_outlet - T_target) → 0  (Numeric.tol, max_iter)
    [중간]  cold 분포 + 출구압력 수렴    : sweep 반복             (Numeric.tol, max_iter)
    [내부]  공간 전진 sweep              : node i=0..N-1 forward, solve_node 1회/노드

향류 인덱싱:
              i=0          i=1                    i=N-1        i=N
    hot   :  T_h_in ─→  T_h[1]  ─→ ...  ─→  T_h[N-1]  ─→  T_h[N]
    cold  :  T_c[0] ←─  T_c[1]  ←─ ...  ←─  T_c[N-1]  ←─  T_c_in

    노드 i의 입력 = (T_h[i], T_c[i+1]),  출력 = (T_h[i+1], T_c[i])
"""

import numpy as np

from Data_model import Geometry, HotInlet, ColdInlet, Target, Numeric, H_from_PT, T_from_PH
from Physics_engine import compute_node


# =================================================================
# 공간 전진 sweep + cold 분포 수렴 (한 L 후보에 대한 해석)
# =================================================================

def _allocate_arrays(N):
    """노드별 결과 배열 사전 할당."""
    return {
        # 노드 경계값 (N+1)
        "T_h": np.zeros(N + 1), "P_h": np.zeros(N + 1), "H_h": np.zeros(N + 1),
        "T_c": np.zeros(N + 1), "P_c": np.zeros(N + 1), "H_c": np.zeros(N + 1),
        # 셀(노드 내부) 값 (N)
        "h_hot":  np.zeros(N), "h_cold": np.zeros(N), "U":      np.zeros(N),
        "Re_hot": np.zeros(N), "Re_cold": np.zeros(N),
        "Nu_hot": np.zeros(N), "Nu_cold": np.zeros(N),
        "f_hot":  np.zeros(N), "f_cold":  np.zeros(N),
        "v_hot":  np.zeros(N), "v_cold":  np.zeros(N),
        "q_node": np.zeros(N), "dP_h":    np.zeros(N), "dP_c":   np.zeros(N),
        "x_cold": np.full(N, -1.0), "q_flux_est": np.full(N, np.nan),
        "x_di": np.full(N, np.nan), "R_LL": np.full(N, np.nan),
        "dryout": np.zeros(N, dtype=bool), "h_cold_pre_dryout": np.full(N, np.nan),
        "active_correlation": np.full(N, "", dtype=object),
        "phase_cold": np.full(N, "", dtype=object),
        "subcooled_boiling": np.zeros(N, dtype=bool),
        "T_sat_cold": np.full(N, np.nan), "T_wall_cold_est": np.full(N, np.nan),
        "wall_superheat_sat": np.full(N, np.nan), "delta_T_onb": np.full(N, np.nan),
        "Ja_star": np.full(N, np.nan), "psi_subcooled": np.full(N, np.nan),
        "h_sp_l_subcooled": np.full(N, np.nan),
    }


def forward_sweep(L, geo: Geometry,
                  hot: HotInlet, cold: ColdInlet,
                  target: Target, num: Numeric):
    """
    주어진 L에 대해 공간 전진 sweep 을 수행하고 노드별 분포를 반환한다.
    cold 분포는 초기 추정 후 sweep 반복으로 수렴시킨다 (출구압력 포함).
    """
    N  = num.n_nodes
    dx = L / N
    arr = _allocate_arrays(N)

    # ----- 경계조건 (boundary) -----
    # x=0: hot inlet and the specified cold outlet.
    # Marching x=0 -> L moves opposite to the cold-flow direction, so cold
    # pressure increases and cold enthalpy decreases toward the cold inlet.
    arr["T_h"][0] = hot.T_in;   arr["P_h"][0] = hot.P_in
    arr["T_c"][0] = target.T_cold_out;  arr["P_c"][0] = target.P_cold_out
    arr["H_h"][0] = H_from_PT(hot.P_in,  hot.T_in,  hot.fluid)
    arr["H_c"][0] = H_from_PT(target.P_cold_out, target.T_cold_out, cold.fluid)

    mdot_h_per_ch = hot.mdot  / geo.N_channels
    mdot_c_per_ch = cold.mdot / geo.N_channels
    A_ht = np.pi * geo.Dh * dx

    for i in range(N):
        state_h = {
            "T": arr["T_h"][i], "P": arr["P_h"][i], "mdot": mdot_h_per_ch,
            "fluid": hot.fluid, "H": arr["H_h"][i], "dx": dx,
        }
        state_c = {
            "T": arr["T_c"][i], "P": arr["P_c"][i], "mdot": mdot_c_per_ch,
            "fluid": cold.fluid, "H": arr["H_c"][i], "dx": dx,
            "T_inlet_ref": cold.T_in,
        }

        info = compute_node(state_h, state_c, geo)
        q_node = max(0.0, info["U"] * A_ht * (arr["T_h"][i] - arr["T_c"][i]))
        q_node = min(q_node, mdot_h_per_ch * max(arr["H_h"][i] - H_from_PT(arr["P_h"][i], arr["T_c"][i], hot.fluid), 0.0))
        q_node = min(q_node, mdot_c_per_ch * max(float(num.max_cold_dh_per_cell), 0.0))

        rho_h = info["props_hot"]["rho"]
        rho_c = info["props_cold"]["rho"]
        dP_h = info["f_hot"] * (dx / geo.Dh) * 0.5 * rho_h * info["v_hot"] ** 2
        dP_c = info["f_cold"] * (dx / geo.Dh) * 0.5 * rho_c * info["v_cold"] ** 2

        arr["H_h"][i + 1] = arr["H_h"][i] - q_node / mdot_h_per_ch
        arr["P_h"][i + 1] = arr["P_h"][i] - dP_h
        arr["T_h"][i + 1] = T_from_PH(arr["P_h"][i + 1], arr["H_h"][i + 1], hot.fluid)

        arr["H_c"][i + 1] = arr["H_c"][i] - q_node / mdot_c_per_ch
        arr["P_c"][i + 1] = arr["P_c"][i] + dP_c
        arr["T_c"][i + 1] = T_from_PH(arr["P_c"][i + 1], arr["H_c"][i + 1], cold.fluid)

        arr["h_hot"][i]  = info["h_hot"];  arr["h_cold"][i] = info["h_cold"]
        arr["U"][i]      = info["U"]
        arr["Re_hot"][i] = info["Re_hot"]; arr["Re_cold"][i] = info["Re_cold"]
        arr["Nu_hot"][i] = info["Nu_hot"]; arr["Nu_cold"][i] = info["Nu_cold"]
        arr["f_hot"][i]  = info["f_hot"];  arr["f_cold"][i]  = info["f_cold"]
        arr["v_hot"][i]  = info["v_hot"];  arr["v_cold"][i]  = info["v_cold"]
        arr["q_node"][i] = q_node
        arr["dP_h"][i]   = dP_h;           arr["dP_c"][i]    = dP_c
        arr["x_cold"][i] = info.get("x_cold", -1.0)
        arr["x_di"][i] = info.get("x_di", np.nan)
        arr["R_LL"][i] = info.get("R_LL", np.nan)
        arr["dryout"][i] = info.get("dryout", False)
        arr["h_cold_pre_dryout"][i] = info.get("h_pre_dryout", info["h_cold"])
        arr["active_correlation"][i] = info.get("boiling_correlation", "Dittus-Boelter")
        arr["phase_cold"][i] = info.get("phase_cold", "single")
        arr["subcooled_boiling"][i] = info.get("subcooled_boiling", False)
        arr["T_sat_cold"][i] = info.get("T_sat_cold", np.nan)
        arr["T_wall_cold_est"][i] = info.get("T_wall_cold_est", np.nan)
        arr["wall_superheat_sat"][i] = info.get("wall_superheat_sat", np.nan)
        arr["delta_T_onb"][i] = info.get("delta_T_onb", np.nan)
        arr["Ja_star"][i] = info.get("Ja_star", np.nan)
        arr["psi_subcooled"][i] = info.get("psi_subcooled", np.nan)
        arr["h_sp_l_subcooled"][i] = info.get("h_sp_l_subcooled", np.nan)
        arr["q_flux_est"][i] = info.get("q_flux_est", np.nan)

    return arr


# =================================================================
# 외부 : L 이분법 (cold 출구 온도 = 목표 도달)
# =================================================================

def optimize(geo: Geometry,
             hot: HotInlet, cold: ColdInlet,
             target: Target, num: Numeric,
             L_low: float = 0.01, L_high: float = 50.0):
    """
    cold 출구 온도가 target.T_cold_out 이 되는 L 을 이분법으로 탐색.

    Returns
    -------
    L : float
        목표를 만족하는 열교환기 길이 [m].
    result : dict
        해당 L 에서의 노드별 분포 (forward_sweep 반환과 동일).
    x : ndarray
        축방향 위치 좌표 [m] (노드 경계 N+1개).
    """
    if L_high > 2.0 * L_low:
        L_scan = np.unique(np.r_[np.linspace(L_low, min(2.0, L_high), 7),
                                 np.linspace(min(2.0, L_high), L_high, 7)])
    else:
        L_scan = np.linspace(L_low, L_high, max(9, min(13, num.max_iter + 1)))

    scan = []
    for L in L_scan:
        result_L = forward_sweep(L, geo, hot, cold, target, num)
        residual_L = result_L["T_c"][-1] - cold.T_in
        scan.append((L, residual_L, result_L))

    brackets = []
    for (L_a, r_a, _), (L_b, r_b, _) in zip(scan[:-1], scan[1:]):
        if abs(r_a) < num.tol:
            scan[L_scan.tolist().index(L_a)][2]["converged"] = True
            scan[L_scan.tolist().index(L_a)][2]["cold_inlet_residual_K"] = r_a
            scan[L_scan.tolist().index(L_a)][2]["target_residual_K"] = 0.0
            x = np.linspace(0.0, L_a, num.n_nodes + 1)
            return L_a, scan[L_scan.tolist().index(L_a)][2], x
        if r_a * r_b < 0.0:
            brackets.append((L_a, L_b))

    if not brackets:
        L_best, residual_best, result_best = min(scan, key=lambda item: abs(item[1]))
        result_best["converged"] = False
        result_best["cold_inlet_residual_K"] = residual_best
        result_best["target_residual_K"] = result_best["T_c"][0] - target.T_cold_out
        print(
            "  optimizer warning: target is not bracketed in "
            f"{L_low:.4f}-{L_high:.4f} m; reporting closest checked result "
            f"L={L_best:.4f} m, T_c_in={result_best['T_c'][-1]:.3f} K, "
            f"inlet_residual={residual_best:.3f} K"
        )
        x = np.linspace(0.0, L_best, num.n_nodes + 1)
        return L_best, result_best, x

    L_lo, L_hi = brackets[0]
    L_mid      = 0.5 * (L_lo + L_hi)
    result     = None

    for it in range(num.max_iter):
        L_mid  = 0.5 * (L_lo + L_hi)
        result = forward_sweep(L_mid, geo, hot, cold, target, num)

        T_c_inlet = result["T_c"][-1]
        residual = T_c_inlet - cold.T_in
        print(
            f"  optimizer {it+1:02d}/{num.max_iter}: L={L_mid:.4f} m, "
            f"T_c_in={T_c_inlet:.3f} K, inlet_residual={residual:.3f} K"
        )

        if abs(residual) < num.tol:
            break

        # L 증가 → x=L에서 역산된 cold inlet 온도 감소
        if residual > 0.0:
            L_lo = L_mid
        else:
            L_hi = L_mid

    result["converged"] = (
        abs(result["T_c"][-1] - cold.T_in) < num.tol
        and abs(result["T_c"][0] - target.T_cold_out) < num.tol
        and abs(result["P_c"][0] - target.P_cold_out) < num.tol * target.P_cold_out
    )
    result["target_residual_K"] = result["T_c"][0] - target.T_cold_out
    result["cold_inlet_residual_K"] = result["T_c"][-1] - cold.T_in
    x = np.linspace(0.0, L_mid, num.n_nodes + 1)
    return L_mid, result, x
