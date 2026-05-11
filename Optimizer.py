"""
[Task 4] Optimizer.py
Counter-current space marching with shooting method.

x-coordinate data:
    x=0 is hot inlet and cold outlet.
    x=L is hot outlet and cold inlet.

Cold-flow data:
    reversed order of x-coordinate data.
    This is the physically intuitive cold-flow direction, where cold temperature rises
    and cold pressure drops along the flow.
"""

import csv
from Data_model import get_fixed_conditions, H_from_PT, T_from_PH
from Nodal_solver import advance_single_cell


def _make_geom(c, geom_extra):
    return {"A_flow_hot": geom_extra["A_flow_hot"], "A_flow_cold": geom_extra["A_flow_cold"],
            "P_w_hot": geom_extra["P_w_hot"], "P_w_cold": geom_extra["P_w_cold"],
            "D_h": c["geometry"]["D_h"], "t_wall": c["geometry"]["t_wall"], "k_wall": c["geometry"]["k_wall"],
            "correlation_model": geom_extra.get("correlation_model", c["model"].get("correlation_model", "chen"))}


def march_with_guess(L, N, T_cold_x0_guess, P_cold_x0_guess, geom_extra):
    c = get_fixed_conditions()
    fc_hot, fc_cold = c["hot_inlet"], c["cold_inlet"]
    geom = _make_geom(c, geom_extra)
    dx = L / N

    hot = {"fluid": fc_hot["fluid"], "P": fc_hot["P_in"], "H": H_from_PT(fc_hot["fluid"], fc_hot["P_in"], fc_hot["T_in"]), "m_dot": fc_hot["m_dot"]}
    cold = {"fluid": fc_cold["fluid"], "P": P_cold_x0_guess, "H": H_from_PT(fc_cold["fluid"], P_cold_x0_guess, T_cold_x0_guess), "m_dot": fc_cold["m_dot"]}

    node_data = []
    for i in range(N + 1):
        node_data.append({"node": i, "x": i * dx,
                          "T_hot": T_from_PH(hot["fluid"], hot["P"], hot["H"]), "P_hot": hot["P"],
                          "T_cold": T_from_PH(cold["fluid"], cold["P"], cold["H"]), "P_cold": cold["P"],
                          "U": 0.0, "q_cell": 0.0, "q_flux_sp": 0.0,
                          "phase_hot": "", "phase_cold": "", "x_hot": "", "x_cold": ""})
        if i == N:
            break
        hot, cold, info = advance_single_cell(hot, cold, geom, dx)
        node_data[-1].update({"U": info["U"], "q_cell": info["q_cell"], "q_flux_sp": info["q_flux_sp"],
                              "phase_hot": info["phase_hot"], "phase_cold": info["phase_cold"],
                              "x_hot": info["x_hot"], "x_cold": info["x_cold"]})

    T_cold_at_L = node_data[-1]["T_cold"]
    P_cold_at_L = node_data[-1]["P_cold"]
    T_hot_at_L = node_data[-1]["T_hot"]
    return T_cold_at_L, P_cold_at_L, T_hot_at_L, node_data


def shoot_for_boundary(L, N, geom_extra, tol_T=0.2, tol_P=500.0, max_iter=15):
    c = get_fixed_conditions()
    T_cold_in = c["cold_inlet"]["T_in"]
    P_cold_in = c["cold_inlet"]["P_in"]
    T_target = c["target"]["T_cold_out"]

    # In counter-current calculation, x=0 is the cold outlet.
    # Therefore P_cold(x=0) must be lower than P_cold,in at x=L.
    # The previous version updated P_guess directly and could overshoot below zero.
    # Here P_guess is always clamped to a physically meaningful range.
    P_MIN_GUESS = max(1.0e5, 0.05 * P_cold_in)
    P_MAX_GUESS = P_cold_in - 1.0
    P_guess = P_cold_in - 2.0e4

    T_lo = T_cold_in + 0.1
    T_hi = min(c["hot_inlet"]["T_in"] - 1.0, T_target + 30.0)
    last = None

    for _ in range(max_iter):
        T_guess = 0.5 * (T_lo + T_hi)
        P_guess = min(max(P_guess, P_MIN_GUESS), P_MAX_GUESS)

        try:
            T_L, P_L, T_hot_L, data = march_with_guess(L, N, T_guess, P_guess, geom_extra)
        except ValueError:
            # If a trial pressure/temperature becomes non-physical, move the
            # guessed cold outlet pressure upward and reduce the cold outlet
            # temperature guess. This prevents negative pressure calls to CoolProp.
            P_guess = 0.5 * (P_guess + P_MAX_GUESS)
            T_hi = T_guess
            continue

        last = (T_guess, P_guess, T_L, P_L, T_hot_L, data)

        # Pressure correction: x=L should match the specified cold inlet pressure.
        # Use a small relaxation and hard bounds to avoid negative pressure.
        P_error = P_cold_in - P_L
        P_guess = P_guess + 0.15 * P_error
        P_guess = min(max(P_guess, P_MIN_GUESS), P_MAX_GUESS)

        # Temperature shooting: x=L should match actual cold inlet temperature.
        diff_T = T_L - T_cold_in
        if abs(diff_T) < tol_T and abs(P_L - P_cold_in) < tol_P:
            return T_guess, P_guess, data, True
        if diff_T > 0:
            T_hi = T_guess
        else:
            T_lo = T_guess

    if last is None:
        # Fallback: return a safe one-step result rather than crashing.
        T_guess = 0.5 * (T_lo + T_hi)
        P_guess = min(max(P_guess, P_MIN_GUESS), P_MAX_GUESS)
        T_L, P_L, T_hot_L, data = march_with_guess(L, N, T_guess, P_guess, geom_extra)
        last = (T_guess, P_guess, T_L, P_L, T_hot_L, data)
    return last[0], last[1], last[5], False


def optimize_length(N, geom_extra, L_min=0.1, L_max=20.0, tol=0.5, max_iter=12, progress_cb=None):
    c = get_fixed_conditions()
    T_target = c["target"]["T_cold_out"]
    L_lo, L_hi = L_min, L_max
    history = []
    last_result = None

    for it in range(1, max_iter + 1):
        L_mid = 0.5 * (L_lo + L_hi)
        T_x0, P_x0, node_data, ok_inner = shoot_for_boundary(L_mid, N, geom_extra)
        diff = T_x0 - T_target
        history.append((it, L_mid, T_x0, diff, ok_inner))
        if progress_cb:
            progress_cb(it, L_mid, T_x0, diff)
        last_result = (L_mid, T_x0, P_x0, node_data, ok_inner)
        if abs(diff) < tol and ok_inner:
            break
        if diff > 0:
            L_hi = L_mid
        else:
            L_lo = L_mid

    L, T_x0, P_x0, node_data, ok = last_result
    converged = ok and abs(T_x0 - T_target) < tol
    message = "converged" if converged else (
        "not converged: target may require a larger L_max, smaller cold flow rate, or higher hot inlet temperature"
    )
    return {"L": L, "T_cold_out": T_x0, "P_cold_out": P_x0, "node_data": node_data,
            "converged": converged, "message": message, "history": history, "N": N}


def cold_flow_view(node_data):
    reversed_data = []
    nmax = len(node_data) - 1
    for j, row in enumerate(reversed(node_data)):
        r = dict(row)
        r["cold_flow_node"] = j
        r["s_cold"] = j / max(nmax, 1) * node_data[-1]["x"]
        reversed_data.append(r)
    return reversed_data


def save_node_data(node_data, path):
    keys = ["node", "x", "T_hot", "T_cold", "P_hot", "P_cold", "U", "q_cell", "q_flux_sp", "phase_hot", "phase_cold", "x_hot", "x_cold"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in node_data:
            w.writerow({k: r.get(k, "") for k in keys})


def save_cold_flow_data(node_data, path):
    keys = ["cold_flow_node", "s_cold", "node", "x", "T_cold", "P_cold", "phase_cold", "x_cold", "T_hot", "P_hot", "U", "q_cell", "q_flux_sp"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in cold_flow_view(node_data):
            w.writerow({k: r.get(k, "") for k in keys})
