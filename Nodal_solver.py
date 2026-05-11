"""
[Task 3] Nodal_solver.py
Counter-current single-cell solver.

Coordinate convention:
    x = 0 -> L is the HOT flow direction.
    COLD actual flow is L -> 0.

Therefore, when marching numerically from x=0 to x=L:
    Hot enthalpy decreases.
    Cold enthalpy decreases as well, because we are moving opposite to the cold flow.
    Cold pressure increases in the x direction, because x=L is the cold inlet pressure.
"""

from Physics_engine import evaluate_node_from_PH, compute_U_for_pair


def advance_single_cell(hot_state, cold_state, geom, dx):
    # Preliminary single-phase based heat flux estimate.
    hot_sp = evaluate_node_from_PH(hot_state["fluid"], hot_state["P"], hot_state["H"], hot_state["m_dot"], geom["A_flow_hot"], geom["D_h"], mode="cooling", correlation_model="single_phase")
    cold_sp = evaluate_node_from_PH(cold_state["fluid"], cold_state["P"], cold_state["H"], cold_state["m_dot"], geom["A_flow_cold"], geom["D_h"], mode="heating", correlation_model="single_phase")
    U_sp = compute_U_for_pair(hot_sp, cold_sp, geom["t_wall"], geom["k_wall"])
    dT = hot_sp["T"] - cold_sp["T"]
    q_flux_sp = max(U_sp * dT, 0.0)

    model = geom.get("correlation_model", "single_phase")
    hot_node = evaluate_node_from_PH(hot_state["fluid"], hot_state["P"], hot_state["H"], hot_state["m_dot"], geom["A_flow_hot"], geom["D_h"], mode="cooling", correlation_model=model, q_flux_sp=q_flux_sp, P_H=geom.get("P_w_hot"), P_F=geom.get("P_w_hot"))
    cold_node = evaluate_node_from_PH(cold_state["fluid"], cold_state["P"], cold_state["H"], cold_state["m_dot"], geom["A_flow_cold"], geom["D_h"], mode="heating", correlation_model=model, q_flux_sp=q_flux_sp, P_H=geom.get("P_w_cold"), P_F=geom.get("P_w_cold"))
    U = compute_U_for_pair(hot_node, cold_node, geom["t_wall"], geom["k_wall"])

    P_w_avg = 0.5 * (geom["P_w_hot"] + geom["P_w_cold"])
    dA = P_w_avg * dx
    dT = hot_node["T"] - cold_node["T"]
    q_cell = max(U * dA * dT, 0.0)

    # Energy update in x direction.
    H_hot_new = hot_state["H"] - q_cell / hot_state["m_dot"]
    H_cold_new = cold_state["H"] - q_cell / cold_state["m_dot"]

    # Pressure update in x direction.
    dP_hot = hot_node["f"] * (dx / geom["D_h"]) * hot_node["rho"] * hot_node["V"] ** 2 / 2.0
    dP_cold = cold_node["f"] * (dx / geom["D_h"]) * cold_node["rho"] * cold_node["V"] ** 2 / 2.0
    P_hot_new = hot_state["P"] - dP_hot
    P_cold_new = cold_state["P"] + dP_cold

    P_MIN = 1e5
    if P_hot_new < P_MIN or P_cold_new < P_MIN:
        raise ValueError("Pressure became too low. Increase flow area or reduce length step.")

    hot_next = {"fluid": hot_state["fluid"], "m_dot": hot_state["m_dot"], "P": P_hot_new, "H": H_hot_new}
    cold_next = {"fluid": cold_state["fluid"], "m_dot": cold_state["m_dot"], "P": P_cold_new, "H": H_cold_new}

    info = {"U": U, "U_sp": U_sp, "q_flux_sp": q_flux_sp, "q_cell": q_cell, "dA": dA, "dT": dT,
            "h_hot": hot_node["h"], "h_cold": cold_node["h"], "Re_hot": hot_node["Re"], "Re_cold": cold_node["Re"],
            "phase_hot": hot_node["phase"], "phase_cold": cold_node["phase"], "x_hot": hot_node["x"], "x_cold": cold_node["x"],
            "T_hot": hot_node["T"], "T_cold": cold_node["T"], "dP_hot": dP_hot, "dP_cold": dP_cold}
    return hot_next, cold_next, info
