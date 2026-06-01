"""
Physics_engine.py  -  Water/Water boiling heat-exchanger 1D analysis [Task 2]

Cold side can become two-phase. In subcooled/superheated single-phase
regions, Dittus-Boelter is used. In 0 < quality < 1, the selected boiling
correlation in Geometry.boiling_correlation is used through correlation.py.
"""

import numpy as np

from Data_model import Geometry, props_PT, phase_quality_PH
from correlation import saturated_water_props, two_phase_boiling_h


def nusselt_DB(Re, Pr, mode):
    """Single-phase forced convection Nusselt number."""
    if Re < 2300.0:
        return 4.36
    if mode == "heating":
        return 0.023 * Re ** 0.8 * Pr ** 0.4
    if mode == "cooling":
        return 0.023 * Re ** 0.8 * Pr ** 0.3
    raise ValueError(f"Unknown nusselt mode: {mode}")


def friction_factor(Re):
    """Darcy friction factor for a smooth circular channel."""
    Re = max(float(Re), 1.0e-12)
    if Re < 2300.0:
        return 64.0 / Re
    return (-1.8 * np.log10(6.9 / Re)) ** -2


def _channel_area(Dh):
    return 0.25 * np.pi * Dh ** 2


def _single_phase_transport(state, Dh, A_flow, mode):
    pr = props_PT(state["P"], state["T"], state["fluid"])
    v = state["mdot"] / (pr["rho"] * A_flow)
    G = state["mdot"] / A_flow
    Re = pr["rho"] * v * Dh / pr["mu"]
    Nu = nusselt_DB(Re, pr["Pr"], mode)
    h = Nu * pr["k"] / Dh
    f = friction_factor(Re)
    return {
        "props": pr, "v": v, "G": G, "Re": Re, "Pr": pr["Pr"],
        "Nu": Nu, "h": h, "f": f, "quality": -1.0, "phase": "single",
        "correlation": "Dittus-Boelter",
    }


def _two_phase_cold_transport(state, Dh, A_flow, h_hot, R_wall, geo):
    """Cold-side boiling h from the selected correlation."""
    H = state.get("H")
    if H is None:
        return _single_phase_transport(state, Dh, A_flow, mode="heating")

    x_raw = phase_quality_PH(state["P"], H, state["fluid"])
    if not (0.0 < x_raw < 1.0):
        return _single_phase_transport(state, Dh, A_flow, mode="heating")

    # Initial heat-flux estimate using saturated-liquid single-phase h.
    sp = saturated_water_props(state["P"], state["fluid"])
    G = state["mdot"] / A_flow
    h_liq = _single_phase_transport(
        {"T": sp.T_sat - 0.01, "P": state["P"], "mdot": state["mdot"], "fluid": state["fluid"]},
        Dh, A_flow, mode="heating",
    )["h"]
    U_guess = 1.0 / (1.0 / h_hot + R_wall + 1.0 / h_liq)
    dT = max(state.get("T_hot_ref", sp.T_sat + 1.0) - sp.T_sat, 1.0e-6)
    q_flux = max(U_guess * dT, 1.0)

    tp = two_phase_boiling_h(
        name=geo.boiling_correlation,
        x=x_raw,
        G=G,
        Dh=Dh,
        q_flux=q_flux,
        P=state["P"],
        L=state.get("L_node", Dh),
        fluid=state["fluid"],
    )

    rho_mix = tp["rho_mix"]
    mu_mix = tp["mu_mix"]
    v = state["mdot"] / (rho_mix * A_flow)
    Re = G * Dh / mu_mix
    f = friction_factor(Re)
    Nu = tp["h"] * Dh / max(sp.k_l, 1.0e-12)
    return {
        "props": {"rho": rho_mix, "mu": mu_mix, "k": sp.k_l, "cp": np.nan, "Pr": np.nan},
        "v": v, "G": G, "Re": Re, "Pr": np.nan, "Nu": Nu,
        "h": tp["h"], "f": f, "quality": tp["x"], "phase": "two-phase",
        "correlation": tp["correlation"], "T_sat": tp["T_sat"],
        "Bo": tp["Bo"], "Co": tp["Co"], "Xtt": tp["Xtt"],
        "q_flux_est": q_flux,
    }


def compute_node(state_hot, state_cold, geo: Geometry):
    """Return hot/cold h, U, Re, Nu, f, v and phase information for one node."""
    A_flow = _channel_area(geo.Dh)
    R_wall = geo.t_wall / geo.k_wall

    hot = _single_phase_transport(state_hot, geo.Dh, A_flow, mode="cooling")

    cold_state = dict(state_cold)
    cold_state["T_hot_ref"] = state_hot["T"]
    cold_state["L_node"] = state_cold.get("dx", geo.Dh)
    cold = _two_phase_cold_transport(cold_state, geo.Dh, A_flow, hot["h"], R_wall, geo)

    U = 1.0 / (1.0 / hot["h"] + R_wall + 1.0 / cold["h"])

    return {
        "h_hot": hot["h"], "h_cold": cold["h"], "U": U,
        "Re_hot": hot["Re"], "Re_cold": cold["Re"],
        "Nu_hot": hot["Nu"], "Nu_cold": cold["Nu"],
        "f_hot": hot["f"], "f_cold": cold["f"],
        "v_hot": hot["v"], "v_cold": cold["v"],
        "props_hot": hot["props"], "props_cold": cold["props"],
        "A_flow": A_flow, "R_wall": R_wall,
        "x_cold": cold.get("quality", -1.0),
        "phase_cold": cold.get("phase", "single"),
        "boiling_correlation": cold.get("correlation", "Dittus-Boelter"),
        "q_flux_est": cold.get("q_flux_est", np.nan),
    }
