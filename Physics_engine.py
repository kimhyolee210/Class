"""
Physics_engine.py  -  Water/Water boiling heat-exchanger 1D analysis [Task 2]

Cold side can become two-phase. In subcooled/superheated single-phase
regions, Dittus-Boelter is used. In 0 < quality < 1, the selected boiling
correlation in Geometry.boiling_correlation is used through correlation.py.
"""

import numpy as np

from Data_model import Geometry, props_PT, phase_quality_PH
from correlation import (
    baburajan_subcooled_boiling_h,
    bergles_rohsenow_onb_superheat,
    saturated_water_props,
    two_phase_boiling_h,
)


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

    sp = saturated_water_props(state["P"], state["fluid"])
    if H < sp.h_l:
        return _subcooled_cold_transport(state, Dh, A_flow, h_hot, R_wall)
    if H > sp.h_g:
        return _single_phase_transport(state, Dh, A_flow, mode="heating")
    x_raw = (H - sp.h_l) / max(sp.h_fg, 1.0e-12)

    # Initial heat-flux estimate using saturated-liquid single-phase h.
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
        dryout_transition_width=geo.dryout_transition_width,
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
        "x_di": tp["x_di"], "R_LL": tp["R_LL"], "dryout": tp["dryout"],
        "h_pre_dryout": tp["h_pre_dryout"], "h_post_dryout": tp["h_post_dryout"],
        "dryout_weight": tp["dryout_weight"],
        "q_flux_est": q_flux,
    }


def _subcooled_cold_transport(state, Dh, A_flow, h_hot, R_wall):
    """Apply ONB criterion and Baburajan subcooled-boiling correlation."""
    base = _single_phase_transport(state, Dh, A_flow, mode="heating")
    sp = saturated_water_props(state["P"], state["fluid"])
    G = state["mdot"] / A_flow
    T_bulk = state["T"]
    T_hot = state.get("T_hot_ref", T_bulk)
    U = 1.0 / (1.0 / h_hot + R_wall + 1.0 / base["h"])
    q_flux = max(U * (T_hot - T_bulk), 0.0)
    T_wall = T_bulk + q_flux / max(base["h"], 1.0e-12)
    delta_T_onb = bergles_rohsenow_onb_superheat(q_flux, state["P"])
    wall_superheat_sat = T_wall - sp.T_sat
    subcooled_boiling = wall_superheat_sat >= delta_T_onb and q_flux > 0.0

    if not subcooled_boiling:
        return {
            **base,
            "quality": phase_quality_PH(state["P"], state["H"], state["fluid"]),
            "phase": "subcooled-liquid",
            "T_sat": sp.T_sat,
            "T_wall_est": T_wall,
            "wall_superheat_sat": wall_superheat_sat,
            "delta_T_onb": delta_T_onb,
            "subcooled_boiling": False,
            "q_flux_est": q_flux,
        }

    delta_T_sub_in = max(sp.T_sat - state.get("T_inlet_ref", T_bulk), 1.0e-6)
    babu = None
    h = base["h"]
    for _ in range(5):
        U = 1.0 / (1.0 / h_hot + R_wall + 1.0 / h)
        q_flux = max(U * (T_hot - T_bulk), 1.0e-6)
        babu = baburajan_subcooled_boiling_h(
            G=G,
            Dh=Dh,
            q_flux=q_flux,
            delta_T_sub_in=delta_T_sub_in,
            mu_l_bulk=base["props"]["mu"],
            mu_l_wall=sp.mu_l,
            k_l=base["props"]["k"],
            Pr_l=base["props"]["Pr"],
            cp_l=base["props"]["cp"],
            h_fg=sp.h_fg,
        )
        h = babu["h"]

    T_wall = T_bulk + q_flux / max(h, 1.0e-12)
    return {
        **base,
        "h": h,
        "Nu": h * Dh / max(base["props"]["k"], 1.0e-12),
        "quality": phase_quality_PH(state["P"], state["H"], state["fluid"]),
        "phase": "subcooled-boiling",
        "correlation": "Baburajan",
        "T_sat": sp.T_sat,
        "T_wall_est": T_wall,
        "wall_superheat_sat": T_wall - sp.T_sat,
        "delta_T_onb": delta_T_onb,
        "subcooled_boiling": True,
        "Ja_star": babu["Ja_star"],
        "psi_subcooled": babu["psi"],
        "psi_subcooled_raw": babu["psi_raw"],
        "h_sp_l_subcooled": babu["h_sp_l"],
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
        "boiling_correlation": cold.get("correlation", "Dittus-Boelter"),
        "phase_cold": cold.get("phase", "single"),
        "subcooled_boiling": cold.get("subcooled_boiling", False),
        "T_sat_cold": cold.get("T_sat", np.nan),
        "T_wall_cold_est": cold.get("T_wall_est", np.nan),
        "wall_superheat_sat": cold.get("wall_superheat_sat", np.nan),
        "delta_T_onb": cold.get("delta_T_onb", np.nan),
        "Ja_star": cold.get("Ja_star", np.nan),
        "psi_subcooled": cold.get("psi_subcooled", np.nan),
        "psi_subcooled_raw": cold.get("psi_subcooled_raw", np.nan),
        "h_sp_l_subcooled": cold.get("h_sp_l_subcooled", np.nan),
        "x_di": cold.get("x_di", np.nan),
        "R_LL": cold.get("R_LL", np.nan),
        "dryout": cold.get("dryout", False),
        "h_pre_dryout": cold.get("h_pre_dryout", cold["h"]),
        "h_post_dryout": cold.get("h_post_dryout", cold["h"]),
        "dryout_weight": cold.get("dryout_weight", 0.0),
        "q_flux_est": cold.get("q_flux_est", np.nan),
    }
