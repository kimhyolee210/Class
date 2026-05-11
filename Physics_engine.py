"""
[Task 2] Physics_engine.py
Phase-aware heat-transfer calculation.

- Single-phase liquid/vapor: Dittus-Boelter based h
- Two-phase heating: selected boiling correlation from correlation.py
- q'' is estimated outside as q_flux_sp = U_single_phase * DeltaT and passed in.
"""

import math
from Data_model import get_state, get_state_PH, T_from_PH, sat_props, phase_from_PH
import correlation as corr


def velocity(m_dot, rho, A_flow):
    return m_dot / (rho * A_flow)


def reynolds(rho, V, D_h, mu):
    return rho * V * D_h / mu


def prandtl(Cp, mu, k):
    return mu * Cp / k


def nusselt_dittus_boelter(Re, Pr, mode="heating"):
    if Re < 2300:
        return 4.36
    if mode == "cooling":
        return 0.0265 * Re ** 0.8 * Pr ** 0.3
    return 0.0243 * Re ** 0.8 * Pr ** 0.4


def htc(Nu, k, D_h):
    return Nu * k / D_h


def friction_factor(Re):
    if Re < 2300:
        return 64.0 / max(Re, 1e-12)
    return 0.25 / (math.log10(5.74 / (Re ** 0.9))) ** 2


def overall_U(h_hot, h_cold, t_wall, k_wall):
    return 1.0 / (1.0 / h_hot + t_wall / k_wall + 1.0 / h_cold)


def single_phase_node_from_PT(fluid, P, T, m_dot, A_flow, D_h, mode="heating"):
    state = get_state(fluid, P, T)
    V = velocity(m_dot, state["rho"], A_flow)
    Re = reynolds(state["rho"], V, D_h, state["mu"])
    Pr = prandtl(state["Cp"], state["mu"], state["k"])
    Nu = nusselt_dittus_boelter(Re, Pr, mode=mode)
    h = htc(Nu, state["k"], D_h)
    f = friction_factor(Re)
    return {
        "fluid": fluid, "P": P, "T": T, "H": state["H"], "phase": "single_phase",
        "rho": state["rho"], "Cp": state["Cp"], "k": state["k"], "mu": state["mu"],
        "V": V, "Re": Re, "Pr": Pr, "Nu": Nu, "h": h, "f": f, "x": None,
    }




def single_phase_node_from_PH(fluid, P, H, m_dot, A_flow, D_h, mode="heating", phase_label="single_phase", x_value=None):
    """
    Single-phase node evaluated from P-H.
    This is safer than P-T near the saturation boundary.
    """
    state = get_state_PH(fluid, P, H)
    V = velocity(m_dot, state["rho"], A_flow)
    Re = reynolds(state["rho"], V, D_h, state["mu"])
    Pr = prandtl(state["Cp"], state["mu"], state["k"])
    Nu = nusselt_dittus_boelter(Re, Pr, mode=mode)
    h = htc(Nu, state["k"], D_h)
    f = friction_factor(Re)
    return {
        "fluid": fluid, "P": P, "T": state["T"], "H": H, "phase": phase_label,
        "rho": state["rho"], "Cp": state["Cp"], "k": state["k"], "mu": state["mu"],
        "V": V, "Re": Re, "Pr": Pr, "Nu": Nu, "h": h, "f": f, "x": x_value,
    }

def saturated_single_phase_node(fluid, P, quality, m_dot, A_flow, D_h, mode="heating"):
    s = sat_props(P, fluid)
    if quality <= 0:
        rho, Cp, k, mu, H = s["rho_l"], s["Cp_l"], s["k_l"], s["mu_l"], s["H_l"]
    else:
        rho, Cp, k, mu, H = s["rho_g"], s["Cp_g"], s["k_g"], s["mu_g"], s["H_g"]
    V = velocity(m_dot, rho, A_flow)
    Re = reynolds(rho, V, D_h, mu)
    Pr = prandtl(Cp, mu, k)
    Nu = nusselt_dittus_boelter(Re, Pr, mode=mode)
    h = htc(Nu, k, D_h)
    f = friction_factor(Re)
    return {"fluid": fluid, "P": P, "T": s["T_sat"], "H": H, "phase": "sat_liq" if quality <= 0 else "sat_vap",
            "rho": rho, "Cp": Cp, "k": k, "mu": mu, "V": V, "Re": Re, "Pr": Pr, "Nu": Nu, "h": h, "f": f, "x": quality}


def evaluate_node_from_PH(fluid, P, H, m_dot, A_flow, D_h, mode="heating", correlation_model="single_phase", q_flux_sp=None, P_H=None, P_F=None):
    phase, x = phase_from_PH(fluid, P, H)

    # For true single-phase states, evaluate properties with P-H rather than P-T.
    # This prevents CoolProp saturation-line errors when T is extremely close to Tsat.
    if phase == "subcooled_liquid" or phase == "superheated_vapor":
        return single_phase_node_from_PH(fluid, P, H, m_dot, A_flow, D_h, mode=mode, phase_label=phase, x_value=x)

    # Two-phase region. Even if correlation_model == "single_phase", do NOT call
    # single_phase_node_from_PT at Tsat. Use saturated properties and quality.
    s = sat_props(P, fluid)
    h_fg = s["H_g"] - s["H_l"]
    G_mass = m_dot / A_flow
    q_flux = abs(q_flux_sp) if q_flux_sp is not None else 0.0

    liq = saturated_single_phase_node(fluid, P, 0, m_dot, A_flow, D_h, mode=mode)
    vap = saturated_single_phase_node(fluid, P, 1, m_dot, A_flow, D_h, mode=mode)
    h_sp = liq["h"]
    h_pool = corr.cooper_pool_boiling(q_flux, P / s["P_crit"])
    rho_mix = 1.0 / ((1.0 - x) / s["rho_l"] + x / s["rho_g"])
    mu_mix = (1.0 - x) * s["mu_l"] + x * s["mu_g"]
    V_mix = velocity(m_dot, rho_mix, A_flow)
    Re_mix = reynolds(rho_mix, V_mix, D_h, mu_mix)
    f_mix = friction_factor(Re_mix)

    model = correlation_model.lower()
    if mode == "cooling" or model == "single_phase":
        # PDF correlations are boiling/heating correlations.
        # For hot-side condensation or for a preliminary single-phase estimate,
        # use a saturated-mixture fallback without any P-T call at Tsat.
        h = (1.0 - x) * liq["h"] + x * vap["h"]
    elif model == "chen":
        h = corr.chen_correlation(h_sp, h_pool, Re_mix, liq["Re"], S=1.0)
    elif model == "shah":
        h = corr.shah_correlation(h_sp, x, q_flux, G_mass, h_fg, s["rho_l"], s["rho_g"], D_h)
    elif model == "gungor_winterton":
        h = corr.gungor_winterton_correlation(h_sp, h_pool, x, q_flux, G_mass, h_fg, s["rho_l"], s["rho_g"], s["mu_l"], s["mu_g"], liq["Re"])
    elif model == "bertsch":
        h = corr.bertsch_correlation(liq["h"], vap["h"], h_pool, x, s["sigma"], s["rho_l"], s["rho_g"], D_h)
    elif model == "kim_mudawar":
        h = corr.kim_mudawar_correlation(h_sp, x, q_flux, G_mass, h_fg, P, s["P_crit"], s["rho_l"], s["rho_g"], s["mu_l"], s["mu_g"], s["sigma"], D_h, P_H or 1.0, P_F or 1.0)
    elif model == "zhang":
        X = corr.x_tt(x, s["rho_l"], s["rho_g"], s["mu_l"], s["mu_g"])
        h = corr.zhang_correlation(vap["h"], h_pool, X)
    else:
        h = h_sp

    return {"fluid": fluid, "P": P, "T": s["T_sat"], "H": H, "phase": "two_phase", "x": x,
            "rho": rho_mix, "Cp": None, "k": None, "mu": mu_mix, "V": V_mix,
            "Re": Re_mix, "Pr": None, "Nu": None, "h": h, "f": f_mix}


def compute_U_for_pair(hot_node, cold_node, t_wall, k_wall):
    return overall_U(hot_node["h"], cold_node["h"], t_wall, k_wall)
