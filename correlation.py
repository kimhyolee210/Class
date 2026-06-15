"""
correlation.py

Boiling heat-transfer correlations for the cold-side water channel.
Implemented correlations:
    1. Chen correlation
    2. Shah correlation
    3. Gungor & Winterton correlation
    4. Bertsch et al. correlation
    5. Kim & Mudawar correlation
    6. Zhang correlation
    7. Del Col et al. dryout-inception quality
    8. Dougall & Rohsenow post-dryout correlation

All inputs/outputs use SI units.
The formulas follow the lecture PDF pages 5-10. Some nucleate-boiling
terms require wall superheat; in the coupled heat-exchanger calculation,
wall superheat is estimated from the current heat flux and liquid single-
phase heat-transfer coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, log10, pi, sqrt

from CoolProp.CoolProp import PropsSI

G0 = 9.80665
EPS = 1.0e-12


@dataclass(frozen=True)
class SatWaterProps:
    P: float
    T_sat: float
    rho_l: float
    rho_g: float
    mu_l: float
    mu_g: float
    k_l: float
    k_g: float
    cp_l: float
    cp_g: float
    Pr_l: float
    Pr_g: float
    h_l: float
    h_g: float
    h_fg: float
    sigma: float
    p_crit: float
    molar_mass: float


def clamp_quality(x: float) -> float:
    """Limit vapor quality to the stable two-phase range."""
    return min(max(float(x), 1.0e-6), 0.999999)


def saturated_water_props(P: float, fluid: str = "Water") -> SatWaterProps:
    """Return saturated liquid/vapor properties of water at pressure P [Pa]."""
    T_sat = PropsSI("T", "P", P, "Q", 0, fluid)
    h_l = PropsSI("H", "P", P, "Q", 0, fluid)
    h_g = PropsSI("H", "P", P, "Q", 1, fluid)
    return SatWaterProps(
        P=P,
        T_sat=T_sat,
        rho_l=PropsSI("D", "P", P, "Q", 0, fluid),
        rho_g=PropsSI("D", "P", P, "Q", 1, fluid),
        mu_l=PropsSI("V", "P", P, "Q", 0, fluid),
        mu_g=PropsSI("V", "P", P, "Q", 1, fluid),
        k_l=PropsSI("L", "P", P, "Q", 0, fluid),
        k_g=PropsSI("L", "P", P, "Q", 1, fluid),
        cp_l=PropsSI("C", "P", P, "Q", 0, fluid),
        cp_g=PropsSI("C", "P", P, "Q", 1, fluid),
        Pr_l=PropsSI("Prandtl", "P", P, "Q", 0, fluid),
        Pr_g=PropsSI("Prandtl", "P", P, "Q", 1, fluid),
        h_l=h_l,
        h_g=h_g,
        h_fg=h_g - h_l,
        sigma=PropsSI("I", "P", P, "Q", 0, fluid),
        p_crit=PropsSI("PCRIT", fluid),
        molar_mass=PropsSI("M", fluid),
    )


def quality_from_PH(P: float, H: float, fluid: str = "Water") -> float:
    """Return thermodynamic vapor quality from pressure and enthalpy."""
    try:
        return PropsSI("Q", "P", P, "H", H, fluid)
    except ValueError:
        sp = saturated_water_props(P, fluid)
        return (H - sp.h_l) / max(sp.h_fg, EPS)


def dittus_boelter_h(G: float, Dh: float, mu: float, k: float, Pr: float, n: float = 0.4) -> float:
    """Single-phase Dittus-Boelter heat-transfer coefficient."""
    Re = max(G * Dh / max(mu, EPS), EPS)
    Nu = 4.36 if Re < 2300.0 else 0.023 * Re**0.8 * Pr**n
    return Nu * k / Dh


def hausen_h(G: float, Dh: float, L: float, mu: float, k: float, Pr: float) -> float:
    """Hausen correlation used in Bertsch et al. slide."""
    Re = max(G * Dh / max(mu, EPS), EPS)
    Graetz = max((Dh / max(L, Dh)) * Re * Pr, EPS)
    Nu = 3.66 + (0.0668 * Graetz) / (1.0 + 0.04 * Graetz ** (2.0 / 3.0))
    return Nu * k / Dh


def lockhart_martinelli_x_tt(x: float, sp: SatWaterProps) -> float:
    """Turbulent-turbulent Lockhart-Martinelli parameter X_tt."""
    x = clamp_quality(x)
    return ((1.0 - x) / x) ** 0.9 * (sp.rho_g / sp.rho_l) ** 0.5 * (sp.mu_l / sp.mu_g) ** 0.1


def convection_number(x: float, sp: SatWaterProps) -> float:
    """Shah convection number Co."""
    x = clamp_quality(x)
    return ((1.0 - x) / x) ** 0.8 * (sp.rho_g / sp.rho_l) ** 0.5


def boiling_number(q_flux: float, G: float, sp: SatWaterProps) -> float:
    """Boiling number Bo = q''/(G h_fg)."""
    return abs(q_flux) / max(G * sp.h_fg, EPS)


def del_col_dryout_quality(G: float, Dh: float, q_flux: float, P: float, sp: SatWaterProps) -> tuple[float, float]:
    """Del Col et al. (2010) dryout-inception quality x_di and R_LL.

    The expression is evaluated with SI inputs exactly as shown in the
    supplied correlation. The returned x_di is limited to [0, 1].
    """
    Bo = max(boiling_number(q_flux, G, sp), EPS)
    p_r = min(max(P / max(sp.p_crit, EPS), 0.0), 0.999999)
    R_LL = (
        0.437
        * (sp.rho_g / max(sp.rho_l, EPS)) ** 0.073
        * (sp.rho_l * sp.sigma / max(G**2, EPS)) ** 0.24
        * Dh**0.72
        / Bo
    ) ** (1.0 / 0.96)
    x_di = (
        0.4695
        * (4.0 * abs(q_flux) * R_LL / max(G * Dh * sp.h_fg, EPS)) ** 1.472
        * (G**2 * Dh / max(sp.rho_l * sp.sigma, EPS)) ** 0.3024
        * (Dh / 0.001) ** 0.1836
        * (1.0 - p_r) ** 1.239
    )
    return min(max(float(x_di), 0.0), 1.0), float(R_LL)


def dougall_rohsenow_correlation(x: float, G: float, Dh: float, sp: SatWaterProps) -> float:
    """Dougall & Rohsenow (1963) post-dryout heat-transfer coefficient."""
    x = clamp_quality(x)
    two_phase_multiplier = x + (sp.rho_g / max(sp.rho_l, EPS)) * (1.0 - x)
    Re_equivalent = G * Dh / max(sp.mu_g, EPS) * two_phase_multiplier
    Nu = 0.023 * max(Re_equivalent, EPS) ** 0.8 * sp.Pr_g**0.4
    return Nu * sp.k_g / Dh


def bergles_rohsenow_onb_superheat(q_flux: float, P: float) -> float:
    """Bergles & Rohsenow (1964) ONB wall superheat [K].

    P is converted from Pa to bar as required by the supplied correlation;
    q_flux remains in W/m2.
    """
    P_bar = max(P / 1.0e5, EPS)
    n = 0.463 * P_bar**0.0234
    return 0.556 * (max(abs(q_flux), EPS) / (1082.0 * P_bar**1.156)) ** n


def baburajan_subcooled_boiling_h(
    G: float,
    Dh: float,
    q_flux: float,
    delta_T_sub_in: float,
    mu_l_bulk: float,
    mu_l_wall: float,
    k_l: float,
    Pr_l: float,
    cp_l: float,
    h_fg: float,
) -> dict:
    """Baburajan et al. (2013) subcooled-boiling heat-transfer coefficient."""
    Re_l = max(G * Dh / max(mu_l_bulk, EPS), EPS)
    h_sp_l = (
        0.023
        * Re_l**0.8
        * max(Pr_l, EPS) ** 0.4
        * (mu_l_bulk / max(mu_l_wall, EPS)) ** 0.262
        * k_l
        / Dh
    )
    Bo = max(abs(q_flux) / max(G * h_fg, EPS), EPS)
    Ja_star = max(cp_l * max(delta_T_sub_in, EPS) / max(h_fg, EPS), EPS)
    psi_raw = 267.0 * Bo**0.86 * Ja_star**-0.6 * max(Pr_l, EPS) ** 0.23
    psi = max(1.0, psi_raw)
    return {
        "h": max(psi * h_sp_l, EPS),
        "h_sp_l": h_sp_l,
        "psi": psi,
        "psi_raw": psi_raw,
        "Bo": Bo,
        "Ja_star": Ja_star,
        "Re_l": Re_l,
    }


def liquid_froude(G: float, Dh: float, sp: SatWaterProps) -> float:
    """Liquid Froude number used by Shah correlation."""
    return G**2 / max(sp.rho_l**2 * G0 * Dh, EPS)


def cooper_pool_boiling_h(q_flux: float, P: float, sp: SatWaterProps) -> float:
    """Cooper pool-boiling heat-transfer coefficient."""
    p_r = min(max(P / max(sp.p_crit, EPS), 1.0e-8), 0.999999)
    M = max(sp.molar_mass * 1000.0, EPS)  # kg/mol -> g/mol
    return 55.0 * p_r ** (0.12 - 0.2 * log10(p_r)) * (-log10(p_r)) ** -0.55 * M ** -0.5 * abs(q_flux) ** 0.67


def forster_zuber_h(q_flux: float, P: float, sp: SatWaterProps, h_sp_l: float) -> float:
    """Forster-Zuber nucleate-boiling term used by Chen/Zhang."""
    dT = max(abs(q_flux) / max(h_sp_l, EPS), 1.0e-6)
    try:
        p_sat_wall = PropsSI("P", "T", min(sp.T_sat + dT, 646.0), "Q", 0, "Water")
        dP_sat = max(p_sat_wall - P, 1.0)
    except ValueError:
        dP_sat = max(P * 1.0e-4, 1.0)
    return (
        0.00122
        * sp.k_l**0.79
        * sp.cp_l**0.45
        * sp.rho_l**0.49
        * G0**0.25
        / max(sp.sigma**0.5 * sp.mu_l**0.29 * sp.h_fg**0.24 * sp.rho_g**0.24, EPS)
        * dT**0.24
        * dP_sat**0.75
    )


def chen_correlation(x: float, G: float, Dh: float, q_flux: float, P: float, h_sp_l: float, sp: SatWaterProps) -> float:
    """Chen correlation: h_tp = F h_DB + S h_FZ."""
    x = clamp_quality(x)
    Re_l = max(G * (1.0 - x) * Dh / max(sp.mu_l, EPS), EPS)
    Re_lo = max(G * Dh / max(sp.mu_l, EPS), EPS)
    F = max((Re_lo / Re_l) ** 0.8, 1.0)
    h_mac = F * h_sp_l
    h_mic = forster_zuber_h(q_flux, P, sp, h_sp_l)
    S = 1.0 / (1.0 + 2.53e-6 * Re_l**1.17 * F**1.25)
    return h_mac + S * h_mic


def shah_correlation(x: float, G: float, Dh: float, q_flux: float, h_sp_l: float, sp: SatWaterProps) -> float:
    """Shah correlation: h_tp = psi h_sp."""
    x = clamp_quality(x)
    Bo = boiling_number(q_flux, G, sp)
    Co = convection_number(x, sp)
    Fr_l = liquid_froude(G, Dh, sp)
    N = Co if Fr_l >= 0.04 else 0.38 * Fr_l ** -0.3 * Co

    psi_nb = 230.0 * Bo**0.5 if Bo > 0.3e-4 else 1.0 + 46.0 * Bo**0.5
    psi_cb = 1.8 / max(N**0.8, EPS)
    if 0.1 < N <= 1.0:
        F = 14.7 if Bo >= 11.0e-4 else 15.43
        psi_bs = F * Bo**0.5 * exp(2.74 * N**-0.1)
    else:
        F = 14.7 if Bo >= 11.0e-4 else 15.43
        psi_bs = F * Bo**0.5 * exp(2.74 * N**-0.15)
    return max(psi_nb, psi_cb, psi_bs) * h_sp_l


def gungor_winterton_correlation(x: float, G: float, Dh: float, q_flux: float, P: float, h_sp_l: float, sp: SatWaterProps) -> float:
    """Gungor & Winterton correlation: h_tp = E h_sp + S h_pool."""
    x = clamp_quality(x)
    Bo = boiling_number(q_flux, G, sp)
    Xtt = max(lockhart_martinelli_x_tt(x, sp), EPS)
    Re_l = max(G * (1.0 - x) * Dh / max(sp.mu_l, EPS), EPS)
    E = 1.0 + 24000.0 * Bo**1.16 + 1.37 * (1.0 / Xtt) ** 0.86
    S = 1.0 / (1.0 + 1.15e-6 * E**2 * Re_l**1.17)
    h_pool = cooper_pool_boiling_h(q_flux, P, sp)
    return E * h_sp_l + S * h_pool


def bertsch_correlation(x: float, G: float, Dh: float, q_flux: float, P: float, L: float, sp: SatWaterProps) -> float:
    """Bertsch et al. correlation: h_tp = E h_cb + S h_nb."""
    x = clamp_quality(x)
    h_sp_lo = hausen_h(G, Dh, L, sp.mu_l, sp.k_l, sp.Pr_l)
    h_sp_go = hausen_h(G, Dh, L, sp.mu_g, sp.k_g, sp.Pr_g)
    h_cb = h_sp_lo * (1.0 - x) + h_sp_go * x
    h_nb = cooper_pool_boiling_h(q_flux, P, sp)
    Co = sqrt(sp.sigma / max(G0 * (sp.rho_l - sp.rho_g) * Dh**2, EPS))
    E = 1.0 + 80.0 * (x**2 - x**6) * exp(-0.6 * Co)
    S = 1.0 - x
    return E * h_cb + S * h_nb


def kim_mudawar_correlation(x: float, G: float, Dh: float, q_flux: float, P: float, h_DB: float, sp: SatWaterProps, heated_perimeter: float | None = None, wetted_perimeter: float | None = None) -> float:
    """Kim & Mudawar correlation."""
    x = clamp_quality(x)
    Bo = boiling_number(q_flux, G, sp)
    PR = min(max(P / max(sp.p_crit, EPS), 1.0e-8), 0.999999)
    We_fo = G**2 * Dh / max(sp.rho_l * sp.sigma, EPS)
    Xtt = max(lockhart_martinelli_x_tt(x, sp), EPS)
    PH = heated_perimeter if heated_perimeter is not None else pi * Dh
    PF = wetted_perimeter if wetted_perimeter is not None else pi * Dh
    perimeter_ratio = max(PH / max(PF, EPS), EPS)
    h_cb = 2345.0 * (Bo * perimeter_ratio) ** 0.7 * PR**0.38 * (1.0 - x) ** -0.51 * h_DB
    h_nb = (
        5.2 * (Bo * perimeter_ratio) ** 0.08 * We_fo**-0.54
        + 3.5 * (1.0 / Xtt) ** 0.94 * (sp.rho_g / sp.rho_l) ** 0.25
    ) * h_DB
    return sqrt(h_nb**2 + h_cb**2)


def zhang_correlation(x: float, G: float, Dh: float, q_flux: float, P: float, h_sp_v: float, h_sp_l: float, sp: SatWaterProps, C: float = 20.0) -> float:
    """Zhang correlation: h = h_pb + xi phi_f h_sp,v."""
    x = clamp_quality(x)
    h_pb = forster_zuber_h(q_flux, P, sp, h_sp_l)
    Xtt = max(lockhart_martinelli_x_tt(x, sp), EPS)
    phi_f = 1.0 + C / Xtt + 1.0 / Xtt**2
    xi = 0.64
    return h_pb + xi * phi_f * h_sp_v


def two_phase_boiling_h(
    name: str,
    x: float,
    G: float,
    Dh: float,
    q_flux: float,
    P: float,
    L: float,
    fluid: str = "Water",
    heated_perimeter: float | None = None,
    wetted_perimeter: float | None = None,
    dryout_transition_width: float = 0.06,
) -> dict:
    """Dispatch one boiling correlation and return h plus useful intermediate data."""
    sp = saturated_water_props(P, fluid)
    x = clamp_quality(x)
    h_sp_l = dittus_boelter_h(G, Dh, sp.mu_l, sp.k_l, sp.Pr_l, n=0.4)
    h_sp_v = dittus_boelter_h(G, Dh, sp.mu_g, sp.k_g, sp.Pr_g, n=0.4)
    key = name.lower().replace("&", "and").replace(" ", "_").replace("-", "_")
    x_di, R_LL = del_col_dryout_quality(G, Dh, q_flux, P, sp)
    x_pre_eval = min(x, x_di)

    if key in {"chen", "chen_correlation"}:
        h_pre_dryout = chen_correlation(x_pre_eval, G, Dh, q_flux, P, h_sp_l, sp)
    elif key in {"shah", "shah_correlation"}:
        h_pre_dryout = shah_correlation(x_pre_eval, G, Dh, q_flux, h_sp_l, sp)
    elif key in {"gungor_winterton", "gungor_and_winterton", "gungor_winterton_correlation"}:
        h_pre_dryout = gungor_winterton_correlation(x_pre_eval, G, Dh, q_flux, P, h_sp_l, sp)
    elif key in {"bertsch", "bertsch_et_al", "bertsch_correlation"}:
        h_pre_dryout = bertsch_correlation(x_pre_eval, G, Dh, q_flux, P, L, sp)
    elif key in {"kim_mudawar", "kim_and_mudawar", "kim_mudawar_correlation"}:
        h_pre_dryout = kim_mudawar_correlation(x_pre_eval, G, Dh, q_flux, P, h_sp_l, sp, heated_perimeter, wetted_perimeter)
    elif key in {"zhang", "zhang_correlation"}:
        h_pre_dryout = zhang_correlation(x_pre_eval, G, Dh, q_flux, P, h_sp_v, h_sp_l, sp)
    else:
        raise ValueError(f"Unknown boiling correlation: {name}")

    h_post_dryout = dougall_rohsenow_correlation(x, G, Dh, sp)
    width = max(float(dryout_transition_width), 1.0e-6)
    transition_coordinate = min(max((x - (x_di - 0.5 * width)) / width, 0.0), 1.0)
    dryout_weight = transition_coordinate**2 * (3.0 - 2.0 * transition_coordinate)
    dryout = x >= x_di
    if dryout_weight >= 1.0:
        h = h_post_dryout
        active_correlation = "Dougall-Rohsenow"
    elif dryout_weight <= 0.0:
        h = h_pre_dryout
        active_correlation = name
    else:
        h = exp(
            (1.0 - dryout_weight) * log(max(h_pre_dryout, EPS))
            + dryout_weight * log(max(h_post_dryout, EPS))
        )
        active_correlation = "Dryout transition"

    return {
        "h": max(float(h), EPS),
        "h_pre_dryout": max(float(h_pre_dryout), EPS),
        "h_post_dryout": max(float(h_post_dryout), EPS),
        "dryout_weight": dryout_weight,
        "x": x,
        "x_di": x_di,
        "R_LL": R_LL,
        "dryout": dryout,
        "T_sat": sp.T_sat,
        "Bo": boiling_number(q_flux, G, sp),
        "Co": convection_number(x, sp),
        "Xtt": lockhart_martinelli_x_tt(x, sp),
        "rho_mix": 1.0 / (x / sp.rho_g + (1.0 - x) / sp.rho_l),
        "mu_mix": 1.0 / (x / sp.mu_g + (1.0 - x) / sp.mu_l),
        "h_sp_l": h_sp_l,
        "h_sp_v": h_sp_v,
        "correlation": active_correlation,
    }
