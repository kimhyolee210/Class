"""
correlation.py
Boiling heat-transfer correlations used by Physics_engine.py.

q_flux is NOT hard-coded. It must be supplied from the preliminary
single-phase heat-flux estimate q'' = U_single_phase * DeltaT.
"""

import math

G = 9.80665
M_WATER = 18.01528  # kg/kmol-scale value used only in Cooper-style term


def _clip_x(x):
    return min(max(x, 1e-6), 1.0 - 1e-6)


def _positive(v, eps=1e-12):
    return max(abs(v), eps)


def x_tt(x, rho_l, rho_g, mu_l, mu_g):
    x = _clip_x(x)
    return ((1.0 - x) / x) ** 0.9 * (rho_g / rho_l) ** 0.5 * (mu_l / mu_g) ** 0.1


def boiling_number(q_flux, G_mass, h_fg):
    return _positive(q_flux) / (_positive(G_mass) * _positive(h_fg))


def confinement_number(sigma, rho_l, rho_g, D_h):
    return math.sqrt(_positive(sigma) / (_positive(G) * _positive(rho_l - rho_g) * _positive(D_h) ** 2))


def cooper_pool_boiling(q_flux, P_r, M=M_WATER):
    """Cooper pool-boiling form used in several slides."""
    P_r = min(max(P_r, 1e-6), 0.999999)
    return 55.0 * (P_r ** (0.12 - 0.2 * math.log10(P_r))) * ((-math.log10(P_r)) ** -0.55) * (M ** -0.5) * (_positive(q_flux) ** 0.67)


def chen_correlation(h_sp, h_pool, Re_tp, Re_l, S=1.0):
    """Chen: h_tp = h_mac + h_mic = F*h_sp + S*h_pool."""
    F = (_positive(Re_tp) / _positive(Re_l)) ** 0.8
    return F * h_sp + S * h_pool


def shah_correlation(h_sp, x, q_flux, G_mass, h_fg, rho_l, rho_g, D_h):
    x = _clip_x(x)
    Co = ((1.0 - x) / x) ** 0.8 * (rho_g / rho_l) ** 0.5
    Bo = boiling_number(q_flux, G_mass, h_fg)
    Fr_l = G_mass ** 2 / (_positive(rho_l) ** 2 * G * _positive(D_h))
    N = Co if Fr_l >= 0.04 else 0.38 * (Fr_l ** -0.3) * Co
    psi_nb = max(230.0 * math.sqrt(Bo), 1.0 + 46.0 * math.sqrt(Bo))
    psi_cb = 1.8 / (_positive(N) ** 0.8)
    F = 14.7 if Bo >= 11e-4 else 15.43
    if 0.1 < N <= 1.0:
        psi_bs = F * math.sqrt(Bo) * math.exp(2.74 * (N ** -0.1))
    else:
        psi_bs = F * math.sqrt(Bo) * math.exp(2.74 * (_positive(N) ** -0.15))
    psi = max(psi_nb, psi_cb, psi_bs)
    return psi * h_sp


def gungor_winterton_correlation(h_sp, h_pool, x, q_flux, G_mass, h_fg, rho_l, rho_g, mu_l, mu_g, Re_l):
    Xtt = x_tt(x, rho_l, rho_g, mu_l, mu_g)
    Bo = boiling_number(q_flux, G_mass, h_fg)
    E = 1.0 + 24000.0 * (Bo ** 1.16) + 1.37 * (1.0 / _positive(Xtt)) ** 0.86
    S = 1.0 / (1.0 + 1.15e-6 * (E ** 2) * (_positive(Re_l) ** 1.17))
    return E * h_sp + S * h_pool


def bertsch_correlation(h_l, h_g, h_pool, x, sigma, rho_l, rho_g, D_h):
    x = _clip_x(x)
    h_cb = h_l * (1.0 - x) + h_g * x
    Co = confinement_number(sigma, rho_l, rho_g, D_h)
    E = 1.0 + 80.0 * (x ** 2 - x ** 6) * math.exp(-0.6 * Co)
    S = 1.0 - x
    return E * h_cb + S * h_pool


def kim_mudawar_correlation(h_db, x, q_flux, G_mass, h_fg, P, P_crit, rho_l, rho_g, mu_l, mu_g, sigma, D_h, P_H, P_F):
    x = _clip_x(x)
    Bo = boiling_number(q_flux, G_mass, h_fg)
    P_R = min(max(P / _positive(P_crit), 1e-6), 0.999999)
    We_fo = G_mass ** 2 * D_h / (_positive(rho_l) * _positive(sigma))
    Xtt = x_tt(x, rho_l, rho_g, mu_l, mu_g)
    perimeter_ratio = _positive(P_H) / _positive(P_F)
    h_cb = 2345.0 * (Bo * perimeter_ratio) ** 0.7 * (P_R ** 0.38) * ((1.0 - x) ** -0.51) * h_db
    h_nb = (5.2 * (Bo * perimeter_ratio) ** 0.08 * (_positive(We_fo) ** -0.54) +
            3.5 * (1.0 / _positive(Xtt)) ** 0.94 * (rho_g / rho_l) ** 0.25) * h_db
    return math.sqrt(h_nb ** 2 + h_cb ** 2)


def zhang_correlation(h_sp_v, h_pool, X, xi=0.64, C=20.0):
    phi_f = 1.0 + C / _positive(X) + 1.0 / (_positive(X) ** 2)
    return h_pool + xi * phi_f * h_sp_v
