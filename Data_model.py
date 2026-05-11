"""
[Task 1] Data_model.py
Counter-current heat exchanger data model.

Important boundary condition:
    HOT inlet  : x = 0
    COLD inlet : x = L

The cold-side target is set to saturated vapor temperature only:
    T_cold_out,target = T_sat(P_cold) + 0 K

This version removes the 20 K superheat target because the previous case
was not reachable under the given quick-test conditions.

All property calls are wrapped here. Other files should use these wrappers.
"""

try:
    from CoolProp.CoolProp import PropsSI
except ModuleNotFoundError as e:
    PropsSI = None
    _COOLPROP_IMPORT_ERROR = e


def require_coolprop():
    if PropsSI is None:
        raise ImportError("CoolProp is required. Install with: pip install CoolProp") from _COOLPROP_IMPORT_ERROR


FLUID = "Water"
SUPERHEAT_MARGIN_K = 0.0

# Saturation temperatures are evaluated with CoolProp when available.
# If CoolProp is not available during import, values are filled later when functions are called.

def sat_T(P, fluid=FLUID):
    require_coolprop()
    return PropsSI("T", "P", P, "Q", 0, fluid)


def sat_props(P, fluid=FLUID):
    """Saturated liquid/vapor properties at pressure P."""
    require_coolprop()
    return {
        "T_sat": PropsSI("T", "P", P, "Q", 0, fluid),
        "H_l": PropsSI("H", "P", P, "Q", 0, fluid),
        "H_g": PropsSI("H", "P", P, "Q", 1, fluid),
        "rho_l": PropsSI("D", "P", P, "Q", 0, fluid),
        "rho_g": PropsSI("D", "P", P, "Q", 1, fluid),
        "Cp_l": PropsSI("C", "P", P, "Q", 0, fluid),
        "Cp_g": PropsSI("C", "P", P, "Q", 1, fluid),
        "k_l": PropsSI("L", "P", P, "Q", 0, fluid),
        "k_g": PropsSI("L", "P", P, "Q", 1, fluid),
        "mu_l": PropsSI("V", "P", P, "Q", 0, fluid),
        "mu_g": PropsSI("V", "P", P, "Q", 1, fluid),
        "sigma": PropsSI("I", "P", P, "Q", 0, fluid),
        "P_crit": PropsSI("PCRIT", fluid),
    }


P_HOT_IN = 15.0e6
P_COLD_IN = 6.0e6

# Hot inlet is kept as the user-specified condition, not Tsat-based.
# Original changed condition: Hot side = 15 MPa, 600 K.
def hot_inlet_temperature():
    return 600.0


def cold_outlet_target_temperature():
    return sat_T(P_COLD_IN) + SUPERHEAT_MARGIN_K


def get_fixed_conditions():
    require_coolprop()
    return {
        "geometry": {
            "D_h": 2.0e-3,
            "t_wall": 1.0e-3,
            "k_wall": 20.0,
        },
        "hot_inlet": {
            "fluid": FLUID,
            "P_in": P_HOT_IN,
            "T_in": hot_inlet_temperature(),
            "m_dot": 1.5,
        },
        "cold_inlet": {
            "fluid": FLUID,
            "P_in": P_COLD_IN,
            "T_in": 530.0,
            "m_dot": 5.0,
        },
        "target": {
            "T_cold_out": cold_outlet_target_temperature(),
            "superheat_margin_K": SUPERHEAT_MARGIN_K,
        },
        "model": {
            "correlation_model": "chen",  # single_phase, chen, shah, gungor_winterton, bertsch, kim_mudawar, zhang
        },
    }


def get_state(fluid, P, T):
    require_coolprop()
    return {
        "rho": PropsSI("D", "P", P, "T", T, fluid),
        "Cp": PropsSI("C", "P", P, "T", T, fluid),
        "k": PropsSI("L", "P", P, "T", T, fluid),
        "mu": PropsSI("V", "P", P, "T", T, fluid),
        "H": PropsSI("H", "P", P, "T", T, fluid),
    }


def get_state_PH(fluid, P, H):
    """
    Phase-safe property call using P-H instead of P-T.
    This avoids CoolProp errors exactly at saturation when P and T are nearly saturated.
    Use only for single-phase states. Two-phase states should use sat_props() + quality.
    """
    require_coolprop()
    return {
        "T": PropsSI("T", "P", P, "H", H, fluid),
        "rho": PropsSI("D", "P", P, "H", H, fluid),
        "Cp": PropsSI("C", "P", P, "H", H, fluid),
        "k": PropsSI("L", "P", P, "H", H, fluid),
        "mu": PropsSI("V", "P", P, "H", H, fluid),
        "H": H,
    }


def T_from_PH(fluid, P, H):
    require_coolprop()
    return PropsSI("T", "P", P, "H", H, fluid)


def H_from_PT(fluid, P, T):
    require_coolprop()
    if P <= 0:
        raise ValueError(f"Non-physical pressure was given to H_from_PT: P={P} Pa")
    return PropsSI("H", "P", P, "T", T, fluid)


def phase_from_PH(fluid, P, H):
    """
    Return phase name and vapor quality x.

    A small enthalpy tolerance is applied around saturated liquid/vapor values.
    Without this, CoolProp can fail when a marching point lands almost exactly
    on the saturation line and another function later tries PropsSI(P,T).
    """
    s = sat_props(P, fluid)
    h_fg = max(s["H_g"] - s["H_l"], 1e-12)
    tol_H = 1.0e-7 * h_fg

    if H < s["H_l"] - tol_H:
        return "subcooled_liquid", 0.0
    if H > s["H_g"] + tol_H:
        return "superheated_vapor", 1.0

    x = (H - s["H_l"]) / h_fg
    return "two_phase", min(max(x, 0.0), 1.0)


if __name__ == "__main__":
    c = get_fixed_conditions()
    print("[Task 1] Data model")
    print(f"Hot inlet  : P={c['hot_inlet']['P_in']/1e6:.2f} MPa, T={c['hot_inlet']['T_in']:.2f} K")
    print(f"Cold inlet : P={c['cold_inlet']['P_in']/1e6:.2f} MPa, T={c['cold_inlet']['T_in']:.2f} K")
    print(f"Cold target: T={c['target']['T_cold_out']:.2f} K = Tsat + {SUPERHEAT_MARGIN_K:.1f} K")
