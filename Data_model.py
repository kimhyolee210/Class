"""
Data_model.py  -  Water/Water PCHE 열교환기 1D 해석 [Task 1]

변경 조건:
    - Hot  side: Water, 15 MPa, 600 K
    - Cold side: Water,  6 MPa, 530 K
    - Hot side inlet velocity = 2 m/s 가 되도록 전체 질량유량 자동 계산
    - Cold side 질량유량은 hot side와 동일하게 사용
    - Target cold outlet temperature = 550 K
"""

from dataclasses import dataclass, replace
from math import pi

from CoolProp.CoolProp import PropsSI

from correlation import quality_from_PH, saturated_water_props

MIN_PRESSURE_PA = 1.0e5  # CoolProp 안정성 확보용 하한 압력 [Pa]
MAX_QUALITY_EPS = 1.0e-8


@dataclass(frozen=True)
class Geometry:
    """채널 형상 및 벽체 물성 (PCHE 단일 채널 기준)."""
    Dh:         float = 2.0e-3      # 수력직경 [m]
    t_wall:     float = 1.0e-3      # 벽 두께 [m]
    k_wall:     float = 20.0        # 벽 열전도도 [W/m/K]
    N_channels: int   = 5000        # 평행 채널 수 (per-side, PCHE)
    boiling_correlation: str = "Chen"  # cold side 비등 상관식 선택
    dryout_transition_width: float = 0.20  # coarse-grid x_di 주변 상관식 완화 구간 폭


@dataclass(frozen=True)
class HotInlet:
    """고온측 물 입구 조건."""
    T_in:  float = 600.0            # 입구 온도 [K]
    P_in:  float = 15.0e6           # 입구 압력 [Pa] (=15 MPa)
    mdot:  float = 0.0              # 전체 질량유량 [kg/s], make_water_inlets()에서 계산
    fluid: str   = "Water"


@dataclass(frozen=True)
class ColdInlet:
    """저온측 물 입구 조건."""
    T_in:  float = 530.0            # 입구 온도 [K]
    P_in:  float = 6.0e6            # 입구 압력 [Pa] (=6 MPa)
    mdot:  float = 0.0              # 전체 질량유량 [kg/s], hot side와 동일하게 설정
    fluid: str   = "Water"


@dataclass(frozen=True)
class Target:
    """설계 목표 - 저온측 물 출구 온도."""
    T_cold_out: float = 563.15      # 목표 출구 온도 [K] (= 290 °C)
    P_cold_out: float = 6.0e6       # 목표 출구 압력 [Pa] (= 6 MPa)
    min_hot_out_approach: float = 5.0  # hot outlet이 cold inlet보다 최소한 높게 남길 온도차 [K]
    cold_mdot_margin: float = 0.73     # pinch 여유를 포함한 cold 유량 안전계수


@dataclass(frozen=True)
class Numeric:
    """수치해석 파라미터."""
    n_nodes:  int   = 20
    tol:      float = 1.0e-3
    max_iter: int   = 20
    max_cold_dh_per_cell: float = 1.0e5  # 한 셀 cold 엔탈피 증가 상한 [J/kg]


def hot_mdot_for_velocity(geo: Geometry, hot: HotInlet, velocity: float = 2.0):
    """Hot side 물의 입구 유속이 velocity[m/s]가 되도록 전체 질량유량을 계산."""
    rho = PropsSI("D", "P", hot.P_in, "T", hot.T_in, hot.fluid)
    area_one_channel = 0.25 * pi * geo.Dh ** 2
    return rho * velocity * area_one_channel * geo.N_channels


def feasible_cold_mdot_ratio(hot: HotInlet, cold: ColdInlet, target: Target):
    """현재 hot 조건으로 target cold outlet까지 가열 가능한 cold/hot 유량비."""
    H_hot_in = H_from_PT(hot.P_in, hot.T_in, hot.fluid)
    T_hot_min = cold.T_in + target.min_hot_out_approach
    H_hot_min = H_from_PT(hot.P_in, T_hot_min, hot.fluid)
    H_cold_in = H_from_PT(cold.P_in, cold.T_in, cold.fluid)
    H_cold_target = H_from_PT(target.P_cold_out, target.T_cold_out, cold.fluid)

    q_hot_available = max(H_hot_in - H_hot_min, 0.0)
    q_cold_required = max(H_cold_target - H_cold_in, 1.0e-12)
    return q_hot_available / q_cold_required


def make_water_inlets(geo: Geometry, hot_velocity: float = 2.0,
                      target: Target | None = None,
                      cold_mdot_ratio: float | None = None):
    """고온측 2 m/s 기준 질량유량을 계산하고 cold 유량을 설정."""
    hot0 = HotInlet()
    mdot = hot_mdot_for_velocity(geo, hot0, hot_velocity)
    hot = replace(hot0, mdot=mdot)
    cold0 = ColdInlet()

    if cold_mdot_ratio is None and target is not None:
        cold_mdot_ratio = min(
            1.0,
            target.cold_mdot_margin * feasible_cold_mdot_ratio(hot, cold0, target),
        )
    if cold_mdot_ratio is None:
        cold_mdot_ratio = 1.0

    cold = replace(cold0, P_in=target.P_cold_out if target is not None else cold0.P_in,
                   mdot=mdot * cold_mdot_ratio)
    return hot, cold


def _props_safe(output, P, T, fluid):
    P = max(float(P), MIN_PRESSURE_PA)
    """CoolProp P,T property call with saturation-line fallback for Water.

    Water at exactly the saturation line often raises a CoolProp error for
    P,T inputs.  For transport-property evaluation, use the saturated-liquid
    side when T <= Tsat and the saturated-vapor side when T > Tsat.
    """
    try:
        return PropsSI(output, "P", P, "T", T, fluid)
    except ValueError:
        if fluid.lower() != "water":
            raise
        T_sat = PropsSI("T", "P", P, "Q", 0, fluid)
        Q = 0 if T <= T_sat else 1
        return PropsSI(output, "P", P, "Q", Q, fluid)


def props_PT(P, T, fluid):
    """(P[Pa], T[K], fluid name) -> 물성치 dict."""
    P = max(float(P), MIN_PRESSURE_PA)
    return {
        "rho": _props_safe("D",       P, T, fluid),
        "mu":  _props_safe("V",       P, T, fluid),
        "k":   _props_safe("L",       P, T, fluid),
        "cp":  _props_safe("C",       P, T, fluid),
        "Pr":  _props_safe("Prandtl", P, T, fluid),
    }


def T_from_PH(P, H, fluid):
    P = max(float(P), MIN_PRESSURE_PA)
    """(P[Pa], H[J/kg], fluid name) -> T[K].

    Water의 포화영역에서는 P,H 조합이 두 상의 혼합 상태를 의미하므로
    온도는 포화온도로 고정된다. 포화액보다 낮은 엔탈피는 정상적인
    subcooled liquid 상태일 수 있으므로 CoolProp P-H flash에 그대로 맡긴다.
    """
    if fluid.lower() == "water":
        try:
            H_f = PropsSI("H", "P", P, "Q", 0, fluid)
            H_g = PropsSI("H", "P", P, "Q", 1, fluid)
            T_sat = PropsSI("T", "P", P, "Q", 0, fluid)
            if H_f <= H <= H_g:
                return T_sat
        except ValueError:
            pass
    return PropsSI("T", "P", P, "H", H, fluid)


def phase_quality_PH(P, H, fluid="Water"):
    P = max(float(P), MIN_PRESSURE_PA)
    """(P,H) 기준 건도 Q를 반환. 단상 영역에서는 Q<0 또는 Q>1일 수 있음."""
    return quality_from_PH(P, H, fluid)


def Tsat_from_P(P, fluid="Water"):
    P = max(float(P), MIN_PRESSURE_PA)
    """압력 기준 포화온도 [K]."""
    return saturated_water_props(P, fluid).T_sat


def H_from_PT(P, T, fluid):
    P = max(float(P), MIN_PRESSURE_PA)
    """(P[Pa], T[K], fluid name) -> H[J/kg].

    CoolProp는 Water가 포화선 바로 위에 있을 때 P,T 조합을 오류로 처리할 수 있다.
    이 경우 포화온도와 비교하여 포화액(Q=0) 또는 포화증기(Q=1) 엔탈피를 사용한다.
    """
    try:
        return PropsSI("H", "P", P, "T", T, fluid)
    except ValueError:
        if fluid.lower() != "water":
            raise
        T_sat = PropsSI("T", "P", P, "Q", 0, fluid)
        if T <= T_sat:
            return PropsSI("H", "P", P, "Q", 0, fluid)
        return PropsSI("H", "P", P, "Q", 1, fluid)
