"""
[Task 5] Main.py
Run counter-current HX simulation with saturated cold outlet target.

The previous Tsat + 20 K target was not practically reachable with the
quick-test setting. This version uses Tsat + 0 K first.
"""

import matplotlib.pyplot as plt
from Data_model import get_fixed_conditions
from Optimizer import optimize_length, save_node_data, save_cold_flow_data, cold_flow_view


def run_simulation():
    c = get_fixed_conditions()
    geom_extra = {"A_flow_hot": 1e-3, "A_flow_cold": 1e-3, "P_w_hot": 0.628, "P_w_cold": 0.628,
                  "correlation_model": c["model"]["correlation_model"]}

    print("=" * 60)
    print("[Task 5] Counter-current full simulation")
    print(f"Correlation model = {geom_extra['correlation_model']}")
    print(f"Hot inlet       : x=0, P={c['hot_inlet']['P_in']/1e6:.2f} MPa, T={c['hot_inlet']['T_in']:.2f} K")
    print(f"Cold inlet      : x=L, P={c['cold_inlet']['P_in']/1e6:.2f} MPa, T={c['cold_inlet']['T_in']:.2f} K")
    print(f"Cold target out : x=0, T={c['target']['T_cold_out']:.2f} K = Tsat + {c['target']['superheat_margin_K']:.1f} K")
    print("q'' is estimated from preliminary single-phase U*dT, not hard-coded.")
    print("=" * 60)

    def progress(it, L, T, err):
        print(f"iter {it:02d}: L={L:9.4f} m | T_cold_out={T:9.3f} K | err={err:+8.3f} K")

    result = optimize_length(N=20, geom_extra=geom_extra, L_min=0.1, L_max=20.0, tol=0.5, max_iter=12, progress_cb=progress)
    save_node_data(result["node_data"], "node_data_x_coordinate.csv")
    save_node_data(result["node_data"], "node_data.csv")
    save_cold_flow_data(result["node_data"], "cold_flow_node_data.csv")

    nd = result["node_data"]
    Q_total = sum(r.get("q_cell", 0.0) for r in nd)
    print("\n" + "=" * 60)
    print("Final result")
    print(f"L = {result['L']:.5f} m")
    print(f"Cold outlet at x=0 = {result['T_cold_out']:.3f} K")
    print(f"Hot outlet at x=L  = {nd[-1]['T_hot']:.3f} K")
    print(f"Q_total = {Q_total:.2f} W")
    print(f"Converged = {result['converged']}")
    print(f"Message = {result.get('message', '')}")
    print("CSV saved: node_data_x_coordinate.csv, cold_flow_node_data.csv")
    print("=" * 60)
    return result


def plot_result(result):
    nd = result["node_data"]
    x = [r["x"] for r in nd]
    T_hot = [r["T_hot"] for r in nd]
    T_cold_x = [r["T_cold"] for r in nd]
    U = [r.get("U", 0.0) for r in nd]
    q_flux = [r.get("q_flux_sp", 0.0) for r in nd]

    plt.figure()
    plt.plot(x, T_hot, label="Hot, x direction")
    plt.plot(x, T_cold_x, label="Cold shown on x-coordinate")
    plt.xlabel("x [m]  (hot flow direction)")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature profile on x-coordinate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("temperature_profile_x_coordinate.png", dpi=200)

    cold = cold_flow_view(nd)
    s = [r["s_cold"] for r in cold]
    T_cold = [r["T_cold"] for r in cold]
    P_cold = [r["P_cold"] / 1e6 for r in cold]

    plt.figure()
    plt.plot(s, T_cold)
    plt.xlabel("s_cold [m]  (cold actual flow direction)")
    plt.ylabel("Cold temperature [K]")
    plt.title("Cold temperature along actual cold-flow direction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cold_temperature_flow_direction.png", dpi=200)

    plt.figure()
    plt.plot(s, P_cold)
    plt.xlabel("s_cold [m]  (cold actual flow direction)")
    plt.ylabel("Cold pressure [MPa]")
    plt.title("Cold pressure along actual cold-flow direction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cold_pressure_flow_direction.png", dpi=200)

    plt.figure()
    plt.plot(x, q_flux)
    plt.xlabel("x [m]")
    plt.ylabel("q'' estimate [W/m²]")
    plt.title("Estimated heat-flux profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("heat_flux_profile.png", dpi=200)

    plt.figure()
    plt.plot(x, U)
    plt.xlabel("x [m]")
    plt.ylabel("U [W/m²-K]")
    plt.title("Overall heat-transfer coefficient profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("U_profile.png", dpi=200)

    print("Plots saved.")


if __name__ == "__main__":
    result = run_simulation()
    plot_result(result)
