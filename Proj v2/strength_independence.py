"""
Does the layer part of the vanishing-viscosity error depend on the shock strength?
Prediction (Section 8.4): a viscous shock of ANY strength and speed has L1 distance
4 ln2 * nu from the sharp shock (Proposition 8.x), so the shock-layer contribution to
||u^nu - u||_L1 should be 4 ln2 * nu ~ 2.773 nu independent of the strength a = u_l(t).

Test on u0 = -arctan x at t = 1.1, 1.5, 3, 10 (a = 0.52 ... 1.50), nu = 3e-4.  The error is
integrated over |x| < delta; because the outer (smooth) solution also contributes O(nu)
inside the window, we report delta = 0.05 and delta = 0.005 (the layer half-width is ~ 4 nu/(2a)).

Run:  python3 strength_independence.py     (writes figures/strength_independence.csv)
"""
import os
import numpy as np
import arctan_example as ae

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
FOUR_LN2 = 4 * np.log(2)


def layer_ratio(t, nu, deltas):
    Y = 6 + 1.6 * t + 8                                    # y-range covering all minimisers
    xg = np.unique(np.concatenate([np.linspace(-6, 6, 2401), np.linspace(-0.1, 0.1, 2001),
                                   np.linspace(-0.02, 0.02, 8001)]))
    xg = xg[xg != 0.0]                                     # jump point must not be a node
    diff = np.abs(ae.cole_hopf(xg, t, nu, Y=Y) - ae.entropy_solution(xg, t))
    return [np.trapezoid(diff[np.abs(xg) < d], xg[np.abs(xg) < d]) / (FOUR_LN2 * nu) for d in deltas]


if __name__ == "__main__":
    nu, deltas, rows = 3e-4, [0.05, 0.005], []
    print(f"layer error / (4 ln2 nu)   at nu={nu}   (prediction: 1)")
    print("    t   strength a   outer slope u_x(0+)   delta=0.05   delta=0.005")
    for t in [1.1, 1.5, 3.0, 10.0]:
        ys = ae.ystar(t); a = np.arctan(ys)
        slope = -(1 / (1 + ys ** 2)) / (1 - t / (1 + ys ** 2))   # outer solution's slope beside the shock
        r = layer_ratio(t, nu, deltas)
        rows.append((t, a, slope, *r))
        print(f"{t:6.1f}   {a:8.4f}   {slope:14.3f}       {r[0]:8.4f}     {r[1]:8.4f}")
    np.savetxt(os.path.join(OUT, "strength_independence.csv"), rows, delimiter=",",
               header="t,strength_a,outer_slope,ratio_delta_0.05,ratio_delta_0.005",
               comments="", fmt="%.6g")
