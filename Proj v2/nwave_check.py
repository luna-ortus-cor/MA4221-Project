"""
Numerical check of the N-wave theorem (Section 8.5).

For compactly supported piecewise-constant u0 the entropy solution is computed
exactly (up to grid resolution) from the Lax-Oleinik formula
    u(x,t) = (x - y*)/t,   y* = argmin_y { U(y) + (x-y)^2/(2t) },   U(y) = int_{-inf}^y u0.
We compare it, in L^1, with the N-wave built from
  * CORRECTED constants   Q = -min U,  P = M - min U     (Theorem 8.x)
  * NAIVE constants       P = int u0^+, Q = int u0^-      (what is NOT true in general)

Data sets
  (a) N-shaped, no cancellation:  -1 on (0,1), +1 on (1,2)         (both constants agree)
  (b) cancelling:                 +1 on (0,1), -1 on (1,2)         (P=Q=0; ||u|| = 1/t exactly)
  (c) mixed signs:                six pieces of alternating sign

Outputs: figures/nwave_convergence.pdf, figures/nwave_table.csv
Run:  python3 nwave_check.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

DATA = {
    "(a) N-shaped": ([0, 1, 2], [-1.0, 1.0]),
    "(b) cancelling": ([0, 1, 2], [1.0, -1.0]),
    "(c) mixed": ([0, 0.7, 1.5, 2.1, 3.0, 3.8, 4.4], [1.2, -0.8, 1.6, -1.4, 0.9, -0.6]),
}


def primitive(br, vals, y):
    out = np.zeros_like(y)
    for i, v in enumerate(vals):
        out += v * np.clip(y - br[i], 0, br[i + 1] - br[i])
    return out


def constants(br, vals):
    yg = np.linspace(br[0] - 1, br[-1] + 1, 200001)
    Uy = primitive(br, vals, yg)
    M, mn = Uy[-1], Uy.min()
    corrected = (M - mn, -mn)                                   # (P, Q)
    naive = (sum(max(v, 0) * (br[i + 1] - br[i]) for i, v in enumerate(vals)),
             sum(max(-v, 0) * (br[i + 1] - br[i]) for i, v in enumerate(vals)))
    return corrected, naive


def entropy_solution(br, vals, X, t):
    L = np.abs(X).max() + 5
    yg = np.unique(np.concatenate([np.linspace(br[0] - 3, br[-1] + 3, 40001),
                                   np.linspace(-L, L, 20001)]))
    Uy = primitive(br, vals, yg)
    u = np.empty_like(X)
    for k, xx in enumerate(X):
        u[k] = (xx - yg[np.argmin(Uy + (xx - yg) ** 2 / (2 * t))]) / t
    return u


def nwave(P, Q, X, t):
    return np.where((X > -np.sqrt(2 * Q * t)) & (X < np.sqrt(2 * P * t)), X / t, 0.0)


if __name__ == "__main__":
    ts = [25, 100, 400, 1600, 6400]
    rows, results = [], {}
    for name, (br, vals) in DATA.items():
        (Pc, Qc), (Pn, Qn) = constants(br, vals)
        print(f"\n{name}:  corrected (P,Q)=({Pc:.4f},{Qc:.4f})   naive (P,Q)=({Pn:.4f},{Qn:.4f})")
        print("      t      L1 error (corrected)   L1 error (naive)   ||u||_L1")
        ec, en = [], []
        for t in ts:
            # corrected comparison: window sized to the corrected N-wave / the support, so the
            # (possibly very narrow) solution is resolved; naive comparison needs a window that
            # contains the (larger) naive N-wave.
            def window(P, Q):
                R = br[-1] + 4 + 1.5 * np.sqrt(2 * max(P, Q) * t)   # support + N-wave width
                return np.linspace(-R, R, 6001)
            Xc = window(Pc, Qc); dxc = Xc[1] - Xc[0]
            u = entropy_solution(br, vals, Xc, t)
            a = np.sum(np.abs(u - nwave(Pc, Qc, Xc, t))) * dxc
            Xn = window(max(Pc, Pn), max(Qc, Qn)); dxn = Xn[1] - Xn[0]
            un = entropy_solution(br, vals, Xn, t)
            b = np.sum(np.abs(un - nwave(Pn, Qn, Xn, t))) * dxn
            norm = np.sum(np.abs(u)) * dxc
            ec.append(a); en.append(b)
            rows.append((name, t, a, b, norm))
            print(f"  {t:6d}   {a:.4e}            {b:.4e}        {norm:.4e}")
        slope = np.polyfit(np.log(ts[1:]), np.log(np.maximum(ec[1:], 1e-300)), 1)[0]
        print(f"  fitted slope of corrected error vs t (t >= 100): {slope:.2f}")
        results[name] = (ec, en, slope)

    with open(os.path.join(OUT, "nwave_table.csv"), "w") as fh:
        fh.write("dataset,t,L1_corrected,L1_naive,L1_norm_u\n")
        for r in rows:
            fh.write(f'"{r[0]}",{r[1]},{r[2]:.6e},{r[3]:.6e},{r[4]:.6e}\n')

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for (name, (ec, en, s)), c in zip(results.items(), ["C0", "C1", "C2"]):
        ax.loglog(ts, np.maximum(ec, 1e-12), "o-", color=c, label=f"{name}: corrected $P,Q$")
        ax.loglog(ts, en, "x--", color=c, alpha=0.7, label=f"{name}: naive $P,Q$")
    ax.loglog(ts, 3 * np.array(ts, float) ** -0.5, "k:", lw=0.8, label=r"$t^{-1/2}$")
    ax.set(xlabel="$t$", ylabel=r"$\|u(\cdot,t)-N(\cdot,t)\|_{L^1}$")
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nwave_convergence.pdf")); plt.close(fig)
