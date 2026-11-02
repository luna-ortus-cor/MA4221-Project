"""
Godunov's method for Burgers' equation  u_t + (u^2/2)_x = 0   (Section 8.4)

Reproduces:
  figures/godunov_rarefaction.pdf   Riemann data (u_l,u_r) = (0,1), t = 1
  figures/godunov_shock.pdf         Riemann data (u_l,u_r) = (1,0), t = 1
  figures/godunov_convergence.pdf   L1 error vs N (both cases)
  figures/convergence_table.csv     the numbers behind the convergence plot

Setup (as in the paper): x in [-2,2], N cells, CFL number 0.45, run to t=1,
zero-gradient (outflow) boundary conditions, initial cell averages of the
piecewise-constant Riemann data (the discontinuity sits on a cell interface
since x=0 is a multiple of dx for even N).

Run:  python3 godunov.py            (from the code/ directory)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, T, CFL = 2.0, 1.0, 0.45
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def f(u):
    return 0.5 * u * u


def godunov_flux(ul, ur):
    """Godunov flux f(u^R(0; ul, ur)) via the exact Riemann solution (eq. 46).

    shock  (ul > ur): value at xi = 0 is ul if s=(ul+ur)/2 > 0, else ur
    fan    (ul <= ur): ul if ul >= 0;  ur if ur <= 0;  0 if ul < 0 < ur
    """
    shock = np.where((ul + ur) / 2 > 0, f(ul), f(ur))
    fan = np.where(ul >= 0, f(ul), np.where(ur <= 0, f(ur), 0.0))
    return np.where(ul > ur, shock, fan)


def godunov_flux_minmax(ul, ur):
    """Independent closed form for convex f with minimum at 0:
         F = max( f(max(ul,0)), f(min(ur,0)) )   if ul <= ur ... generally
         F = min over [ul,ur] of f  if ul<=ur,  max over [ur,ul] of f  if ul>ur.
    Used only as a cross-check of godunov_flux."""
    lo, hi = np.minimum(ul, ur), np.maximum(ul, ur)
    fmin = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(f(lo), f(hi)))
    fmax = np.maximum(f(lo), f(hi))
    return np.where(ul <= ur, fmin, fmax)


def solve(N, ul, ur, T=T, cfl=CFL):
    dx = 2 * L / N
    x = -L + dx * (np.arange(N) + 0.5)
    U = np.where(x < 0, ul, ur).astype(float)
    t, steps = 0.0, 0
    while t < T - 1e-14:
        dt = min(cfl * dx / max(np.abs(U).max(), 1e-12), T - t)
        Ue = np.concatenate([[U[0]], U, [U[-1]]])          # outflow BCs
        F = godunov_flux(Ue[:-1], Ue[1:])                  # F_{j+1/2}
        U = U - dt / dx * (F[1:] - F[:-1])
        t += dt
        steps += 1
    return x, U, steps


def exact(x, ul, ur, t=T):
    """Entropy solution of Theorem 7.1 (cell centres; x != s t generically)."""
    if ul > ur:
        return np.where(x < 0.5 * (ul + ur) * t, ul, ur)
    return np.clip(x / t, ul, ur)


def l1_error(x, U, ul, ur):
    """Exact L1 error: integrate |U_j - u_exact| over each cell using the
    exact solution's cell average (fine quadrature), not just its midpoint."""
    dx = x[1] - x[0]
    m = 20
    xs = (x[:, None] - dx / 2 + dx * (np.arange(m) + 0.5) / m).ravel()
    ue = exact(xs, ul, ur).reshape(-1, m)
    return np.sum(np.abs(U[:, None] - ue)) * dx / m


def make_figure(ul, ur, name, title, N=400):
    x, U, _ = solve(N, ul, ur)
    xf = np.linspace(-L, L, 4001)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(xf, exact(xf, ul, ur), "r-", lw=1.6, label="Exact entropy solution")
    ax.plot(x, U, "b.", ms=3, label=f"Godunov ($N={N}$ cells)")
    ax.set(xlabel="$x$", ylabel="$u(x,t)$", title=title, xlim=(-L, L), ylim=(-0.02, 1.02))
    ax.legend(loc="upper left" if ul < ur else "lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


if __name__ == "__main__":
    # ---- sanity checks -------------------------------------------------
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-3, 3, 10**5), rng.uniform(-3, 3, 10**5)
    assert np.allclose(godunov_flux(a, b), godunov_flux_minmax(a, b)), "flux mismatch"
    print("flux check:  case-based Godunov flux == min/max closed form  (1e5 random pairs)")

    # conservation: with outflow BCs, d/dt(mass) = f(u_left_bdry) - f(u_right_bdry)
    for ul, ur in [(0, 1), (1, 0)]:
        x, U, _ = solve(400, ul, ur)
        dx = x[1] - x[0]
        mass0 = (ul + ur) * L                      # integral of the initial data
        expected = mass0 + T * (f(ul) - f(ur))     # boundary states stay ul, ur
        print(f"  (ul,ur)=({ul},{ur}): mass = {U.sum()*dx:.10f}, expected {expected:.10f}; "
              f"range [{U.min():.4f},{U.max():.4f}] (monotone scheme: no over/undershoot)")

    # ---- figures -------------------------------------------------------
    make_figure(0, 1, "godunov_rarefaction.pdf", r"Rarefaction: $u_\ell=0,\ u_r=1$ at $t=1$")
    make_figure(1, 0, "godunov_shock.pdf", r"Shock: $u_\ell=1,\ u_r=0$ at $t=1$")

    # ---- convergence study --------------------------------------------
    Ns = [50, 100, 200, 400, 800, 1600, 3200]
    rows = []
    print("\n   N      dx      L1 rare    order    L1 shock   order")
    prev = {}
    for N in Ns:
        r = {}
        for key, (ul, ur) in {"rare": (0, 1), "shock": (1, 0)}.items():
            x, U, _ = solve(N, ul, ur)
            e = l1_error(x, U, ul, ur)
            order = np.log2(prev[key] / e) if key in prev else float("nan")
            prev[key] = e
            r[key] = (e, order)
        rows.append((N, 4.0 / N, r["rare"][0], r["rare"][1], r["shock"][0], r["shock"][1]))
        print(f"{N:5d} {4.0/N:8.5f} {r['rare'][0]:10.3e} {r['rare'][1]:7.2f} "
              f"{r['shock'][0]:10.3e} {r['shock'][1]:7.2f}")

    np.savetxt(os.path.join(OUT, "convergence_table.csv"), rows, delimiter=",",
               header="N,dx,L1_rarefaction,order_rarefaction,L1_shock,order_shock",
               comments="", fmt=["%d", "%.6f", "%.6e", "%.3f", "%.6e", "%.3f"])

    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.loglog(arr[:, 1], arr[:, 2], "o-", label="rarefaction")
    ax.loglog(arr[:, 1], arr[:, 4], "s-", label="shock")
    ax.loglog(arr[:, 1], 0.5 * arr[:, 1], "k--", lw=0.8, label=r"$O(\Delta x)$")
    ax.set(xlabel=r"$\Delta x$", ylabel=r"$L^1$ error at $t=1$", title="Godunov convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "godunov_convergence.pdf"))
    plt.close(fig)

    # N = 400 headline numbers
    i = Ns.index(400)
    print(f"\nN=400:  L1 rarefaction = {arr[i,2]:.3e}  (mean abs = {arr[i,2]/4:.3e})")
    print(f"        L1 shock       = {arr[i,4]:.3e}  (mean abs = {arr[i,4]/4:.3e})")
