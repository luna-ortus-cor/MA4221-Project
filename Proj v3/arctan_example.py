"""
Worked example u0(x) = -arctan(x) carried past breakdown (Sections 3, 6, 8).

(1) Exact entropy solution for t > 1 from the Lax-Oleinik formula:
      x = y - t*arctan(y),  u = -arctan(y)  (outer branch), odd in x,
    with a stationary shock at x = 0 of strength u_l = -u_r = arctan(y*(t)),
    where y*(t) > 0 is the unique positive root of  arctan(y) = y/t.
(2) Vanishing viscosity: Cole-Hopf solutions u^nu (eq. 42) vs the entropy solution.

Outputs: figures/arctan_shock_strength.pdf, figures/arctan_vanishing_viscosity.pdf,
         figures/arctan_viscosity_table.csv
Run:  python3 arctan_example.py
"""
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def U(y):                       # U(y) = int_0^y u0 = -(y atan y - 0.5 ln(1+y^2))
    return -(y * np.arctan(y) - 0.5 * np.log1p(y * y))


def ystar(t):
    """Unique positive root of arctan(y) = y/t (exists iff t > 1)."""
    return brentq(lambda y: np.arctan(y) - y / t, 1e-9, t * np.pi / 2 + 1)


def entropy_solution(x, t):
    """Exact entropy solution via the Lax-Oleinik parametrisation."""
    x = np.atleast_1d(np.asarray(x, float))
    u = np.empty_like(x)
    for k, xx in enumerate(x):
        a = abs(xx)
        if t <= 1:
            lo = -1.0 - t * np.pi
            y = brentq(lambda y: y - t * np.arctan(y) - xx, lo - abs(xx), abs(xx) + t * np.pi)
            u[k] = -np.arctan(y)
        else:
            ys = ystar(t)
            if a == 0:
                u[k] = 0.0
                continue
            # outer branch y >= y*(t): y - t*atan(y) is increasing there, equals 0 at y*
            hi = a + t * np.pi
            y = brentq(lambda y: y - t * np.arctan(y) - a, ys, hi)
            u[k] = -np.sign(xx) * np.arctan(y)
    return u


def lax_oleinik_bruteforce(x, t):
    yg = np.linspace(-60, 60, 240001)
    Uy = U(yg)
    out = np.empty_like(x)
    for k, xx in enumerate(x):
        out[k] = (xx - yg[np.argmin(Uy + (xx - yg) ** 2 / (2 * t))]) / t
    return out


def cole_hopf(x, t, nu, Y=20.0, dy=0.002):
    y = np.arange(-Y, Y + dy / 2, dy)
    Uy = U(y)
    out = np.empty_like(x)
    for k, xx in enumerate(x):
        G = Uy + (xx - y) ** 2 / (2 * t)
        w = np.exp(-(G - G.min()) / (2 * nu))            # log-sum-exp stabilisation
        out[k] = np.sum((xx - y) / t * w) / np.sum(w)
    return out


if __name__ == "__main__":
    # ---------------- (1) shock strength ----------------
    print("t      y*        u_l=atan(y*)   sqrt(3(t-1))   pi/2")
    for t in [1.001, 1.01, 1.1, 1.5, 2, 5, 20, 100, 1000]:
        ys = ystar(t)
        print(f"{t:7.3f} {ys:9.4f} {np.arctan(ys):12.6f} {np.sqrt(3*(t-1)):12.6f} {np.pi/2:9.6f}")
    # check the parametrisation against brute-force Lax-Oleinik minimisation
    for t in [0.5, 1.0, 2.0, 5.0]:
        xs = np.linspace(-4, 4, 161); xs = xs[np.abs(xs) > 1e-9]
        err = np.max(np.abs(entropy_solution(xs, t) - lax_oleinik_bruteforce(xs, t)))
        print(f"parametrisation vs brute-force Lax-Oleinik, t={t}: max |diff| = {err:.2e}")
    # critic's expansion: relative error of sqrt(3(t-1))
    for eps in [1e-4, 1e-3, 1e-2, 1e-1]:
        t = 1 + eps
        print(f"  t-1={eps:g}: u_l/sqrt(3(t-1)) = {np.arctan(ystar(t))/np.sqrt(3*eps):.5f}")

    ts = np.linspace(1.0001, 12, 400)
    ul = [np.arctan(ystar(t)) for t in ts]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(ts, ul, "b-", label=r"$u_\ell(t)=\arctan y^*(t)$")
    ax.plot(ts, np.sqrt(3 * (ts - 1)), "r--", lw=1, label=r"$\sqrt{3(t-1)}$ (near $t=1$)")
    ax.axhline(np.pi / 2, color="k", ls=":", label=r"$\pi/2$")
    ax.set(xlabel="$t$", ylabel="shock strength: $u_\\ell=-u_r$", ylim=(0, 1.8), xlim=(1, 12))
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "arctan_shock_strength.pdf")); plt.close(fig)

    # ---------------- (2) vanishing viscosity at t = 3 ----------------
    t = 3.0
    nus = [0.1, 0.03, 0.01, 0.003, 0.001]          # plotted
    nus_table = nus + [0.0003, 0.0001]             # table / error study
    xplot = np.linspace(-3, 3, 1201)
    ue = entropy_solution(xplot, t)
    fig, ax0 = plt.subplots(figsize=(4.6, 3.4))
    ax0.plot(xplot, ue, "k-", lw=2, label="entropy solution")
    for nu in nus:
        ax0.plot(xplot, cole_hopf(xplot, t, nu), lw=1, label=rf"$\nu={nu}$")
    ax0.set(xlabel="$x$", ylabel="$u(x,3)$", title=r"Cole--Hopf solutions as $\nu\to0$")
    ax0.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "arctan_vanishing_viscosity.pdf")); plt.close(fig)

    # L1 error on [-6,6], grid refined near the shock layer
    xg = np.unique(np.concatenate([np.linspace(-6, 6, 2401), np.linspace(-0.1, 0.1, 2001),
                                   np.linspace(-0.02, 0.02, 4001)]))
    # The entropy solution is discontinuous at x=0 (value 0 is only the jump midpoint) while the
    # integrand |u^nu - u| is ~u_l on BOTH sides there, so the point x=0 itself must not be a
    # quadrature node (it would be weighted by the local spacing with the wrong value).
    xg = xg[xg != 0.0]
    ue_g = entropy_solution(xg, t)
    rows, prev = [], None
    print("\nnu        L1 error      error/nu   (error-4ln2*nu)/nu   order vs previous nu")
    for nu in nus_table:
        e = np.trapezoid(np.abs(cole_hopf(xg, t, nu) - ue_g), xg)
        order = np.log(prev[1] / e) / np.log(prev[0] / nu) if prev else float("nan")
        prev = (nu, e)
        rows.append((nu, e, order, e / nu, e / nu - 4 * np.log(2)))
        print(f"{nu:8.4f}  {e:.4e}   {e/nu:7.4f}    {e/nu-4*np.log(2):7.4f}            {order:6.3f}")
    np.savetxt(os.path.join(OUT, "arctan_viscosity_table.csv"), rows, delimiter=",",
               header="nu,L1_error,order,error_over_nu,excess_over_4ln2", comments="",
               fmt=["%.4f", "%.6e", "%.3f", "%.4f", "%.4f"])
