"""
Godunov as vanishing viscosity (Sections 8.4-8.5): two checks.

(A) The stationary viscous shock  u = -a*tanh(a x/(2 nu))  (a = u_l = -u_r) solves
    u u_x = nu u_xx and its L1 distance from the sharp shock -a*sgn(x) is 4 ln2 * nu,
    independent of a.

(B) Modified-equation heuristic for Godunov on u_t + (u^2/2)_x = 0 with u >= 0 (upwind):
        u_t + u u_x = (dx/2) ( u (1 - lam u) u_x )_x ,   lam = dt/dx .
    The travelling shock 1 -> 0 (speed 1/2) of this equation satisfies, in xi = (x - t/2)/dx,
        u' = (u-1)/(1 - lam u),
    i.e. xi(u) = xi0 + (1-lam) ln(1-u) - lam u.  Mass conservation fixes xi0 = 1 - lam/2, and
        L1 error of the profile / dx = 2 * int_0^{u0} u (1-lam u)/(1-u) du,  with xi(u0)=0.
    We compare this prediction with the MEASURED Godunov shock error (aligned grids) for
    several CFL numbers.  The heuristic is NOT a proof; this is a test of it.

Run:  python3 numerical_viscosity.py
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import godunov as g

# ---------------- (A) tanh profile ----------------
print("(A) L1 distance of -a*tanh(a x/(2 nu)) from -a*sgn(x), nu=0.01:")
for a in [0.3, 1.0, 2.5]:
    nu = 0.01
    x = np.linspace(-1.5, 1.5, 3_000_001)
    d = np.abs(-a * np.tanh(a * x / (2 * nu)) + a * np.sign(x))
    print(f"   a={a}:  L1/nu = {np.trapezoid(d, x)/nu:.5f}   (4 ln 2 = {4*np.log(2):.5f})")
# ODE check: nu u' = (u^2 - a^2)/2
a, nu = 1.3, 0.05
x = np.linspace(-1, 1, 2001); u = -a*np.tanh(a*x/(2*nu))
ux = np.gradient(u, x)
print(f"   ODE residual max|nu u_x - (u^2-a^2)/2| = {np.max(np.abs(nu*ux-(u**2-a**2)/2)[5:-5]):.2e}")

# ---------------- (B) modified equation vs measured Godunov ----------------
def predicted_L1_over_dx(lam):
    xi0 = 1 - lam / 2
    gfun = lambda u: (1 - lam) * np.log1p(-u) - lam * u
    u0 = brentq(lambda u: xi0 + gfun(u), 1e-12, 1 - 1e-12)
    E, _ = quad(lambda u: u * (1 - lam * u) / (1 - u), 0, u0)
    return 2 * E

print("\n(B) shock L1 error / dx:  measured Godunov (N multiple of 8)  vs  modified-equation prediction")
print("   CFL(lam)   measured(N=400)  measured(N=1600)   predicted   rel.diff")
for cfl in [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95]:
    meas = []
    for N in (400, 1600):
        x, U, _ = g.solve(N, 1, 0, cfl=cfl)
        meas.append(g.l1_error(x, U, 1, 0) / (4.0 / N))
    pred = predicted_L1_over_dx(cfl)          # lam = dt/dx = cfl/max|u| = cfl (max|u| = 1)
    print(f"   {cfl:5.2f}      {meas[0]:9.4f}        {meas[1]:9.4f}        {pred:8.4f}   {100*(pred/meas[1]-1):+6.1f}%")

lam = 0.45
pred = predicted_L1_over_dx(lam)
print(f"\nlam=0.45: predicted L1/dx = {pred:.4f};  equating with 4 ln2 nu_eff gives "
      f"nu_eff/dx = {pred/(4*np.log(2)):.4f}")
print(f"          measured 0.50 dx -> nu_eff/dx = {0.5/(4*np.log(2)):.4f}")
print(f"          critic's crude estimate (dx/2) u (1 - lam u) at u=1/2: "
      f"{0.5*0.5*(1-lam*0.5):.4f} dx")
