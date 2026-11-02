"""Godunov scheme for Burgers' equation: tests quoted in Section 8.4.
Usage: python3 godunov.py"""
import numpy as np

def flux(ul, ur):
    """Godunov flux for f(u)=u^2/2: min of f on [ul,ur] if ul<=ur, else max on [ur,ul]."""
    f = lambda u: 0.5*u*u
    lo = np.where((ul <= 0) & (ur >= 0), 0.0, np.minimum(f(ul), f(ur)))
    hi = np.maximum(f(ul), f(ur))
    return np.where(ul <= ur, lo, hi)

def nonentropy_flux(ul, ur):
    """Consistent, conservative, non-monotone flux of Section 8.4: upwind in the sign of the average."""
    f = lambda u: 0.5*u*u
    return np.where(0.5*(ul+ur) > 0, f(ul), f(ur))

def init(N, a, b, ul, ur):
    dx = (b-a)/N
    xl = a + dx*np.arange(N); xr = xl + dx
    # cell average of Riemann data with jump at 0
    left = np.clip(0 - xl, 0, dx); right = dx - left
    return (ul*left + ur*right)/dx, dx

def run(U, dx, T=1.0, cfl=0.45, fl=flux):
    t = 0.0
    while t < T - 1e-14:
        m = max(np.max(np.abs(U)), 1e-12)
        dt = min(cfl*dx/m, T - t)
        Ue = np.concatenate(([U[0]], U, [U[-1]]))   # outflow (zero-gradient)
        F = fl(Ue[:-1], Ue[1:])
        U = U - dt/dx*(F[1:] - F[:-1])
        t += dt
    return U

def exact(x, t, ul, ur):
    if ul > ur:
        return np.where(x < 0.5*(ul+ur)*t, ul, ur)
    return np.clip(x/t, ul, ur)

def l1_error(U, a, b, dx, t, ul, ur, m=2000):
    N = len(U)
    xs = a + dx*(np.arange(N)[:, None] + (np.arange(m)[None, :]+0.5)/m)
    return np.sum(np.abs(U[:, None] - exact(xs, t, ul, ur)))*dx/m

def checks():
    """Implementation checks quoted in Section 8.4: flux against closed form, mass balance."""
    r = np.random.default_rng(1); a = r.uniform(-2, 2, 10**5); b = r.uniform(-2, 2, 10**5)
    g = a[:, None] + (b-a)[:, None]*np.linspace(0, 1, 2001)[None, :]
    closed = np.where(a <= b, 0.5*np.min(g**2, axis=1), 0.5*np.max(g**2, axis=1))
    print("flux vs closed form: max diff %.1e" % np.max(np.abs(flux(a, b)-closed)))
    for ul, ur in [(0., 1.), (1., 0.)]:
        U, dx = init(400, -2, 2, ul, ur); m0 = U.sum()*dx; t = 0.; bf = 0.
        while t < 1-1e-14:
            dt = min(0.45*dx/np.max(np.abs(U)), 1-t)
            Ue = np.concatenate(([U[0]], U, [U[-1]])); F = flux(Ue[:-1], Ue[1:])
            bf += -dt*(F[-1]-F[0]); U = U - dt/dx*(F[1:]-F[:-1]); t += dt
        print("mass balance", (ul, ur), "defect %.1e" % abs(U.sum()*dx-m0-bf))

if __name__ == "__main__":
    checks()
    for (ul, ur, name) in [(0., 1., "rarefaction"), (1., 0., "shock")]:
        for N in (200, 400, 800, 1600):
            U, dx = init(N, -2, 2, ul, ur); U = run(U, dx)
            print(name, N, "L1 = %.3e" % l1_error(U, -2, 2, dx, 1.0, ul, ur))
    # misaligned shock grids
    for N in (97, 100, 103):
        U, dx = init(N, -2, 2, 1., 0.); U = run(U, dx)
        print("shock N=%d  err/dx=%.3f" % (N, l1_error(U, -2, 2, dx, 1.0, 1., 0.)/dx))
    # boundary check: N=400 on [-2,2] vs same dx on [-4,4]
    for (ul, ur, name) in [(0., 1., "rarefaction"), (1., 0., "shock")]:
        U1, dx = init(400, -2, 2, ul, ur); U1 = run(U1, dx)
        U2, dx2 = init(800, -4, 4, ul, ur); U2 = run(U2, dx2)
        assert abs(dx-dx2) < 1e-15
        print(name, "max|U[-2,2]-U[-4,4]| = %.2e, L1 = %.2e" % (np.max(np.abs(U1-U2[200:600])), np.sum(np.abs(U1-U2[200:600]))*dx))
    # non-entropy scheme on (-1,1)
    for N in (200, 400, 800):
        U0, dx = init(N, -2, 2, -1., 1.)
        Ug = run(U0.copy(), dx); Un = run(U0.copy(), dx, fl=nonentropy_flux)
        print("data(-1,1) N=%d  Godunov L1=%.3e  non-entropy L1=%.3e  (change of non-entropy sol: %.1e)" % (
          N, l1_error(Ug, -2, 2, dx, 1.0, -1., 1.), l1_error(Un, -2, 2, dx, 1.0, -1., 1.), np.max(np.abs(Un-U0))))
