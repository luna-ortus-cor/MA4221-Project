"""Table 2 of Section 8.4: measured Godunov shock error / dx versus the prediction of the
modified equation, and the effective viscosities.  Usage: python3 numerical_viscosity.py"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from godunov import init, run, l1_error

def predicted(lam):
    g = lambda u: (1-lam)*np.log(1-u) - lam*u + (1-lam/2)      # = 0 at u0
    u0 = brentq(g, 1e-12, 1-1e-15)
    return 2*quad(lambda u: u*(1-lam*u)/(1-u), 0, u0)[0]

def measured(lam, N=1600):
    U, dx = init(N, -2, 2, 1., 0.); U = run(U, dx, cfl=lam)
    return l1_error(U, -2, 2, dx, 1.0, 1., 0.)/dx

if __name__ == "__main__":
    print("CFL  measured(N=1600)  predicted  (pred-meas)/meas")
    for lam in (0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 0.95):
        m, p = measured(lam), predicted(lam)
        print("%.2f  %.3f  %.3f  %+.1f%%" % (lam, m, p, 100*(p-m)/m))
    lam = 0.45; m, p = measured(lam), predicted(lam)
    print("nu_eff/dx = E/(4 ln2 dx): measured %.3f, modified-equation %.3f" % (m/(4*np.log(2)), p/(4*np.log(2))))
