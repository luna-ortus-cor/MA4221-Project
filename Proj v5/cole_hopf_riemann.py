"""Cole-Hopf solution of viscous Burgers' for Riemann data, closed form via erfc, and L1 distance
to the entropy solution at t=1 (Section 8.3 check of Theorem 8.4).  Usage: python3 cole_hopf_riemann.py"""
import numpy as np
from scipy.special import log_ndtr

def ch(x, t, nu, ul, ur):
    s = np.sqrt(2*nu*t)
    # left piece y<0: G = ul*y + (x-y)^2/2t = (y-cl)^2/2t + ul*x - ul^2 t/2,  cl = x - ul t
    cl, cr = x - ul*t, x - ur*t
    Al = -(ul*x - 0.5*ul**2*t)/(2*nu); Ar = -(ur*x - 0.5*ur**2*t)/(2*nu)
    lZl = np.log(s*np.sqrt(2*np.pi)) + log_ndtr(-cl/s)      # int_{-inf}^0 exp(-(y-cl)^2/2s^2)
    lZr = np.log(s*np.sqrt(2*np.pi)) + log_ndtr(cr/s)       # int_0^{inf}
    lw_l, lw_r = Al + lZl, Ar + lZr
    m = np.maximum(lw_l, lw_r); wl = np.exp(lw_l-m); wr = np.exp(lw_r-m)
    lphi = lambda z: -0.5*z*z - 0.5*np.log(2*np.pi)
    yl = cl - s*np.exp(lphi(cl/s) - log_ndtr(-cl/s))   # mean of y over y<0 piece
    yr = cr + s*np.exp(lphi(cr/s) - log_ndtr(cr/s))    # mean over y>0 piece
    ybar = (wl*yl + wr*yr)/(wl+wr)
    return (x - ybar)/t

def exact(x, t, ul, ur):
    return np.where(x < 0.5*(ul+ur)*t, ul, ur) if ul > ur else np.clip(x/t, ul, ur)

if __name__ == "__main__":
    x = np.linspace(-2, 2, 800001); dx = x[1]-x[0]
    print("nu, err/nu (shock 1->0), err/nu (rarefaction 0->1), err (rarefaction)")
    prev = None
    for nu in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        e = []
        for ul, ur in [(1., 0.), (0., 1.)]:
            e.append(np.sum(np.abs(ch(x, 1.0, nu, ul, ur) - exact(x, 1.0, ul, ur)))*dx)
        print("%.0e  %.3f  %.3f  %.3e" % (nu, e[0]/nu, e[1]/nu, e[1]))
