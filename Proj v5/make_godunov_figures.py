"""Figure 2 (godunov_rarefaction.pdf, godunov_shock.pdf) and Table 1 (L1 errors, observed orders)
of Section 8.4.  Usage: python3 make_godunov_figures.py   (writes ../figures/*.pdf)"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from godunov import init, run, exact, l1_error

def figures(N=400):
    for ul, ur, name, title in [(0., 1., "rarefaction", "Rarefaction data $(0,1)$"),
                                (1., 0., "shock", "Shock data $(1,0)$")]:
        U, dx = init(N, -2, 2, ul, ur); U = run(U, dx)
        xc = -2 + dx*(np.arange(N)+0.5); xf = np.linspace(-2, 2, 4001)
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ax.plot(xf, exact(xf, 1.0, ul, ur), "-", lw=1.2, label="exact")
        ax.plot(xc[::4], U[::4], ".", ms=3, label="Godunov, $N=%d$" % N)
        ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,1)$"); ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig("../figures/godunov_%s.pdf" % name); plt.close(fig)

def table1():
    prev = {}
    print("N, dx, L1(rarefaction), order, L1(shock), order")
    for N in (200, 400, 800, 1600, 3200, 6400):
        row = [N, 4.0/N]
        for ul, ur, k in [(0., 1., "r"), (1., 0., "s")]:
            U, dx = init(N, -2, 2, ul, ur); e = l1_error(U := run(U, dx), -2, 2, dx, 1.0, ul, ur)
            row += [e, np.log2(prev[k]/e) if k in prev else float("nan")]; prev[k] = e
        print("%d  %.3g  %.3e  %.2f  %.3e  %.2f" % tuple(row))

if __name__ == "__main__":
    figures(); table1()
