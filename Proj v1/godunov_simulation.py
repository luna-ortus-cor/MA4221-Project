"""
Godunov's method for the inviscid Burgers' equation u_t + (u^2/2)_x = 0,
compared against the exact entropy solutions of the Riemann problem
(Theorem 7.1 in the paper).
"""
import numpy as np
import matplotlib.pyplot as plt


def godunov_flux_burgers(ul, ur):
    """Exact Godunov numerical flux for Burgers' equation f(u) = u^2/2."""
    if ul > ur:
        s = 0.5 * (ul + ur)
        u_star = ul if s > 0 else ur
    else:
        if ul >= 0:
            u_star = ul
        elif ur <= 0:
            u_star = ur
        else:
            u_star = 0.0
    return 0.5 * u_star ** 2


def run_godunov(u_left, u_right, x_min=-2.0, x_max=2.0, ncells=400,
                 t_final=1.0, cfl=0.45):
    """Run Godunov's method on the Riemann problem (u_left, u_right)."""
    dx = (x_max - x_min) / ncells
    x_centers = x_min + (np.arange(ncells) + 0.5) * dx

    U = np.where(x_centers < 0.0, u_left, u_right).astype(float)

    t = 0.0
    while t < t_final:
        alpha = np.max(np.abs(U)) if np.max(np.abs(U)) > 0 else 1.0
        dt = cfl * dx / alpha
        if t + dt > t_final:
            dt = t_final - t

        # Interface fluxes: F[j] is flux between cell j-1 and cell j
        U_ext = np.concatenate(([U[0]], U, [U[-1]]))  # ghost cells
        F = np.array([godunov_flux_burgers(U_ext[j], U_ext[j + 1])
                      for j in range(ncells + 1)])

        U = U - (dt / dx) * (F[1:] - F[:-1])
        t += dt

    return x_centers, U


def exact_riemann_burgers(x, t, ul, ur):
    """Exact entropy solution of the Burgers Riemann problem at time t."""
    u = np.empty_like(x)
    if ul > ur:
        s = 0.5 * (ul + ur)
        u = np.where(x < s * t, ul, ur)
    else:
        u = np.where(x <= ul * t, ul,
             np.where(x >= ur * t, ur, x / t))
    return u


def make_comparison_plot(ul, ur, t_final, filename, title):
    x_num, U_num = run_godunov(ul, ur, t_final=t_final)
    x_exact = np.linspace(-2.0, 2.0, 2000)
    U_exact = exact_riemann_burgers(x_exact, t_final, ul, ur)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(x_exact, U_exact, 'r-', linewidth=2, label='Exact entropy solution')
    ax.plot(x_num, U_num, 'b.', markersize=3, label='Godunov ($N=400$ cells)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)
    fig.tight_layout(pad=0.3)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    l1_error = np.mean(np.abs(np.interp(x_num, x_exact, U_exact) - U_num))
    return l1_error


if __name__ == "__main__":
    # Rarefaction case: u_l = 0, u_r = 1
    err_raref = make_comparison_plot(
        ul=0.0, ur=1.0, t_final=1.0,
        filename="/home/claude/burgers_paper/figures/godunov_rarefaction.pdf",
        title="Rarefaction: $u_\\ell=0,\\ u_r=1$ at $t=1$"
    )
    print(f"Rarefaction case L1 error: {err_raref:.5f}")

    # Shock case: u_l = 1, u_r = 0
    err_shock = make_comparison_plot(
        ul=1.0, ur=0.0, t_final=1.0,
        filename="/home/claude/burgers_paper/figures/godunov_shock.pdf",
        title="Shock: $u_\\ell=1,\\ u_r=0$ at $t=1$"
    )
    print(f"Shock case L1 error: {err_shock:.5f}")
