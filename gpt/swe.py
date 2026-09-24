import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1D Shallow-Water Equation Solver
#
#   h_t + (hu)_x = 0
#   (hu)_t + (hu^2 + 1/2 g h^2)_x = 0
#
# Finite-volume method with Rusanov flux.
# Left boundary: open
# Right boundary: reflective wall
# ============================================================

# -----------------------------
# Physical parameters
# -----------------------------
g = 9.81
H = 2.0                 # still-water depth

# -----------------------------
# Numerical parameters
# -----------------------------
L = 30.0
N = 600
dx = L / N

CFL = 0.4
dt_max = 0.02

# Cell centers
x = (np.arange(N) + 0.5) * dx

# -----------------------------
# Conserved variables
#
# U[0] = h
# U[1] = h*u
# -----------------------------
h = np.full(N, H)
hu = np.zeros(N)

# ============================================================
# Initial waves
# ============================================================

# Wave 1: starts near the left and travels right
x1 = 6.0
sigma1 = 1.0
amplitude1 = 0.4

h += amplitude1 * np.exp(
    -((x - x1) / sigma1) ** 2
)

# Give the wave rightward velocity
hu += np.sqrt(g * H) * (
    h - H
)

# Wave 2: another right-moving disturbance
x2 = 12.0
sigma2 = 0.8
amplitude2 = 0.25

h += amplitude2 * np.exp(
    -((x - x2) / sigma2) ** 2
)

hu += np.sqrt(g * H) * (
    amplitude2 * np.exp(-((x - x2) / sigma2) ** 2)
)


# ============================================================
# Numerical flux
# ============================================================

def flux(h, hu):
    """
    Physical flux for the shallow-water equations.
    """
    u = hu / h

    return np.array([
        hu,
        hu * u + 0.5 * g * h**2
    ])


def rusanov_flux(hL, huL, hR, huR):
    """
    Rusanov (local Lax-Friedrichs) numerical flux.
    """

    # Prevent division by zero
    hL = max(hL, 1e-8)
    hR = max(hR, 1e-8)

    uL = huL / hL
    uR = huR / hR

    FL = flux(hL, huL)
    FR = flux(hR, huR)

    # Maximum wave speed
    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)

    a = max(
        abs(uL) + cL,
        abs(uR) + cR
    )

    UL = np.array([hL, huL])
    UR = np.array([hR, huR])

    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL)


# ============================================================
# One simulation timestep
# ============================================================

def step(h, hu, dt):

    # --------------------------------------------------------
    # Construct ghost cells
    #
    # Right wall:
    #
    #     u -> | WALL
    #     u <- |
    #
    # Reflect velocity, preserve height.
    # --------------------------------------------------------

    hg = np.empty(N + 2)
    hug = np.empty(N + 2)

    hg[1:-1] = h
    hug[1:-1] = hu

    # Left: transmissive/open boundary
    hg[0] = h[0]
    hug[0] = hu[0]

    # Right: reflective wall
    hg[-1] = h[-1]
    hug[-1] = -hu[-1]

    # --------------------------------------------------------
    # Compute fluxes at every cell interface
    # --------------------------------------------------------

    F = np.zeros((N + 1, 2))

    for i in range(N + 1):
        F[i] = rusanov_flux(
            hg[i],
            hug[i],
            hg[i + 1],
            hug[i + 1]
        )

    # --------------------------------------------------------
    # Finite-volume update
    #
    # U_i^(n+1) =
    # U_i^n - dt/dx (F_(i+1/2) - F_(i-1/2))
    # --------------------------------------------------------

    U = np.vstack((h, hu))

    U -= (dt / dx) * (
        F[1:].T - F[:-1].T
    )

    h_new = U[0]
    hu_new = U[1]

    # Avoid numerical negative depths
    h_new = np.maximum(h_new, 1e-6)

    return h_new, hu_new


# ============================================================
# Animation
# ============================================================

fig, ax = plt.subplots(figsize=(11, 4))

line, = ax.plot(x, h, linewidth=2)

# Still-water level
ax.axhline(H, linestyle="--", linewidth=1)

# Wall
ax.axvline(L, linewidth=4)

ax.set_xlim(0, L)
ax.set_ylim(H - 1, H + 1.5)

ax.set_xlabel("x")
ax.set_ylabel("Water depth h")
ax.set_title("1D Shallow-Water Equations")

time_text = ax.text(
    0.02, 0.92,
    "",
    transform=ax.transAxes
)


# Simulation time
t = 0.0


def update(frame):
    global h, hu, t

    # Advance several small timesteps per frame
    for _ in range(4):

        # CFL timestep based on current solution
        u = hu / h
        c = np.sqrt(g * h)

        max_speed = np.max(
            np.abs(u) + c
        )

        dt = min(
            CFL * dx / max_speed,
            dt_max
        )

        h, hu = step(h, hu, dt)

        t += dt

    line.set_data(x, h)

    time_text.set_text(
        f"t = {t:.2f} s"
    )

    return line, time_text


ani = FuncAnimation(
    fig,
    update,
    interval=20,
    blit=True
)

plt.tight_layout()
plt.show()