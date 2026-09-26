# h: height
# u: horizontal velocity
# hu: momentum per unit width
# U_i = [h_i, h_i * u_i]
# flux: F(U) = [h*u, h*u^2 + 0.5*g*h^2]

# update: U_{n+1} = U_n - (del t / del x)(F_{i+1/2} - F_{i-1/2})
# F_{i+1/2} = 1/2 (F_L + F_R) - 1/2 s_{max}*(U_R - U_L)
# s_{max} = max(|u_L| + sqrt(g*h_L), |u_R| + sqrt(g*h_R))

# del t = C*(del x / (max_i(|u_i|+sqrt(g*h_i))))
# C = 0.5

# boundary: u=0

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ==================== MATH ========================

num_cells = 103     # including boundary cells
cells = []          # array of tuples: (h_i, u_i)
U = []              # (h_i, h_i * u_i)
flux = []
dx = 0.05
x = np.arange(num_cells-2) * dx
C = 0.5
dt = 0.1
g = 9.81            # gravity constant

def iteration():
    boundary_update()
    time_update()
    for i in range(num_cells):
        flux_update(i)

    rusanov_flux = [get_mid_flux(i, i+1) for i in range(num_cells-1)]

    for i in range(1, num_cells-1):
        cell_update(i, rusanov_flux[i-1], rusanov_flux[i])

def time_update():
    global dt
    get_s_max = max([abs(cells[i][1]) + math.sqrt(g*cells[i][0]) for i in range(1, num_cells-1)])
    dt = C * (dx / get_s_max)

def cell_update(i, F_L, F_R):
    U[i][0] -= (dt/dx)*(F_R[0]-F_L[0])
    U[i][1] -= (dt/dx)*(F_R[1]-F_L[1])

    cells[i] = get_cell_info(i)

def flux_update(i: int):
    h_i = cells[i][0]
    u_i = cells[i][1]
    hu_i = U[i][1]
    flux[i][0] = hu_i
    flux[i][1] = hu_i*u_i + 0.5*g*h_i**2

def get_mid_flux(l: int, r: int):
    get_s_max = s_max(l, r)
    F = [0, 0]
    F[0] = 0.5 * (flux[l][0] + flux[r][0]) - 0.5*get_s_max*(U[r][0]-U[l][0])
    F[1] = 0.5 * (flux[l][1] + flux[r][1]) - 0.5*get_s_max*(U[r][1]-U[l][1])
    return F

def s_max(l: int, r: int):
    h_L = cells[l][0]
    u_L = cells[l][1]

    h_R = cells[r][0]
    u_R = cells[r][1]

    # sqrt(g*h): shallow water wave speed
    return max(abs(u_L) + math.sqrt(g*h_L), abs(u_R) + math.sqrt(g*h_R))

def boundary_update():
    # setting heights = neighbor
    U[0][0] = U[1][0]
    U[-1][0] = U[-2][0]

    # setting momentums = -neighbor
    U[0][1] = -U[1][1]
    U[-1][1] = -U[-2][1]

    # updating (h_i, u_i) boundary tuples
    cells[0] = get_cell_info(0)
    cells[-1] = get_cell_info(-1)

def get_cell_info(i):
    if U[i][0] == 0:
        return [0, 0]
    return [U[i][0], U[i][1] / U[i][0]]

def init_vals():
    global U, cells, flux
    init_height = 2.0
    
    h = np.full(num_cells, init_height)
    x_val = np.arange(num_cells) * dx
    h += 1.2*np.exp(-((x_val-3.0)/0.3)**2)
    hu = np.zeros(num_cells)

    U = [list(pair) for pair in zip(h, hu)]
    cells = [get_cell_info(i) for i in range(num_cells)]
    flux = [[U[i][1], U[i][1]*cells[i][1] + 0.5*g*U[i][0]**2] for i in range(num_cells)]

    time_update()
    boundary_update()


# ================= ANIMATION ==================

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True, linestyle='--', alpha=0.5)

line_total, = ax.plot([], [], 'blue', lw=2.5)
fill = None

# Initialization function for FuncAnimation
def init():
    line_total.set_data([], [])
    return line_total,

fps = 50
frames = 500

def animate(i):
    global fill

    iteration()
    height = [U[i][0] for i in range(1, num_cells-1)]

    if fill is not None:
        fill.remove()

    fill = ax.fill_between(x, 0, height, color='blue', alpha=0.5)
    
    line_total.set_data(x, height) 
    
    return line_total,

init_vals()

anim = FuncAnimation(
    fig, animate, init_func=init, frames=frames, interval=1000/fps, blit=False
)

# writer = PillowWriter(fps=35)
# anim.save("gifs/sine_refl.gif", writer=writer)

plt.show()
