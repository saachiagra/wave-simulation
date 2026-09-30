'''
*** 2D Shallow Water Equation Simulation (Finite Volume Solver) ***
- using Rusanov Flux to approximate flux updates
- visualization done using Matplotlib

Variables:
h: height
u: velocity in x direction
v: velocity in y direction
g: gravitational constant
F: flux in x direction
G: flux in y direction
n: normal vector
C: CFL constant

U = [h, hu, hv]
F = [hu, hu^2 + 0.5gh^2, huv]
G = [hv, huv, hv^2 + 0.5gh^2]

Updates (using Rusanov Flux):
U_{k+1} = U_k - (del t / del x)(F_{i+1/2, j} - F_{i-1/2, j}) - (del t / del y)(G_{i, j+1/2} - G_{i, j-1/2})
F_{i+1/2, j} = 0.5(F_{i,j} + F{i+1, j}) - 0.5 s_max*(U_{i+1, j} - U_{i, j})
G_{i+1/2, j} = 0.5(F_{i,j} + F{i, j+1}) - 0.5 s_max*(U_{i, j+1} - U_{i, j})
s_{max} = max(|u_L*n_x + v_L*n_y| + sqrt(g*h_L), |u_R*n_x + v_R*n_y| + sqrt(g*h_R))
del t = C * min(del x / (max(|u| + sqrt(gh), del y / (max(|v| + sqrt(gh))))))

'''
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ========================== MATH ==============================

num_x = 102
num_y = 102
num_cells = num_x*num_y
dx = 0.05
dy = 0.05
C = 0.4
g = 9.81
dt = 0.1

huv = np.zeros((num_x, num_y, 3))
U = np.zeros((num_x, num_y, 3))
F = np.zeros((num_x, num_y, 3))
G = np.zeros((num_x, num_y, 3))

# del t = C * min(del x / (max(|u| + sqrt(gh), del y / (max(|v| + sqrt(gh))))))
def del_t_update():
    global dt
    s_x = np.max([[
            abs(huv[i][j][1])+math.sqrt(g*huv[i][j][0]) for j in range(1, num_y-1)
        ] for i in range(1, num_x-1)])
    s_y = np.max([[
            abs(huv[i][j][2])+math.sqrt(g*huv[i][j][0]) for j in range(1, num_y-1)
        ] for i in range(1, num_x-1)])
    dt = C * min(dx / s_x, dy / s_y)

# F_{i+1/2, j} = 0.5(F_{i,j} + F{i+1, j}) - 0.5 s_max*(U_{i+1, j} - U_{i, j})
def F_rusanov(i_1, i_2, j):
    new_F = np.zeros(3)
    s = s_max(i_1, j, i_2, j, 1, 0)
    new_F += 0.5*(F[i_1][j] + F[i_2][j])
    new_F -= 0.5*(U[i_2][j] - U[i_1][j])*s
    return new_F

# G_{i, j+1/2} = 0.5(F_{i,j} + F{i, j+1}) - 0.5 s_max*(U_{i, j+1} - U_{i, j})
def G_rusanov(i, j_1, j_2):
    new_G = np.zeros(3)
    s = s_max(i, j_1, i, j_2, 0, 1)
    new_G += 0.5*(G[i][j_1] + G[i][j_2])
    new_G -= 0.5*(U[i][j_2]-U[i][j_1])*s
    return new_G

# s_{max} = max(|u_L*n_x + v_L*n_y| + sqrt(g*h_L), |u_R*n_x + v_R*n_y| + sqrt(g*h_R))
def s_max(i_1, j_1, i_2, j_2, n_x, n_y):
    huv_L = huv[i_1][j_1]
    huv_R = huv[i_2][j_2]
    return max(
        abs(huv_L[1]*n_x + huv_L[2]*n_y) + math.sqrt(g*huv_L[0]),
        abs(huv_R[1]*n_x + huv_R[2]*n_y) + math.sqrt(g*huv_R[0])
    )

# U_{k+1} = U_k - (del t / del x)(F_{i+1/2, j} - F_{i-1/2, j}) - (del t / del y)(G_{i, j+1/2} - G_{i, j-1/2})
def U_update(i, j, F_diff, G_diff):
    U[i][j] -= (dt/dx)*(F_diff) + (dt/dy)*(G_diff)
    huv_update(i, j)

# huv = [h, u, v]
def huv_update(i, j):
    height = U[i][j][0]
    huv[i][j][0] = height
    huv[i][j][1] = U[i][j][1] / height
    huv[i][j][2] = U[i][j][2] / height

# F = [hu, hu^2 + 0.5gh^2, huv]
def F_update(i, j):
    h, u, v = huv[i][j]
    F[i, j] = [
        h*u,
        h*u*u + 0.5*g*h*h,
        h*u*v
    ]

# G = [hv, huv, hv^2 + 0.5gh^2]
def G_update(i, j):
    h, u, v = huv[i][j]
    G[i, j] = [
        h*v,
        h*u*v,
        h*v*v + 0.5*g*h*h
    ]

def boundary_update():
    for i in range(1, num_x-1):
        U[i][0][0] = U[i][1][0]
        U[i][0][1] = U[i][1][1]
        U[i][0][2] = -U[i][1][2]
        huv_update(i, 0)

        U[i][-1][0] = U[i][-2][0]
        U[i][-1][1] = U[i][-2][1]
        U[i][-1][2] = -U[i][-2][2]
        huv_update(i, -1)

    for i in range(1, num_y-1):
        U[0][i][0] = U[1][i][0]
        U[0][i][1] = -U[1][i][1]
        U[0][i][2] = U[1][i][2]
        huv_update(0, i)

        U[-1][i][0] = U[-2][i][0]
        U[-1][i][1] = -U[-2][i][1]
        U[-1][i][2] = U[-2][i][2]
        huv_update(-1, i)

def iteration():
    boundary_update()
    del_t_update()
    for i in range(num_x):
        for j in range(num_y):
            F_update(i, j)
            G_update(i, j)

    get_F_rus = [[F_rusanov(i, i+1, j) for j in range(num_y-1)] for i in range(num_x-1)]
    get_G_rus = [[G_rusanov(i, j, j+1) for j in range(num_y-1)] for i in range(num_x)]

    for i in range(1, num_x-1):
        for j in range(1, num_y-1):
            F_diff = get_F_rus[i][j] - get_F_rus[i-1][j]
            G_diff = get_G_rus[i][j] - get_G_rus[i][j-1]
            U_update(i, j, F_diff, G_diff)

def init():
    U[:, :, 0] = 2.0
    huv[:, :, 0] = 2.0

    cx, cy = 26, 26
    amplitude = 1.0
    sigma = 5.0

    for i in range(1, num_x - 1):
        for j in range(1, num_y - 1):
            r2 = (i - cx)**2 + (j - cy)**2
            h = 2.0 + amplitude * math.exp(-r2 / (2 * sigma**2))

            U[i, j, 0] = h
            huv[i, j, 0] = h

init()

# =========================== ANIMATION ============================

fig, ax = plt.subplots(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_zlim(-2, 2)
ax.set_box_aspect((5, 5, 4))
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

x = np.arange(0, 5, dx)
y = np.arange(0, 5, dy)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

surface = [ax.plot_surface(
    X, Y, Z,
    cmap='Blues',
    vmin=-1.0,
    vmax=1.0
)]

wall_x = np.array([[0, 5], [0, 5]])
wall_y = np.array([[0, 5], [0, 5]])
wall_z = np.array([[-2, 2], [-2, 2]])

ax.plot_surface(
    wall_x,
    wall_y,
    wall_z,
    alpha=0.3
)

fps = 35
frames = 500

def animate(i):
    if i > 10:
        iteration()

    Z = np.array([
        [huv[i][j][0] - 2.0 for j in range(1, num_y-1)] for i in range(1, num_x-1)
    ])

    surface[0].remove()

    surface[0] = ax.plot_surface(
        X, Y, Z,
        cmap='Blues',
        vmin=-1.0,
        vmax=1.0
    )

    return surface

anim = FuncAnimation(
    fig, animate, frames=frames, interval=1000/fps, blit=False
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-mg', '-g', '--make_gif',
    action="store_true",
)
args = parser.parse_args()

if args.make_gif:
    writer = PillowWriter(fps=25)
    anim.save("gifs/swe_2d.gif", writer=writer)

plt.show()
