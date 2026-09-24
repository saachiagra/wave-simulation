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

num_cells = 1000    # not including boundary cells
cells = []          # array of tuples: (h_i, u_i)
U = []              # 
flux = []
dx = 0.1
C = 0.5
dt = 0.1            # TODO: change this!
g = 9.81            # gravity constant

def time_update():
    global dt
    get_s_max = max([abs(cells[i][1]) + math.sqrt(g*cells[i][0]) for i in range(1, num_cells)])
    dt = C * (dx / get_s_max)

def cell_update(i: int):
    F_L = get_mid_flux(i-1, i)
    F_R = get_mid_flux(i, i+1)

    U[i][0] -= (dt/dx)*(F_R[0]-F_L[0])
    U[i][1] -= (dt/dx)*(F_R[1]-F_L[1])

    cells[i][0] = U[i][0]
    cells[i][1] = U[i][1] / U[i][0]

def flux_update(i: int):
    h_i = cells[i][0]
    u_i = cells[i][1]
    hu_i = U[i][1]
    flux[i][0] = hu_i
    flux[i][1] = hu_i*u_i + 0.5*g*h_i**2

def get_mid_flux(l: int, r: int):
    get_s_max = s_max(l, r)
    F = [0, 0]
    F[0] = 0.5 * (flux[l][0] + flux[r][0]) - 0.5*get_s_max*(U[l][0]-U[r][0])
    F[1] = 0.5 * (flux[l][1] + flux[r][1]) - 0.5*get_s_max*(U[l][1]-U[r][1])
    return F

def s_max(l: int, r: int):
    h_L = cells[l][0]
    u_L = cells[l][1]

    h_R = cells[r][0]
    u_R = cells[r][1]

    # sqrt(g*h): shallow water wave speed
    return max(u_L + math.sqrt(g*h_L), u_R + math.sqrt(g*h_R))
