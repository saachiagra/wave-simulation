import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 1. Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
ax = fig.add_subplot(projection='3d')
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_zlim(-1, 2.5)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Define wall position
wall_pos = 3

# 2. Wave parameters
dx = 0.02
x = np.arange(0, 3, dx)
dy = 0.02
y = np.arange(0, 3, dy)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

k = 2.0 * np.pi / 6        # Wavelength = 3
omega = 2.0 * np.pi        # Frequency = 1 Hz
v = omega / k              # Wave speed
A = 1                   # Amplitude

# 3. Initialize empty surface for animation
surface = [ax.plot_surface(
    X, Y, Z,
    cmap='Blues',
    vmin=-1,
    vmax=1
)]

wall_x = np.full((2, 2), wall_pos)
wall_y = np.array([[0, 3], [0, 3]])
wall_z = np.array([[-1, -1], [2.5, 2.5]])

ax.plot_surface(
    wall_x,
    wall_y,
    wall_z,
    alpha=0.3
)

# Gaussian envelope function to form a distinct pulse/packet
default_width = 0.75
def envelope(pos, center, width=default_width):
    return np.exp(-((pos - center) / width)**2)

# 4. Animation update loop
fps = 50
dt = 0.5 / fps
t_max = 1
frames = int(t_max / dt)

def animate(i):
    t = i * dt
    
    # Progress center of the incident wave moving right
    pulse_center_inc = 1 + v*t
    z_inc = A*np.sin(k*X - omega*t)*envelope(X, pulse_center_inc)

    # Soft wall boundary = In-phase reflection (keeps the positive sign)
    pulse_center_ref = 2 * wall_pos - pulse_center_inc
    z_ref = A*np.sin(-k*X - omega*t + 2*k*wall_pos)*envelope(X, pulse_center_ref)
    
    # Enforce boundary physics (reflection doesn't exist deep before hitting the wall area)
    if pulse_center_inc < (wall_pos - 3*default_width):
        z_ref[:] = 0
        
    # The wave doesn't travel past the solid wall boundary
    z_inc[x > wall_pos] = 0
    z_ref[x > wall_pos] = 0
    
    # Superposition principle: total wave is the sum of both components
    Z = z_inc + z_ref

    # Update lines
    surface[0].remove()

    # Draw new surface
    surface[0] = ax.plot_surface(
        X, Y, Z,
        cmap='Blues',
        vmin=-1,
        vmax=1
    )

    return surface

# 5. Create and run the animation
anim = FuncAnimation(
    fig, animate, frames=frames, interval=1000/fps, blit=False
)

writer = PillowWriter(fps=35)
anim.save("gifs/sine_refl_3d.gif", writer=writer)

plt.show()
