import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 1. Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 5)
ax.set_ylim(-2.2, 2.2)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True, linestyle='--', alpha=0.5)

# Define wall position
wall_pos = 5

# 2. Wave parameters
dx = 0.02
x = np.arange(0, 5, dx)
k = 2.0 * np.pi / 6        # Wavelength = 3
omega = 2.0 * np.pi        # Frequency = 1 Hz
v = omega / k              # Wave speed
A = 1                   # Amplitude

# 3. Initialize empty lines for animation
line_total, = ax.plot([], [], 'blue', lw=2.5)
fill = None
# Gaussian envelope function to form a distinct pulse/packet
default_width = 0.75
def envelope(pos, center, width=default_width):
    return np.exp(-((pos - center) / width)**2)

# Initialization function for FuncAnimation
def init():
    line_total.set_data([], [])
    return line_total,

# 4. Animation update loop
fps = 50
dt = 0.5 / fps
t_max = 2
frames = int(t_max / dt)

def animate(i):
    global fill

    t = i * dt
    
    # Progress center of the incident wave moving right
    pulse_center_inc = 1 + v*t
    y_inc = A*np.sin(k*x - omega*t)*envelope(x, pulse_center_inc)

    # Soft wall boundary = In-phase reflection (keeps the positive sign)
    pulse_center_ref = 2 * wall_pos - pulse_center_inc
    y_ref = A*np.sin(-k*x - omega*t + 2*k*wall_pos)*envelope(x, pulse_center_ref)
    
    # Enforce boundary physics (reflection doesn't exist deep before hitting the wall area)
    if pulse_center_inc < (wall_pos - 3*default_width):
        y_ref[:] = 0
        
    # The wave doesn't travel past the solid wall boundary
    y_inc[x > wall_pos] = 0
    y_ref[x > wall_pos] = 0
    
    # Superposition principle: total wave is the sum of both components
    y_total = y_inc + y_ref

    if fill is not None:
        fill.remove()

    fill = ax.fill_between(x, -2.2, y_total, color='blue', alpha=0.5)
    
    # Update lines
    line_total.set_data(x, y_total) 
    
    return line_total,

# 5. Create and run the animation
anim = FuncAnimation(
    fig, animate, init_func=init, frames=frames, interval=1000/fps, blit=False
)

writer = PillowWriter(fps=35)
anim.save("gifs/sine_refl.gif", writer=writer)

plt.show()
