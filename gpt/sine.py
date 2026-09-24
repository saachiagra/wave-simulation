import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Set up the figure and axis
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10)
ax.set_ylim(-2.2, 2.2)
ax.set_title("Sine Wave Packet Hitting a Soft Wall (In-Phase Reflection)", fontsize=12, fontweight='bold')
ax.set_xlabel("Position (x)")
ax.set_ylabel("Amplitude (y)")
ax.grid(True, linestyle='--', alpha=0.5)

# Define wall position
wall_pos = 7.0
ax.axvline(x=wall_pos, color='red', linestyle='--', lw=2, label='Soft Wall Boundary')
ax.fill_between([wall_pos, 10], -2.5, 2.5, color='red', alpha=0.1, label='Wall Zone')

# 2. Wave parameters
dx = 0.02
x = np.arange(0, 10, dx)
k = 2.0 * np.pi / 1.5      # Wavelength = 1.5
omega = 2.0 * np.pi * 1.0  # Frequency = 1 Hz
v = omega / k              # Wave speed

# 3. Initialize empty lines for animation
line_total, = ax.plot([], [], 'blue', lw=2.5, label='Total Observed Wave')
line_inc, = ax.plot([], [], 'green', linestyle=':', alpha=0.4, label='Incident Component')
line_ref, = ax.plot([], [], 'purple', linestyle=':', alpha=0.4, label='Reflected Component')
ax.legend(loc='upper left')

# Gaussian envelope function to form a distinct pulse/packet
def envelope(pos, center, width=1):
    return np.exp(-((pos - center) / width)**2)

# Initialization function for FuncAnimation
def init():
    line_total.set_data([], [])
    line_inc.set_data([], [])
    line_ref.set_data([], [])
    return line_total, line_inc, line_ref

# 4. Animation update loop
fps = 50
dt = 1.0 / fps
t_max = 10.0
frames = int(t_max / dt)

def animate(i):
    t = i * dt
    
    # Progress center of the incident wave moving right
    pulse_center_inc = 1.0 + v * t
    y_inc = np.sin(k * x - omega * t) * envelope(x, pulse_center_inc)
    
    # Mirror reflection center (incoming from the phantom side of the wall)
    pulse_center_ref = 2 * wall_pos - pulse_center_inc
    
    # Soft wall boundary = In-phase reflection (keeps the positive sign)
    y_ref = np.sin(-k * x - omega * t + 2 * k * wall_pos) * envelope(x, pulse_center_ref)
    
    # Enforce boundary physics (reflection doesn't exist deep before hitting the wall area)
    if pulse_center_inc < (wall_pos - 3.0):
        y_ref[:] = 0
        
    # The wave doesn't travel past the solid wall boundary
    y_inc[x > wall_pos] = 0
    y_ref[x > wall_pos] = 0
    
    # Superposition principle: total wave is the sum of both components
    y_total = y_inc + y_ref
    
    # Update lines
    line_inc.set_data(x, y_inc)
    line_ref.set_data(x, y_ref)
    line_total.set_data(x, y_total)
    
    return line_total, line_inc, line_ref

# 5. Create and run the animation
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=frames, interval=1000/fps, blit=True
)

plt.show()
