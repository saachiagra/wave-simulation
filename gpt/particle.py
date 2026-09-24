import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
WALL_X = 10.0      # Position of the right wall
START_X = 0.0      # Starting position of the point
START_VX = 0.2     # Initial horizontal velocity
DT = 1.0           # Time step per frame
START_Y = 0
START_VY = 0.5

# --- Setup Plot Elements ---
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(-1, WALL_X + 2)
ax.set_ylim(-3, 3)

# Draw the right wall
ax.axvline(x=WALL_X, color='red', linestyle='--', linewidth=2, label='Right Wall')
ax.axvline(x=START_X, color='red', linestyle='--', linewidth=2, label='Left Wall')


# Initialize the 1D point (y is kept at 0)
point, = ax.plot([START_X], [0], 'bo', markersize=10, label='Point')

ax.set_title("1D Point Bouncing Off Right Wall")
ax.get_yaxis().set_visible(False) # Hide y-axis since it's 1D

# --- State Variables ---
# Using a list or dict allows us to mutate the values inside the animate function
state = {
    'x': START_X,
    'vx': START_VX,
    'y': START_Y,
    'vy': START_VY
}

# --- Animation Loop ---
def update(frame):
    time = frame * 0.5
    # Move the point forward
    state['x'] += state['vx'] * DT
    state['y'] += START_VY*np.sin(time)
    
    # Check for collision with the right wall
    if state['x'] >= WALL_X:
        state['x'] = WALL_X - (state['x'] - WALL_X)  # Perfect elastic reflection
        state['vx'] = -state['vx']                   # Reverse velocity
        state['vy'] = -state['vy']
        
    # Optional: Bounce off a left wall at x=0 to keep it looping indefinitely
    elif state['x'] <= 0:
        state['x'] = -state['x']
        state['vx'] = -state['vx']
        state['vy'] = -state['vy']

    # Update the plot data
    point.set_data([state['x']], [state['y']])
    return point,

# Create the animation
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=200, 
    interval=30, 
    blit=True
)

plt.show()
