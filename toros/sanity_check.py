from omni.isaac.kit import SimulationApp

# Start Isaac Sim
simulation_app = SimulationApp({
    "headless": False,
})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid


# Create world
world = World(stage_units_in_meters=1.0)

# Create a cube 5 meters above the ground
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=[0.0, 0.0, 5.0],
        size=1.0,
        mass=1.0,
    )
)

# Ground plane
world.scene.add_default_ground_plane()

# Initialize physics
world.reset()

print("Starting simulation...")

for i in range(300):
    world.step(render=True)

    if i % 30 == 0:
        position = cube.get_local_pose()[0]
        print(f"Step {i}: cube Z = {position[2]:.3f}")

print("Simulation finished.")

simulation_app.close()