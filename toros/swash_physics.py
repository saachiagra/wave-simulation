from omni.isaac.kit import SimulationApp

# ============================================================
# Start Isaac Sim FIRST
# ============================================================

simulation_app = SimulationApp({
    "headless": False,
})

# ============================================================
# Imports after SimulationApp initialization
# ============================================================

import os
import numpy as np

from pxr import UsdGeom, Gf

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from isaacsim.core.experimental.prims import RigidPrim


# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

ELEVATION_FILE = os.path.join(
    DATA_DIR,
    "elevation.f32bin"
)

NX = 151
NY = 101

DX = 2.0
DY = 2.0

FRAME_DT = 0.05

# Two extra floats occur before every height frame
VALUES_PER_FRAME = 2 + NX * NY

RHO_WATER = 1000.0
G = 9.81

CUBE_SIZE = 2.0
CUBE_MASS = 1.0


# ============================================================
# Load SWASH data
# ============================================================

print("Loading SWASH data...")

raw = np.fromfile(
    ELEVATION_FILE,
    dtype=np.float32
)

num_frames = len(raw) // VALUES_PER_FRAME

print(f"Total floats: {len(raw)}")
print(f"Values per frame: {VALUES_PER_FRAME}")
print(f"Complete frames: {num_frames}")

if num_frames == 0:
    raise RuntimeError(
        "No complete SWASH frames found."
    )

# Ignore incomplete data at the end
raw = raw[:num_frames * VALUES_PER_FRAME]

# Shape:
#     (num_frames, 2 + NY*NX)
frames = raw.reshape(
    num_frames,
    VALUES_PER_FRAME
)

# Remove the two extra floats
height_data = frames[:, 2:]

# Shape:
#     (num_frames, NY, NX)
height_data = height_data.reshape(
    num_frames,
    NY,
    NX
)

print("SWASH data loaded.")
print(
    f"Height range: "
    f"{height_data.min():.3f} to "
    f"{height_data.max():.3f} m"
)


# ============================================================
# Create world
# ============================================================

world = World(
    stage_units_in_meters=1.0
)

stage = world.stage


# ============================================================
# Create water mesh
# ============================================================

water = UsdGeom.Mesh.Define(
    stage,
    "/World/Water"
)


# ============================================================
# Initial water surface
# ============================================================

height = height_data[0]

points = []

for y in range(NY):
    for x in range(NX):

        px = (x - (NX - 1) / 2) * DX
        py = (y - (NY - 1) / 2) * DY
        pz = float(height[y, x])

        points.append(
            Gf.Vec3f(
                px,
                py,
                pz
            )
        )

water.CreatePointsAttr(points)


# ============================================================
# Water mesh topology
# ============================================================

face_vertex_counts = []
face_vertex_indices = []

for y in range(NY - 1):
    for x in range(NX - 1):

        i0 = y * NX + x
        i1 = i0 + 1
        i2 = i0 + NX + 1
        i3 = i0 + NX

        face_vertex_counts.append(4)

        face_vertex_indices.extend([
            i0,
            i1,
            i2,
            i3,
        ])

water.CreateFaceVertexCountsAttr(
    face_vertex_counts
)

water.CreateFaceVertexIndicesAttr(
    face_vertex_indices
)


# ============================================================
# Create cube
# ============================================================

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=[0.0, 0.0, 2.0],
        size=CUBE_SIZE,
        mass=CUBE_MASS,
    )
)


# ============================================================
# Ground
# ============================================================

world.scene.add_default_ground_plane()


# ============================================================
# Initialize physics
# ============================================================

world.reset()

# Experimental RigidPrim API.
# This API does NOT use initialize().

rigid_cube = RigidPrim(
    "/World/Cube",
    masses=[CUBE_MASS],
)

print("Simulation starting...")


# ============================================================
# Simulation
# ============================================================

frame = 0

while simulation_app.is_running():

    world.step(render=True)

    if not world.is_playing():
        continue


    # ========================================================
    # Current SWASH frame
    # ========================================================

    current_height = height_data[frame]


    # ========================================================
    # Update water mesh
    # ========================================================

    points = []

    for y in range(NY):
        for x in range(NX):

            px = (x - (NX - 1) / 2) * DX
            py = (y - (NY - 1) / 2) * DY
            pz = float(current_height[y, x])

            points.append(
                Gf.Vec3f(
                    px,
                    py,
                    pz
                )
            )

    water.GetPointsAttr().Set(points)


    # ========================================================
    # Get cube position
    # ========================================================

    cube_position = cube.get_world_pose()[0]

    cube_x = float(cube_position[0])
    cube_y = float(cube_position[1])
    cube_z = float(cube_position[2])


    # ========================================================
    # Convert world position -> SWASH grid coordinates
    # ========================================================

    gx = (
        cube_x / DX
        + (NX - 1) / 2
    )

    gy = (
        cube_y / DY
        + (NY - 1) / 2
    )

    ix = int(round(gx))
    iy = int(round(gy))

    # Keep sampling inside the SWASH grid
    ix = max(0, min(NX - 1, ix))
    iy = max(0, min(NY - 1, iy))


    # ========================================================
    # Local water height
    # ========================================================

    water_z = float(
        current_height[iy, ix]
    )


    # ========================================================
    # Calculate submerged volume
    # ========================================================

    cube_bottom = (
        cube_z - CUBE_SIZE / 2
    )

    submerged_height = (
        water_z - cube_bottom
    )

    submerged_height = max(
        0.0,
        min(
            CUBE_SIZE,
            submerged_height
        )
    )

    submerged_fraction = (
        submerged_height / CUBE_SIZE
    )

    cube_volume = CUBE_SIZE ** 3

    submerged_volume = (
        cube_volume
        * submerged_fraction
    )


    # ========================================================
    # Buoyant force
    # ========================================================

    buoyant_force = (
        RHO_WATER
        * G
        * submerged_volume
    )


    # ========================================================
    # Apply buoyancy
    # ========================================================

    if buoyant_force > 0.0:

        rigid_cube.apply_forces(
            np.array([
                [
                    0.0,
                    0.0,
                    buoyant_force
                ]
            ])
        )


    # ========================================================
    # Advance SWASH frame
    # ========================================================

    frame += 1

    if frame >= num_frames:
        frame = 0


# ============================================================
# Shutdown
# ============================================================

simulation_app.close()