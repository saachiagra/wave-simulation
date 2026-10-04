import os
import numpy as np

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": False
})

import omni
from pxr import UsdGeom, Gf, UsdLux, UsdShade, Sdf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_FILE = os.path.join(SCRIPT_DIR, "elevation.f32bin")

NX = 151
NY = 101
DX = 2.0
DY = 2.0
FRAME_DT = 0.05

print("=" * 60)
print("Loading SWASH height data")
print("=" * 60)

file_size = os.path.getsize(BIN_FILE)

print(f"File: {BIN_FILE}")
print(f"File size: {file_size:,} bytes")

raw = np.fromfile(BIN_FILE, dtype=np.float32)

# Each frame contains:
#   2 extra float32 values
#   151 x 101 height values
VALUES_PER_FRAME = 2 + NX * NY

num_frames = len(raw) // VALUES_PER_FRAME
remainder = len(raw) % VALUES_PER_FRAME

print(f"Number of float32 values: {len(raw):,}")
print(f"Grid: {NX} x {NY}")
print(f"Values per frame: {VALUES_PER_FRAME:,}")
print(f"Complete frames: {num_frames}")
print(f"Leftover values: {remainder}")

raw = raw[:num_frames * VALUES_PER_FRAME]

# Reshape into frames
frames = raw.reshape(num_frames, VALUES_PER_FRAME)

# Discard the first 2 floats of every frame
height_data = frames[:, 2:].reshape(
    num_frames,
    NY,
    NX
)

print()
print("Height data shape:", height_data.shape)

print(
    "Overall height range:",
    float(height_data.min()),
    "to",
    float(height_data.max()),
    "meters"
)


# ============================================================
# Create USD stage
# ============================================================

stage = omni.usd.get_context().get_stage()

print()
print("Creating water mesh...")

mesh = UsdGeom.Mesh.Define(
    stage,
    "/World/Water"
)


# ============================================================
# Vertex positions
# ============================================================

def make_vertices(height):

    vertices = []

    for y in range(NY):
        for x in range(NX):

            px = (
                x - (NX - 1) / 2
            ) * DX

            py = (
                y - (NY - 1) / 2
            ) * DY

            pz = float(height[y, x])

            vertices.append(
                Gf.Vec3f(px, py, pz)
            )

    return vertices


vertices = make_vertices(
    height_data[0]
)


# ============================================================
# Mesh topology
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
            i3
        ])


mesh.CreatePointsAttr(
    vertices
)

mesh.CreateFaceVertexCountsAttr(
    face_vertex_counts
)

mesh.CreateFaceVertexIndicesAttr(
    face_vertex_indices
)


# ============================================================
# Water material
# ============================================================

print("Creating water material...")

material = UsdShade.Material.Define(
    stage,
    "/World/WaterMaterial"
)

shader = UsdShade.Shader.Define(
    stage,
    "/World/WaterMaterial/Shader"
)

shader.CreateIdAttr(
    "UsdPreviewSurface"
)

shader.CreateInput(
    "diffuseColor",
    Sdf.ValueTypeNames.Color3f
).Set(
    Gf.Vec3f(
        0.0,
        0.3,
        0.8
    )
)

shader.CreateInput(
    "roughness",
    Sdf.ValueTypeNames.Float
).Set(0.3)

surface_output = material.CreateSurfaceOutput()

surface_output.ConnectToSource(
    shader.ConnectableAPI(),
    "surface"
)

UsdShade.MaterialBindingAPI(mesh).Bind(
    material
)


# ============================================================
# Lighting
# ============================================================

light = UsdLux.DistantLight.Define(
    stage,
    "/World/Sun"
)

light.CreateIntensityAttr(
    3000
)


# ============================================================
# Animation
# ============================================================

print()
print("=" * 60)
print("Starting animation")
print("=" * 60)

frame = 0
time_accumulator = 0.0

timeline = omni.timeline.get_timeline_interface()

previous_time = timeline.get_current_time()

timeline.play()


while simulation_app.is_running():

    simulation_app.update()

    current_time = timeline.get_current_time()

    dt = current_time - previous_time

    previous_time = current_time

    time_accumulator += dt

    if time_accumulator >= FRAME_DT:

        time_accumulator -= FRAME_DT

        frame += 1

        if frame >= num_frames:
            frame = 0

        height = height_data[frame]

        vertices = make_vertices(
            height
        )

        mesh.GetPointsAttr().Set(
            vertices
        )

simulation_app.close()