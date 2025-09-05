from build123d import *
from build123d import export_stl, export_step

with BuildPart() as part:
    # Step 1: base cuboid (length=0.5, width=1.0, height=0.2), min-corner at (0,0,0)
    Box(0.5, 1.0, 0.2, align=(Align.MIN, Align.MIN, Align.MIN))

    # Step 2: cut a cuboid on the right corner (touching the x=0.5 face)
    # Dimensions: length=0.1 (x), width=0.6 (y), height=0.2 (z)
    # Centered at (0.45, 0.30, 0.10)
    with Locations((0.45, 0.7, 0.10)):
        Box(0.1, 0.6, 0.2, mode=Mode.SUBTRACT)

    # Step 3: cut two cylindrical holes near the back edge (y ≈ 1.0)
    # Cylinders along Z with height=0.2, radius=0.08
    # Centers at (0.15, 0.90, 0.10) and (0.35, 0.90, 0.10)
    with Locations((0.2, 0.80, 0.10), (0.2, 0.30, 0.10)):
        Cylinder(radius=0.06, height=0.2, mode=Mode.SUBTRACT)

export_stl(part.part, "file.stl")
export_step(part.part, "file.step")
