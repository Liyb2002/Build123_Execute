from build123d import *
from build123d import export_stl, export_step

with BuildPart() as part:
    # Step 1: Make cuboid with height=0.2, length=0.5, width=1.0
    with BuildPart() as base:
        Box(0.5, 1.0, 0.2, align=(Align.MIN, Align.MIN, Align.MIN))

    # Step 2: Cut a cuboid on the right corner, height=0.2, length=0.1, width=0.6
    with BuildPart(part.part, mode=Mode.SUBTRACT):
        Box(0.1, 0.6, 0.2, align=(Align.MAX, Align.MIN, Align.MIN))

    # Step 3: Cut two cylinders (as holes) near the back, height=0.2, radius=0.08
    with BuildPart(part.part, mode=Mode.SUBTRACT):
        # First hole
        Cylinder(radius=0.08, height=0.2, align=(Align.CENTER, Align.MIN, Align.MIN),
                 mode=Mode.SUBTRACT).locate(Location((0.15, 0.9, 0)))
        # Second hole
        Cylinder(radius=0.08, height=0.2, align=(Align.CENTER, Align.MIN, Align.MIN),
                 mode=Mode.SUBTRACT).locate(Location((0.35, 0.9, 0)))

export_stl(part.part, "file.stl")
export_step(part.part, "file.step")
