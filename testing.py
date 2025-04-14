from build123d import *
from build123d import export_stl

with BuildPart() as part:
    # Step 1: Make a cylinder by extruding a circle
    with BuildSketch():
        Circle(10)
    extrude(amount=20)  # cylinder height is 20

    # Step 2: Sketch a rectangle on the side face (we pick one of the vertical faces)
    side_face = part.faces().filter_by(Axis.Z)[0]



    with BuildSketch(Plane(side_face)):
        Rectangle(10, 10, align=(Align.CENTER, Align.CENTER))

    extrude(amount=5, mode=Mode.SUBTRACT)


    with BuildSketch(Plane(side_face)):
        Rectangle(10, 10, align=(Align.CENTER, Align.CENTER))
    
    # Step 3: Extrude the rectangle outward by 5mm
    extrude(amount=5, mode=Mode.ADD)


export_stl(part.part, "file.stl")
export_step(part.part, "file.step")
