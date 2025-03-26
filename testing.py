from build123d import *
from build123d import export_stl

length, width, thickness = 60.0, 60.0, 60.0

with BuildPart() as ex2:
    Box(length, width, thickness)




export_stl(ex2.part, "file.stl")
