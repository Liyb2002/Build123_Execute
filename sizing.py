from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Pnt
from OCC.Core.IFSelect import IFSelect_RetDone

# Load the STEP file
reader = STEPControl_Reader()
status = reader.ReadFile("original.step")

if status == IFSelect_RetDone:
    reader.TransferRoots()
    shape = reader.OneShape()

    # Create a scaling transformation (1000x scale)
    scale_factor = 1000.0
    center_point = gp_Pnt(0, 0, 0)  # Scaling about origin
    scale_transform = gp_Trsf()
    scale_transform.SetScale(center_point, scale_factor)

    # Apply the transformation
    transformer = BRepBuilderAPI_Transform(shape, scale_transform, True)
    scaled_shape = transformer.Shape()

    # Write to a new STEP file
    writer = STEPControl_Writer()
    writer.Transfer(scaled_shape, STEPControl_AsIs)  # Corrected this line
    writer.Write("output.step")
else:
    print("Error: Cannot read STEP file.")
