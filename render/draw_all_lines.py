from pathlib import Path

import brep_read
import numpy as np
import os

import helper
import line_utils

current_folder = Path.cwd().parent
filename = current_folder / "file.step"
edge_features_list, cylinder_features_list = brep_read.sample_strokes_from_step_file(str(filename))  # STEP reader expects a str
feature_lines = edge_features_list + cylinder_features_list

projection_line = line_utils.projection_lines(feature_lines)
bounding_box_line = line_utils.bounding_box_lines(feature_lines)

brep_read.vis_stroke_node_features_and_constructions(np.array(edge_features_list + cylinder_features_list), np.array(projection_line + bounding_box_line))
helper.save_strokes(current_folder, feature_lines, projection_line+bounding_box_line)
