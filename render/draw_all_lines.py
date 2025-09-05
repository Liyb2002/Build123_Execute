from pathlib import Path

import brep_read
import numpy as np
import os

import helper


current_folder = Path.cwd().parent
filename = current_folder / "file.step"
edge_features_list, cylinder_features_list = brep_read.sample_strokes_from_step_file(str(filename))  # STEP reader expects a str
feature_lines = edge_features_list + cylinder_features_list

for c in cylinder_features_list:
    print("len", len(c))
brep_read.vis_stroke_node_features(np.array(edge_features_list + cylinder_features_list))