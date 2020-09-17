#!/usr/bin/env python3
# Author: AnniTang

"""In this file, we write 68 landmarks in one face into a json file.
Usage: python predictor.py --input [drived_json] --save [save_json]
"""

import argparse
import json
import sys

import numpy as np
from typing import Dict, List

sys.path.insert(0, "../face3d")
from face3d import mesh_numpy
from face3d.morphable_model import MorphabelModel


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="json file containing params of driven target")
parser.add_argument("-s", "--save", type=str, help="json file to save landmarks of driven target")
args = parser.parse_args()

with open(args.input, "r") as file:
    drive_dict = json.load(file)

height = 256
shape_para = np.array(drive_dict["shape"])
bfm = MorphabelModel("../face3d/utils/Data/BFM.mat")
landmark_dict: Dict[str, List[List[int]]] = {}

for i in range(drive_dict["total"]):
    filename = "%05d.jpeg" % (i + 1)
    params = drive_dict[filename]
    exp_para = np.array(params["exp"])
    rotation_matrix = np.array(params["R"])
    translation_3d = np.array(params["t3d"])

    vertices = bfm.generate_vertices(shape_para, exp_para)
    angles = mesh_numpy.transform.matrix2angle(rotation_matrix)
    vertices = bfm.transform(vertices, params["scale"], angles, translation_3d)
    image_vertices = vertices
    image_vertices[:, 1] = height - vertices[:, 1] - 1  # ndarray: (n_ver, 3)
    points = image_vertices[bfm.kpt_ind, :2].astype(int)  # ndarray: (68, 2), int
    landmark_dict[filename] = points.tolist()

with open(args.save, "w") as file:
    json.dump(landmark_dict, file)
