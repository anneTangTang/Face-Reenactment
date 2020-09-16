import argparse
import json
import sys

import numpy as np
from numpyencoder import NumpyEncoder

sys.path.append("..")
from face3d import mesh_numpy
from face3d.morphable_model import MorphabelModel


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="The input json file containing all params.")
parser.add_argument("-s", "--save", type=str, help="The path to save mask.")
args = parser.parse_args()

# mask权重分布为：
# 眼、鼻、口：20
# 脸部其他：5
# 背景：1
h = w = 256
WEIGHT_BACKGROUND = 1
WEIGHT_OTHERS = 5
WEIGHT_EYE = 20
WEIGHT_NOSE = 20
WEIGHT_MOUTH = 20
WEIGHT = np.array([WEIGHT_BACKGROUND, WEIGHT_EYE, WEIGHT_MOUTH, WEIGHT_OTHERS, WEIGHT_NOSE], dtype=float)

bfm = MorphabelModel("Data/BFM.mat")

with open("Data/face_segments.json", "r") as file:
    face_segments = json.load(file)

with open(args.input, "r") as file:
    target_dict = json.load(file)

face_segments_3ddfa = np.array(face_segments["3ddfa"])  # ndarray: (53215,) value:1,2,3,4 -> 眼、口、其他、鼻

weights = []
for seg in face_segments_3ddfa:
    weights.append(int(WEIGHT[seg]))
weights = np.array(weights)  # ndarray: (53215,)

fitted_sp = np.array(target_dict["shape"])
weight_mask_dict = {}

for i in range(target_dict["total"]):
    img_name = "%05d.jpeg" % (i + 1)
    fitted_ep = np.array(target_dict[img_name]["exp"])
    fitted_R = np.array(target_dict[img_name]["R"])
    fitted_t3d = np.array(target_dict[img_name]["t3d"])
    fitted_s = target_dict[img_name]["scale"]

    # generate image
    fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)  # (nver, 3)
    fitted_angles = mesh_numpy.transform.matrix2angle(fitted_R)
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t3d)

    image_vertices = transformed_vertices
    image_vertices[:, 1] = h - transformed_vertices[:, 1] - 1

    weight_mask = mesh_numpy.render.render_mask(image_vertices, bfm.triangles, weights, h, w)  # ndarray: (h, w)
    weight_mask_dict[img_name] = weight_mask

with open(args.save, "w") as file:
    json.dump(weight_mask_dict, file, cls=NumpyEncoder)
