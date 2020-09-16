import argparse
import json
import os
import sys

import numpy as np
from skimage import io

sys.path.append("..")
from face3d import mesh_numpy
from face3d.morphable_model import MorphabelModel


# specify
parser = argparse.ArgumentParser(description="Render reconstructed images.")
parser.add_argument("-i", "--input", type=str, default="black4000.json", help="the path to read params")
parser.add_argument("-s", "--save", type=str, default="recons", help="the path to save reconstructed images")
args = parser.parse_args()

# load
bfm = MorphabelModel("Data/BFM.mat")
h = w = 256

with open(args.input, "r") as file:
    target_dict = json.load(file)
shape_para = np.array(target_dict["shape"])
tex_para = np.array(target_dict["tex"])
sh_light = np.array(target_dict["light"])
n_tp = target_dict["param"]["n_tp"]
total = target_dict["total"]

# render
for i in range(total):
    # if i > 9:
    #     break
    img_name = "%05d.jpeg" % (i + 1)
    # print(img_name)
    exp_para = np.array(target_dict[img_name]["exp"])
    rotation_matrix = np.array(target_dict[img_name]["R"])
    translation_3d = np.array(target_dict[img_name]["t3d"])

    vertices = bfm.generate_vertices(shape_para, exp_para)
    angles = mesh_numpy.transform.matrix2angle(rotation_matrix)
    vertices = bfm.transform(vertices, target_dict[img_name]["scale"], angles, translation_3d)
    image_vertices = vertices
    image_vertices[:, 1] = h - vertices[:, 1] - 1

    colors = bfm.model["texMU"] + bfm.model["texPC"][:, :n_tp].dot(tex_para * bfm.model["texEV"][:n_tp, :])
    colors = np.reshape(colors, [-1, 3])  # (n_ver, 3)
    colors = mesh_numpy.light.add_light_sh(image_vertices, bfm.triangles, colors, sh_light) / 255.0  # (n_ver, 3)
    colors = np.minimum(np.maximum(colors, 0), 1)  # 0~1

    image = mesh_numpy.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
    image = np.minimum(np.maximum(image, 0), 1)
    # ------------- save
    io.imsave(os.path.join(args.save, img_name), image)
