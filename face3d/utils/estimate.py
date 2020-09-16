"""Fit a BFM model to the given image."""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.append("..")
from face3d.morphable_model import Fit, MorphabelModel


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="the folder containing images to be estimated")
parser.add_argument("-l", "--landmark", type=str, help="the json file containing landmark info")
parser.add_argument("-s", "--save", type=str, help="json file to save the results")
parser.add_argument("--texture", type=bool, default=True, help="whether estimate texture and lighting, too")

parser.add_argument("--n_sp", type=int, default=199)
parser.add_argument("--n_tp", type=int, default=80)
parser.add_argument("--iter", type=int, default=10)
parser.add_argument("--light_by_channel", type=bool, default=True)

args = parser.parse_args()

# init
bfm = MorphabelModel("Data/BFM.mat")
fit = Fit(bfm, max_iter=args.iter)

# record configs
video_dict = {
    "param": {
        "n_sp": args.n_sp,
        "n_ep": 29,
        "n_tp": args.n_tp,
        "iter": args.iter,
        "light_by_channel": args.light_by_channel,
    }
}

# load data
with open(args.landmark, "r") as file:
    kpt_dict = json.load(file)

# fit
is_first = True
total = 0
total_scale = 0
total_t3d = np.zeros(3)

if args.texture:
    for filename in os.listdir(args.input):
        total += 1
        # print(total)
        image = np.array(Image.open(os.path.join(args.input, filename)))[:, :, :3]
        kpt_coor = np.array(kpt_dict[filename])
        kpt_coor[:, 1] = image.shape[0] - kpt_coor[:, 1] - 1  # flip landmarks along y-axis.

        if is_first:
            fitted_sp, fitted_ep, fitted_s, fitted_R, fitted_t3d, fitted_tp, fitted_sh = fit.fit(
                image, kpt_coor, args.n_sp, args.n_tp, args.light_by_channel
            )
            video_dict["shape"] = fitted_sp.tolist()
            video_dict["tex"] = fitted_tp.tolist()
            video_dict["light"] = fitted_sh.tolist()
            is_first = False
        else:
            fitted_ep, fitted_s, fitted_R, fitted_t3d = fit.fit_exp_pose(kpt_coor, np.array(video_dict["shape"]))
        total_scale += fitted_s
        total_t3d += fitted_t3d

        video_dict[filename] = {
            "scale": fitted_s,
            "R": fitted_R.tolist(),
            "t3d": fitted_t3d.tolist(),
            "exp": fitted_ep.tolist(),
        }
else:
    for filename in os.listdir(args.input):
        total += 1
        # print(total)
        image = np.array(Image.open(os.path.join(args.input, filename)))[:, :, :3]
        kpt_coor = np.array(kpt_dict[filename])
        kpt_coor[:, 1] = image.shape[0] - kpt_coor[:, 1] - 1  # flip landmarks along y-axis.

        if is_first:
            fitted_sp, fitted_ep, fitted_s, fitted_R, fitted_t3d = fit.fit_geometry_pose(kpt_coor, args.n_sp)
            video_dict["shape"] = fitted_sp.tolist()
            is_first = False
        else:
            fitted_ep, fitted_s, fitted_R, fitted_t3d = fit.fit_exp_pose(kpt_coor, np.array(video_dict["shape"]))
        total_scale += fitted_s
        total_t3d += fitted_t3d

        video_dict[filename] = {
            "scale": fitted_s,
            "R": fitted_R.tolist(),
            "t3d": fitted_t3d.tolist(),
            "exp": fitted_ep.tolist(),
        }

video_dict["total"] = total
video_dict["average_s"] = total_scale / total
video_dict["average_t3d"] = (total_t3d / total).tolist()

with open(args.save, "w") as file:
    json.dump(video_dict, file)
