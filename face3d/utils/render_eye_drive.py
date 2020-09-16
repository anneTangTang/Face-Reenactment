import argparse
import json
import os

import numpy as np
from skimage import io
from PIL import Image

from eye_gaze import find_center, generate_eye_image


# specify
parser = argparse.ArgumentParser(description="Render eye images of driven target.")
parser.add_argument("--source", type=str, help="folder containing source images")
parser.add_argument("--landmark_s", type=str, help="landmark file of source")
parser.add_argument("--landmark_t", type=str, help="landmark file of driven target")
parser.add_argument("--save", type=str, help="the path to save rendered eye image")
args = parser.parse_args()

# load
with open(args.landmark_s, "r") as file:
    landmark_s = json.load(file)

with open(args.landmark_t, "r") as file:
    landmark_t = json.load(file)

h = w = 256
# render
for i in range(landmark_s["total"]):
    # if i > 9:
    #     break
    img_name = "%05d.jpeg" % (i + 1)
    img = np.array(Image.open(os.path.join(args.source, img_name)))  # 0~255
    kpt_coor_s = np.array(landmark_s[img_name])[36:48, :]
    left_center, right_center = find_center(img, kpt_coor_s)
    kpt_coor_t = np.array(landmark_t[img_name])[36:48, :]
    eye_image = generate_eye_image(kpt_coor_t, left_center, right_center, h, w)
    io.imsave(os.path.join(args.save, img_name), eye_image)
