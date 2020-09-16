import argparse
import json
import os

import numpy as np
from skimage import io
from PIL import Image

from eye_gaze import find_center, generate_eye_image


# specify
parser = argparse.ArgumentParser(description="Render eye images.")
parser.add_argument("-i", "--input", type=str, default="../../target/origin", help="target image path")
parser.add_argument("-l", "--landmark", type=str, default="../../target_landmark.json", help="landmark path")
parser.add_argument("-s", "--save", type=str, default="../../target/eye", help="the path to save eye image")
args = parser.parse_args()

# load
with open(args.landmark, "r") as file:
    kpt_dict = json.load(file)

h = w = 256
total = kpt_dict["total"]

# render
for i in range(total):
    img_name = "%05d.jpeg" % (i + 1)
    img = np.array(Image.open(os.path.join(args.input, img_name)))  # 0~255
    kpt_coor = np.array(kpt_dict[img_name])[36:48, :]
    left_center, right_center = find_center(img, kpt_coor)
    eye_image = generate_eye_image(kpt_coor, left_center, right_center, h, w)
    io.imsave(os.path.join(args.save, img_name), eye_image)
