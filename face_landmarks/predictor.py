#!/usr/bin/env python3
#
# Copyright AnniTang.
#
"""In this file, we detect 68 landmarks in one face in an image.
Usage: python predictor.py --input [folder_name] --save [save_json]
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import dlib

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", type=str, default="", help="A folder containing images which will be predicted",
)
parser.add_argument("-s", "--save", type=str, default="", help="The path to save output json file.")
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
landmark_dict: Dict[str, List[Tuple[int, int]]] = {}
total = 0

for filename in os.listdir(args.input):
    total += 1
    image = cv2.imread(os.path.join(args.input, filename))
    faces: List[Any] = detector(image, 1)
    if not faces:
        raise RuntimeError(f"There is no face detected in the image: {filename}!")
    shape = predictor(image, faces[0])
    landmark_dict[filename] = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

landmark_dict["total"] = total

with open(args.save, "w") as file:
    json.dump(landmark_dict, file)
