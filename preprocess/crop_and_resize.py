import argparse
import os
from configparser import ConfigParser

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="the folder containing images to be cropped")
parser.add_argument("-f", "--file", type=str, help="the file containing crop info")
args = parser.parse_args()

# read crop info
config_parser = ConfigParser()
config_parser.read(args.file)
xmin = int(config_parser["Credentials"]["x"])
ymin = int(config_parser["Credentials"]["y"])
xmax = xmin + int(config_parser["Credentials"]["w"])
ymax = ymin + int(config_parser["Credentials"]["h"])

for filename in os.listdir(args.input):
    if filename.startswith("."):
        continue
    filepath = os.path.join(args.input, filename)
    cropped_img = Image.open(filepath).crop((xmin, ymin, xmax, ymax))
    resized_img = cropped_img.resize((256, 256))
    # cover origin images with resized ones (256, 256)
    resized_img.save(filepath)
