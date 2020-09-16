import argparse
import json
from typing import Any, Dict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--target", type=str, help="json file containing params of target")
parser.add_argument("-s", "--source", type=str, help="json file containing params of source")
parser.add_argument("--save", type=str, help="json file to save changed params of target")
parser.add_argument("--fix", type=bool, default=True, help="whether to fix the scale and t3d of target")
args = parser.parse_args()

# load
with open(args.target, "r") as file:
    target_dict = json.load(file)

with open(args.source, "r") as file:
    source_dict = json.load(file)

target_average_s = target_dict["average_s"]
source_average_s = source_dict["average_s"]
target_average_t3d = target_dict["average_t3d"]
source_average_t3d = source_dict["average_t3d"]

# calculate changed parameters
changed_dict: Dict[str, Any] = {
    "param": {
        "n_sp": target_dict["param"]["n_sp"],
        "n_ep": source_dict["param"]["n_ep"],
        "n_tp": target_dict["param"]["n_tp"],
        "light_by_channel": target_dict["param"]["light_by_channel"],
    },
    "shape": target_dict["shape"],
    "tex": target_dict["tex"],
    "light": target_dict["light"],
}

# change
for i in range(source_dict["total"]):
    img_name = "%05d.jpeg" % (i + 1)
    source_param = source_dict[img_name]

    if args.fix:
        changed_scale = target_average_s
        changed_t3d = target_average_t3d
    else:
        changed_scale = source_param["scale"] / source_average_s * target_average_s
        changed_t3d = (
            np.array(source_param["t3d"]) - np.array(source_average_t3d)
        ) / source_average_s * target_average_s + np.array(target_average_t3d).tolist()

    changed_dict[img_name] = {
        "scale": changed_scale,
        "R": source_param["R"],
        "t3d": changed_t3d,
        "exp": source_param["exp"],
    }

changed_dict["total"] = source_dict["total"]

with open(args.save, "w") as file:
    json.dump(changed_dict, file)
