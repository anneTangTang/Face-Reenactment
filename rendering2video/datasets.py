import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Load data from a root folder.
    Load ground truth images from root/origin,
    Load reconstruction images from root/recons,
    Load eye images from root/eye.
    Load mask from root/mask.json.

    :param root: the root folder contains origin, recons, eye and mask.json
    :param mode: "train" or "val" or "test"
    :param total: total number of images
    """

    def __init__(self, root: str, mode: str, total: int):
        self._root = root
        self._mode = mode
        self._total = total
        self._masks = json.load(open(os.path.join(self._root, "mask.json"), "r")) if self._mode == "train" else None

    def __getitem__(self, index):
        """ Get an item by index.

        :param index: from 0 to len(self)-1
        :return: a dictionary contains input and ground_truth
        """
        img_name = "%05d.jpeg" % (index + 1)
        if self._mode in ["train", "val"]:
            gt_path = os.path.join(self._root, "origin", img_name)
            ground_truth = np.array(Image.open(gt_path)) / 255 * 2 - 1  # (h,w,3). np. (-1, 1)
            ground_truth = np.swapaxes(ground_truth, 0, 2)  # (3,w,h). np. (-1, 1)
            ground_truth = np.swapaxes(ground_truth, 1, 2)  # (3,h,w). np. (-1, 1)
            ground_truth = torch.from_numpy(ground_truth)  # (3,h,w). tensor. range:(-1,1)
            if self._mode == "train":
                mask = np.array(self._masks[img_name])  # numpy array, (h, w)
                mask = np.expand_dims(mask, 0).repeat(3, axis=0)  # numpy array, (3, h, w)
                mask = torch.from_numpy(mask)  # (3,h,w). tensor. range:(1,20)
        else:
            ground_truth = []

        eye_path = os.path.join(self._root, "eye", img_name)
        recons_path = os.path.join(self._root, "recons", img_name)
        eye = np.array(Image.open(eye_path))  # (h,w,3). np. (0,255)
        recons = np.array(Image.open(recons_path))  # (h,w,3). np. (0,255)
        frame = np.concatenate((recons, eye), axis=2)  # (h,w,6),np,(0,255)
        frame = np.swapaxes(frame, 0, 2)  # (9Nw,w,h). np. range:(0,255)
        frame = np.swapaxes(frame, 1, 2)  # (9Nw,h,w). np. range:(0,255)
        frame = frame / 255 * 2 - 1  # (9Nw,h,w). np. range:(-1,1)
        frame = torch.from_numpy(frame)  # (9Nw,h,w). tensor. range:(-1,1)
        if self._mode == "train":
            return {"input": frame, "ground_truth": ground_truth, "mask": mask}
        return {"input": frame, "ground_truth": ground_truth}

    def __len__(self):
        return self._total
