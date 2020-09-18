from __future__ import absolute_import, division, print_function

from typing import Any, Dict

import numpy as np
import scipy.io as sio


def load_BFM(model_path: str) -> Dict[str, Any]:
    """Load BFM 3DMM model.

    :param model_path: path to BFM model
    :return: a dictionary containing bfm data, here, n_ver = 53215 & n_tri = 105840
        {
            'shapeMU': (3*n_ver, 1)
            'shapePC': (3*n_ver, 199)
            'shapeEV': (199, 1)
            'expMU': (3*n_ver, 1)
            'expPC': (3*n_ver, 29)
            'expEV': (29, 1)
            'texMU': (3*n_ver, 1)
            'texPC': (3*n_ver, 199)
            'texEV': (199, 1)
            'tri': (n_tri, 3) (start from 1, should sub 1 in python and c++)
            'tri_mouth': (114, 3) (start from 1, as a supplement to mouth triangles)
            'kpt_ind': (68,) (start from 1)
        }

    PS:
        You can change codes according to your own saved data.
        Just make sure the model has corresponding attributes.
    """
    model = sio.loadmat(model_path)["model"][0, 0]

    # change dtype from double(np.float64) to np.float32,
    # since big matrix process(especially matrix dot) is too slow in python.
    model["shapeMU"] = (model["shapeMU"] + model["expMU"]).astype(np.float32)
    model["shapePC"] = model["shapePC"].astype(np.float32)
    model["shapeEV"] = model["shapeEV"].astype(np.float32)
    model["expEV"] = model["expEV"].astype(np.float32)
    model["expPC"] = model["expPC"].astype(np.float32)

    # matlab start with 1. change to 0 in python.
    model["tri"] = model["tri"].T.copy(order="C").astype(np.int32) - 1
    model["tri_mouth"] = model["tri_mouth"].T.copy(order="C").astype(np.int32) - 1
    model["kpt_ind"] = (np.squeeze(model["kpt_ind"]) - 1).astype(np.int32)

    return model
