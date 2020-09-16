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


def load_BFM_info(info_path: str = "BFM_info.mat") -> Dict[str, Any]:
    """Load 3DMM model extra information.

    :param info_path: path to BFM info
    :return: a dictionary containing bfm model info
        {
            'symlist': 2 x 26720
            'symlist_tri': 2 x 52937
            'segbin': 4 x n (0: nose, 1: eye, 2: mouth, 3: cheek)
            'segbin_tri': 4 x n_tri
            'tri': 3 x 105840
            'keypoints': 1 x 68
            'trimIndex': 53215 x 1 (may be the alignment from 53490 to 53215)

            listed in github/face3d but not found in BFM_info.mat:
            # 'face_contour': 1 x 28
            # 'face_contour_line': 1 x 512
            # 'face_contour_front': 1 x 28
            # 'face_contour_front_line': 1 x 512
            # 'nose_hole': 1 x 142
            # 'nose_hole_right': 1 x 71
            # 'nose_hole_left': 1 x 71
            # 'parallel': 17 x 1 cell
            # 'parallel_face_contour': 28 x 1 cell
            # 'uv_coords': n x 2
        }
    """
    return sio.loadmat(info_path)


def load_uv_coords(uv_path: str = "BFM_UV.mat") -> Any:
    """Load uv coords of BFM.

    :param uv_path: path to data
    :return: (n_ver, 2). range: 0-1
    """
    return sio.loadmat(uv_path)["UV"].copy(order="C")


def load_pncc_code(pncc_path: str = "pncc_code.mat") -> Any:
    """Load pncc code of BFM.
    PNCC code: Defined in 'Face Alignment Across Large Poses: A 3D Solution Xiangyu'
    download at http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm.

    :param pncc_path: path to data
    :return: (n_ver, 3)
    """
    return sio.loadmat(pncc_path)["vertex_code"].T
