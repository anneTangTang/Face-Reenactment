from __future__ import absolute_import, division, print_function

from typing import Any, Sequence

import numpy as np

from .. import mesh_numpy
from . import load


class MorphabelModel:
    """This class defines some operations on the morphable model.
    
    :param model_path: path to model
    :param model_type: the type of morphable model. Only support BFM now.
    """

    def __init__(self, model_path: str, model_type: str = "BFM") -> None:
        if model_type != "BFM":
            raise RuntimeError("Sorry, not support other 3DMM model now!")
        
        self.model = load.load_BFM(model_path)
        self.n_ver = self.model["shapePC"].shape[0] / 3
        self.n_tri = self.model["tri"].shape[0]
        self.n_shape_basis = self.model["shapePC"].shape[1]
        self.n_exp_basis = self.model["expPC"].shape[1]
        self.n_tex_basis = self.model["texPC"].shape[1]

        self.kpt_ind = self.model["kpt_ind"]
        self.triangles = self.model["tri"]
        self.full_triangles = np.vstack((self.model["tri"], self.model["tri_mouth"]))

    def get_valid_index(self):
        valid_index = np.tile(self.kpt_ind[np.newaxis, :], [3, 1]) * 3
        valid_index[1, :] += 1
        valid_index[2, :] += 2
        return valid_index.flatten("F")

    def get_shape_para(self, type: str = "random") -> Any:
        if type not in ["random", "zero"]:
            raise RuntimeError("Please enter the right type of shape para.")

        if type == "zero":
            return np.random.zeros((self.n_shape_basis, 1))
        return np.random.rand(self.n_shape_basis, 1) * 1e04

    def get_exp_para(self, type: str = "random") -> Any:
        if type not in ["random", "zero"]:
            raise RuntimeError("Please enter the right type of shape para.")

        if type == "zero":
            return np.zeros((self.n_exp_basis, 1))
        ep = -1.5 + 3 * np.random.random([self.n_exp_basis, 1])
        ep[6:, 0] = 0
        return ep

    def generate_vertices(self, shape_para: Any, exp_para: Any) -> Any:
        """Generate vertices according to shape and expression parameters.

        :param shape_para: (n_sp, 1)
        :param exp_para: (n_ep, 1)
        :return: (n_ver, 3)
        """
        n_sp = shape_para.shape[0]
        n_ep = exp_para.shape[0]
        vertices = (
            self.model["shapeMU"]
            + self.model["shapePC"][:, :n_sp].dot(shape_para)
            + self.model["expPC"][:, :n_ep].dot(exp_para)
        )
        return np.reshape(vertices, [-1, 3])

    # texture: here represented with rgb value(colors) in vertices.
    def get_tex_para(self, type: str = "random") -> Any:
        if type not in ["random", "zero"]:
            raise RuntimeError("Please enter the right type of shape para.")

        if type == "zero":
            return np.zeros((self.n_tex_basis, 1))
        return np.random.rand(self.n_tex_basis, 1)

    def generate_colors(self, tex_para: Any) -> Any:
        """Generate colors of all vertices according to texture parameters.

        :param tex_para: (n_tp, 1)
        :return: (n_ver, 3). range 0~1
        """
        colors = self.model["texMU"] + self.model["texPC"].dot(tex_para * self.model["texEV"])
        return np.reshape(colors, [int(3), int(len(colors) / 3)], "F").T / 255.0

    # transform
    def rotate(self, vertices: Any, angles: Sequence[float]) -> Any:
        """Rotate vertices.

        :param vertices: (n_ver, 3)
        :param angles: (3,). x, y, z rotation angle(degree)
            x: pitch. positive for looking down
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        :return: (n_ver, 3). rotated vertices
        """
        return mesh_numpy.transform.rotate(vertices, angles)

    def transform(self, vertices: Any, scale: float, angles: Sequence[float], translation_3d: Sequence[float], is_3ddfa: bool = False) -> Any:
        """Transform vertices through rotation, scale and translation.

        :param vertices: (n_ver, 3)
        :param scale: <float>
        :param angles: (3,). x, y, z rotation angle(degree)
        :param translation_3d: (3,).
        :param is_3ddfa: True only when processing 300W_LP data
        :return: transformed vertices
        """
        rotation_matrix = mesh_numpy.transform.angle2matrix(angles, is_3ddfa=is_3ddfa)
        return mesh_numpy.transform.similarity_transform(vertices, scale, rotation_matrix, translation_3d)
