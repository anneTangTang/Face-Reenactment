"""
Estimating parameters about vertices: shape para, exp para, pose para(s, R, t)
"""
from typing import Any, Sequence, Tuple

import numpy as np

from .. import mesh_numpy
from .morphabel_model import MorphabelModel


""" TODO: a clear document. 
Given: image_points, 3D Model
Estimate: shape, expression, pose

Inference: 

    projected_vertices = s*P*R(mu + shape + exp) + t2d  --> image_points
    s*P*R*shape + s*P*R(mu + exp) + t2d --> image_points

    # Define:
    X = vertices
    x_hat = projected_vertices
    x = image_points
    A = s*P*R
    b = s*P*R(mu + exp) + t2d
    ==>
    x_hat = A*shape + b  (2 x n)

    A*shape (2 x n)
    shape = reshape(shape_pc * sp) (3 x n)
    shape_pc*sp : (3n x 1)

    * flatten:
    x_hat_flatten = A*shape + b_flatten  (2n x 1)
    A*shape (2n x 1)
    --> A*shape_pc (2n x 199)  sp: 199 x 1
    
    # Define:
    pc_2d = A* reshape(shape_pc)
    pc_2d_flatten = flatten(pc_2d) (2n x 199)

    =====>
    x_hat_flatten = pc_2d_flatten * sp + b_flatten ---> x_flatten (2n x 1)

    Goals:
    (ignore flatten, pc_2d-->pc)
    min E = || x_hat - x || + lambda*sum(sp/sigma)^2
          = || pc * sp + b - x || + lambda*sum(sp/sigma)^2

    Solve:
    d(E)/d(sp) = 0
    2 * pc' * (pc * sp + b - x) + 2 * lambda * sp / (sigma' * sigma) = 0

    Get:
    (pc' * pc + lambda / (sigma'* sigma)) * sp  = pc' * (x - b)

"""


class Fit:
    """This class defines some methods to fit parameters of a 3d morphable model.

    :param model: the model to be fitted
    :param max_iter: the number of iterations to update parameters
    """

    def __init__(self, model: MorphabelModel, max_iter: int) -> None:
        self._model = model
        self._max_iter = max_iter

    def estimate_shape(
        self,
        kpt_coor: Any,
        expression: Any,
        scale: float,
        rotation_matrix: Any,
        translation_2d: Sequence[float],
        lamb: float = 3000,
    ) -> Any:
        """Estimate shape parameters given expression and pose.
        Reference: key_points

        :param kpt_coor: (2, 68). the coordinate of face keypoints
        :param expression: (3, 68)
        :param scale: <float>
        :param rotation_matrix: (3, 3)
        :param translation_2d: (2,)
        :param lamb: regulation coefficient
        :return: (199, 1). shape parameters
        """
        kpt_coor = kpt_coor.copy()

        valid_ind = self._model.get_valid_index()
        shape_mu = self._model.model["shapeMU"][valid_ind, :]  # (3x68, 1)
        shape_pc = self._model.model["shapePC"][valid_ind, :199]  # (3x68, 199)
        shape_ev = self._model.model["shapeEV"][:199, :]  # (199, 1)
        assert (
            shape_mu.shape[0] == shape_pc.shape[0]
        ), "shapeMU and shapePC should have the same number of vertexes"
        assert shape_mu.shape[0] == 3 * kpt_coor.shape[1]

        dof = shape_pc.shape[1]  # the number of shape basis -> 199
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        A = scale * P.dot(rotation_matrix)  # (2, 3)

        # --- calc pc
        pc_3d = np.resize(shape_pc.T, [dof, 68, 3])  # (199, 68, 3)
        pc_3d = np.reshape(pc_3d, [dof * 68, 3])  # (199x68, 3)
        pc_2d = pc_3d.dot(A.T.copy())  # (199x68, 2)
        pc = np.reshape(pc_2d, [dof, -1]).T  # (2x68, 199)

        # --- calc b
        # shape_mu
        mu_3d = np.resize(shape_mu, [68, 3]).T  # (3, 68)
        # expression
        b = A.dot(mu_3d + expression) + np.tile(
            np.array(translation_2d)[:, np.newaxis], [1, 68]
        )  # (2, 68)
        b = np.reshape(b.T, [-1, 1])  # (2x68, 1)

        # --- solve
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / shape_ev ** 2)
        kpt_coor = np.reshape(kpt_coor.T, [-1, 1])
        equation_right = np.dot(pc.T, kpt_coor - b)
        return np.dot(np.linalg.inv(equation_left), equation_right)

    def estimate_expression(
        self,
        kpt_coor: Any,
        shape: Any,
        scale: float,
        rotation_matrix: Any,
        translation_2d: Any,
        lamb: float = 2000,
    ) -> Any:
        """Estimate expression parameters given shape and pose.
        Reference: key_points

        :param kpt_coor: (2, 68). the coordinate of face keypoints
        :param shape: (3, 68)
        :param scale: <float>
        :param rotation_matrix: (3, 3)
        :param translation_2d: (2,)
        :param lamb: regulation coefficient
        :return: (29, 1). expression parameters
        """
        kpt_coor = kpt_coor.copy()

        valid_ind = self._model.get_valid_index()
        shape_mu = self._model.model["shapeMU"][valid_ind, :]  # (3x68, 1)
        exp_pc = self._model.model["expPC"][valid_ind, :29]  # (3x68, 29)
        exp_ev = self._model.model["expEV"][:29, :]  # (29, 1)
        assert shape_mu.shape[0] == exp_pc.shape[0]
        assert shape_mu.shape[0] == kpt_coor.shape[1] * 3

        dof = exp_pc.shape[1]  # the number of expression basis -> 29
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        A = scale * P.dot(rotation_matrix)  # (2, 3)

        # --- calc pc
        pc_3d = np.resize(exp_pc.T, [dof, 68, 3])  # (29, 68, 3)
        pc_3d = np.reshape(pc_3d, [dof * 68, 3])  # (29x68, 3)
        pc_2d = pc_3d.dot(A.T)  # (29x68, 2)
        pc = np.reshape(pc_2d, [dof, -1]).T  # (2x68, 29)

        # --- calc b
        # shape_mu
        mu_3d = np.resize(shape_mu, [68, 3]).T  # (3, 68)
        # expression
        b = A.dot(mu_3d + shape) + np.tile(
            np.array(translation_2d)[:, np.newaxis], [1, 68]
        )  # (2, 68)
        b = np.reshape(b.T, [-1, 1])  # (2x68, 1)

        # --- solve
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / exp_ev ** 2)
        kpt_coor = np.reshape(kpt_coor.T, [-1, 1])
        equation_right = np.dot(pc.T, kpt_coor - b)
        return np.dot(np.linalg.inv(equation_left), equation_right)

    def estimate_texture(
        self, image: Any, vertices: Any, n_tp: int, sh_light: Any
    ) -> Any:
        """Use least square method to fit texture parameters.

        :param image: (h, w, 3), range 0~255
        :param vertices: (n_ver, 3). 沿y轴翻转
        :param n_tp: the number of texture parameters
        :param sh_light: (9, 3) or (9, 1). spherical harmonic lighting parameters
        :return: (n_tp, 1). texture parameters
        """

        if sh_light.shape[1] == 1:
            sh_light = np.tile(sh_light, (1, 3))  # (9, 3)

        tex_mu = np.reshape(self._model.model["texMU"], [-1, 3])  # (n_ver, 3)
        tex_pc = np.reshape(self._model.model["texPC"][:, :n_tp], [-1, 3, n_tp])  # (n_ver, 3, n_tp)
        tex_ev = self._model.model["texEV"][:n_tp, :]  # (n_tp, 1)
        triangles = self._model.triangles  # (n_tri, 3)

        # calculate spherical harmonic basis function: sh_basis
        normal = mesh_numpy.light.get_normal(vertices, triangles)  # (n_ver, 3)
        sh_basis = np.array(
            (
                np.ones(vertices.shape[0]),
                normal[:, 0],
                normal[:, 1],
                normal[:, 2],
                normal[:, 0] * normal[:, 1],
                normal[:, 0] * normal[:, 2],
                normal[:, 1] * normal[:, 2],
                normal[:, 0] ** 2 - normal[:, 1] ** 2,
                3 * (normal[:, 2] ** 2) - 1,
            )
        ).T  # (n_ver, 9)

        coefficient_matrix = []
        result_vector = []
        for i in range(triangles.shape[0]):
            tri = triangles[i, :]
            # the inner bounding box
            u_min = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
            u_max = min(int(np.floor(np.max(vertices[tri, 0]))), image.shape[1] - 1)
            v_min = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
            v_max = min(int(np.floor(np.max(vertices[tri, 1]))), image.shape[0] - 1)
            if u_max < u_min or v_max < v_min:
                continue

            sh0 = sh_basis[tri[0]][np.newaxis, :]  # (1,9)
            sh1 = sh_basis[tri[1]][np.newaxis, :]  # (1,9)
            sh2 = sh_basis[tri[2]][np.newaxis, :]  # (1,9)
            for u in range(u_min, u_max + 1):
                for v in range(v_min, v_max + 1):
                    if not mesh_numpy.render.is_point_in_tri([u, v], vertices[tri, :2]):
                        continue
                    w0, w1, w2 = mesh_numpy.render.get_point_weight([u, v], vertices[tri, :2])
                    w0 = w0 * np.dot(sh0, sh_light).T  # (3, 1)
                    w1 = w1 * np.dot(sh1, sh_light).T  # (3, 1)
                    w2 = w2 * np.dot(sh2, sh_light).T  # (3, 1)

                    coefficient_matrix.append(
                        (
                            w0 * tex_pc[tri[0], :, :]
                            + w1 * tex_pc[tri[1], :, :]
                            + w2 * tex_pc[tri[2], :, :]
                        ).tolist()
                    )
                    weighted_tex_mu = np.squeeze(
                        w0 * tex_mu[tri[0], :][:, np.newaxis]
                        + w1 * tex_mu[tri[1], :][:, np.newaxis]
                        + w2 * tex_mu[tri[2], :][:, np.newaxis]
                    )
                    result_vector.append((image[v, u] - weighted_tex_mu).tolist())

        # 假设共n个可见像素，即n个方程，每个方程又包括r/g/b三个方程，所以相当于3n个方程
        coefficient_matrix = np.reshape(
            np.array(coefficient_matrix), [-1, n_tp]
        )  # (n, 3, n_tp) -> (3n, n_tp)
        result_vector = np.reshape(np.array(result_vector), [-1, 1])  # (n, 3) -> (3n, 1)

        # --- solve
        equation_left = np.dot(coefficient_matrix.T, coefficient_matrix)  # (n_tp, n_tp)
        equation_right = np.dot(coefficient_matrix.T, result_vector)  # (n_tp, 1)
        """
        这里除以了 sigma，是因为在 morphable_model.py 中定义的 generate_colors 函数，根据 tex_para 来合成 colors，
        在该函数中， colors = tex_mu + tex_pc.dot(tex_para * tex_ev)
        """
        return np.dot(np.linalg.inv(equation_left), equation_right) / tex_ev

    def estimate_light(
        self, image: Any, vertices: Any, colors: Any, light_by_channel: bool,
    ) -> Any:
        """Use least square method to fit spherical harmonics lighting parameters.

        :param image: (h, w, c). range 0~255
        :param vertices: (n_ver, 3). 沿y轴翻转
        :param colors: (3*n_ver, 1). range 0~255
        :param light_by_channel: True --> sh_para:(9, 3). False --> sh_para:(9, 1)
        :return: (9, 3) or (9, 1). spherical harmonics lighting parameters
        """
        height = image.shape[0]
        width = image.shape[1]
        colors = np.reshape(colors, [-1, 3])  # (n_ver, 3)
        triangles = self._model.model["tri"]  # (n_tri, 3)

        # calculate spherical harmonic basis function: sh
        normal = mesh_numpy.light.get_normal(vertices, triangles)  # (n_ver, 3)
        sh_basis = np.array(
            (
                np.ones(vertices.shape[0]),
                normal[:, 0],
                normal[:, 1],
                normal[:, 2],
                normal[:, 0] * normal[:, 1],
                normal[:, 0] * normal[:, 2],
                normal[:, 1] * normal[:, 2],
                normal[:, 0] ** 2 - normal[:, 1] ** 2,
                3 * (normal[:, 2] ** 2) - 1,
            )
        ).T  # (n_ver, 9)

        if light_by_channel:
            coefficient_matrix = [[], [], []]
            result_vector = [[], [], []]

            for i in range(triangles.shape[0]):
                tri = triangles[i, :]
                # the inner bounding box
                u_min = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
                u_max = min(int(np.floor(np.max(vertices[tri, 0]))), width - 1)
                v_min = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
                v_max = min(int(np.floor(np.max(vertices[tri, 1]))), height - 1)
                if u_max < u_min or v_max < v_min:
                    continue

                for u in range(u_min, u_max + 1):
                    for v in range(v_min, v_max + 1):
                        if not mesh_numpy.render.is_point_in_tri([u, v], vertices[tri, :2]):
                            continue

                        w0, w1, w2 = mesh_numpy.render.get_point_weight([u, v], vertices[tri, :2])
                        w0_pp, w1_pp, w2_pp = (
                            w0 * colors[tri[0]],
                            w1 * colors[tri[1]],
                            w2 * colors[tri[2]],
                        )  # (3,)
                        coefficient_matrix[0].append(
                            (
                                w0_pp[0] * sh_basis[tri[0]]
                                + w1_pp[0] * sh_basis[tri[1]]
                                + w2_pp[0] * sh_basis[tri[2]]
                            ).tolist()
                        )
                        coefficient_matrix[1].append(
                            (
                                w0_pp[1] * sh_basis[tri[0]]
                                + w1_pp[1] * sh_basis[tri[1]]
                                + w2_pp[1] * sh_basis[tri[2]]
                            ).tolist()
                        )
                        coefficient_matrix[2].append(
                            (
                                w0_pp[2] * sh_basis[tri[0]]
                                + w1_pp[2] * sh_basis[tri[1]]
                                + w2_pp[2] * sh_basis[tri[2]]
                            ).tolist()
                        )

                        for i in range(3):
                            result_vector[i].append(image[v, u, i])

            coefficient_matrix = np.array(coefficient_matrix)  # (3, n, 9)

            # channel 0
            equation_left = np.dot(coefficient_matrix[0].T, coefficient_matrix[0])  # (9, 9)
            equation_right = np.dot(
                coefficient_matrix[0].T, np.array(result_vector[0])[:, np.newaxis]
            )  # (9, 1)
            gamma0 = np.dot(np.linalg.inv(equation_left), equation_right)  # (9, 1)

            # channel 1
            equation_left = np.dot(coefficient_matrix[1].T, coefficient_matrix[1])  # (9, 9)
            equation_right = np.dot(
                coefficient_matrix[1].T, np.array(result_vector[1])[:, np.newaxis]
            )  # (9, 1)
            gamma1 = np.dot(np.linalg.inv(equation_left), equation_right)  # (9, 1)

            # channel 2
            equation_left = np.dot(coefficient_matrix[2].T, coefficient_matrix[2])  # (9, 9)
            equation_right = np.dot(
                coefficient_matrix[2].T, np.array(result_vector[2])[:, np.newaxis]
            )  # (9, 1)
            gamma2 = np.dot(np.linalg.inv(equation_left), equation_right)  # (9, 1)
            return np.squeeze(np.array([gamma0, gamma1, gamma2])).T

        coefficient_matrix = []
        result_vector = []

        for i in range(triangles.shape[0]):
            tri = triangles[i, :]
            # the inner bounding box
            u_min = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
            u_max = min(int(np.floor(np.max(vertices[tri, 0]))), width - 1)
            v_min = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
            v_max = min(int(np.floor(np.max(vertices[tri, 1]))), height - 1)
            if u_max < u_min or v_max < v_min:
                continue

            for u in range(u_min, u_max + 1):
                for v in range(v_min, v_max + 1):
                    if not mesh_numpy.render.is_point_in_tri([u, v], vertices[tri, :2]):
                        continue
                    w0, w1, w2 = mesh_numpy.render.get_point_weight([u, v], vertices[tri, :2])
                    w0_pp, w1_pp, w2_pp = (
                        w0 * colors[tri[0]],
                        w1 * colors[tri[1]],
                        w2 * colors[tri[2]],
                    )  # (3,)

                    coefficient_matrix.extend(
                        [
                            (
                                w0_pp[0] * sh_basis[tri[0]]
                                + w1_pp[0] * sh_basis[tri[1]]
                                + w2_pp[0] * sh_basis[tri[2]]
                            ).tolist(),
                            (
                                w0_pp[1] * sh_basis[tri[0]]
                                + w1_pp[1] * sh_basis[tri[1]]
                                + w2_pp[1] * sh_basis[tri[2]]
                            ).tolist(),
                            (
                                w0_pp[2] * sh_basis[tri[0]]
                                + w1_pp[2] * sh_basis[tri[1]]
                                + w2_pp[2] * sh_basis[tri[2]]
                            ).tolist(),
                        ]
                    )

                    result_vector.extend([image[v, u, 0], image[v, u, 1], image[v, u, 2]])

        coefficient_matrix = np.array(coefficient_matrix)  # (3n, 9)
        result_vector = np.array(result_vector)[:, np.newaxis]  # (3n, 1)

        equation_left = np.dot(coefficient_matrix.T, coefficient_matrix)  # (9, 9)
        equation_right = np.dot(coefficient_matrix.T, result_vector)  # (9, 1)
        return np.dot(np.linalg.inv(equation_left), equation_right)

    def fit_exp_pose(
        self, kpt_coor: Any, shape_para: Any
    ) -> Tuple[Any, float, Any, Any]:
        """Fit expression and pose parameters according to shape.
        Reference: keypoints

        :param kpt_coor: (68, 2). facial keypoints coordinates
        :param shape_para: (n_sp, 1). fitted shape parameters
        :returns exp_para: (29, 1). expression parameters
        :returns scale: <float>
        :returns rotation_matrix: (3, 3)
        :returns translation_3d: (3,)
        """
        kpt_coor = kpt_coor.copy().T
        # initialize
        exp_para = np.zeros((29, 1), dtype=np.float32)

        # -------------------- estimate
        valid_ind = self._model.get_valid_index()
        shape_mu = self._model.model["shapeMU"][valid_ind, :]  # (3n, 1)
        shape_pc = self._model.model["shapePC"][valid_ind, : shape_para.shape[0]]  # (3n, n_sp)
        exp_pc = self._model.model["expPC"][valid_ind, :29]  # (3n, 29)

        # Estimating expression & pose...
        # init
        scale: float = 0
        rotation_matrix: Any = np.zeros([3, 3])
        translation_3d: Any = np.zeros([3])

        for i in range(self._max_iter):
            geometry = shape_mu + shape_pc.dot(shape_para) + exp_pc.dot(exp_para)
            geometry = np.reshape(geometry, [-1, 3]).T

            # ----- estimate pose
            P = mesh_numpy.transform.estimate_affine_matrix_3d22d(geometry.T, kpt_coor.T)
            scale, rotation_matrix, translation_3d = mesh_numpy.transform.P2sRt(P)

            # ----- estimate expression
            shape = np.reshape(shape_pc.dot(shape_para), [-1, 3]).T  # (3, 68)
            exp_para = self.estimate_expression(
                kpt_coor, shape, scale, rotation_matrix, translation_3d[:2], lamb=20
            )
        return exp_para, scale, rotation_matrix, translation_3d

    def fit_geometry_pose(
        self, kpt_coor: Any, n_sp: int = 199
    ) -> Tuple[Any, Any, float, Any, Any]:
        """Fit geometry(shape & exp) and pose parameters.

        :param kpt_coor: (68, 2). facial keypoints coordinates
        :param n_sp: the number of shape parameters to be fitted
        :returns shape_para: (n_sp, 1). fitted shape parameters
        :returns exp_para: (29, 1). fitted expression parameters
        :returns scale: <float>
        :returns rotation_matrix: (3, 3)
        :returns translation_3d: (3,)
        """
        kpt_coor = kpt_coor.copy().T

        # -- init
        shape_para = np.zeros((n_sp, 1), dtype=np.float32)
        exp_para = np.zeros((29, 1), dtype=np.float32)

        # -------------------- estimate
        valid_ind = self._model.get_valid_index()
        shape_mu = self._model.model["shapeMU"][valid_ind, :]  # (3x68, 1)
        shape_pc = self._model.model["shapePC"][valid_ind, :n_sp]  # (3x68, n_sp)
        exp_pc = self._model.model["expPC"][valid_ind, :29]  # (3x68, 29)

        # Estimating geometry and pose...
        # init
        scale: float = 0
        rotation_matrix: Any = np.zeros([3, 3])
        translation_3d: Any = np.zeros([3])

        for i in range(self._max_iter):
            geometry = shape_mu + shape_pc.dot(shape_para) + exp_pc.dot(exp_para)
            geometry = np.reshape(geometry, [-1, 3]).T
            # ----- estimate pose
            P = mesh_numpy.transform.estimate_affine_matrix_3d22d(geometry.T, kpt_coor.T)
            scale, rotation_matrix, translation_3d = mesh_numpy.transform.P2sRt(P)
            # shape
            shape_para = self.estimate_shape(
                kpt_coor,
                np.reshape(exp_pc.dot(exp_para), [-1, 3]).T,
                scale,
                rotation_matrix,
                translation_3d[:2],
                lamb=40,
            )
        exp_para = self.estimate_expression(
            kpt_coor,
            np.reshape(shape_pc.dot(shape_para), [-1, 3]).T,
            scale,
            rotation_matrix,
            translation_3d[:2],
            lamb=20,
        )
        return shape_para, exp_para, scale, rotation_matrix, translation_3d

    def fit_light(
        self,
        image: Any,
        shape_para: Any,
        exp_para: Any,
        scale: float,
        rotation_matrix: Any,
        translation_3d: Sequence[float],
        tex_para: Any,
        light_by_channel: bool = True,
    ) -> Any:
        """Fit spherical harmonic lighting parameters.
        Reference: image pixel values

        :param image: (h, w, 3)
        :param shape_para: (n_sp, 1)
        :param exp_para: (29, 1)
        :param scale: <float>
        :param rotation_matrix: (3, 3)
        :param translation_3d: (3,)
        :param tex_para: (n_tp, 1)
        :param light_by_channel: True -> sh_para:(9, 3). False -> sh_para:(9, 1)
        :return: fitted spherical harmonic lighting parameters. (9, 3) or (9, 1)
        """
        n_tp = tex_para.shape[0]
        shape_mu = self._model.model["shapeMU"]  # (3*n_ver, 1)
        shape_pc = self._model.model["shapePC"][:, : shape_para.shape[0]]  # (3*n_ver, n_sp)
        exp_pc = self._model.model["expPC"][:, : exp_para.shape[0]]  # (3*n_ver, n_ep)
        tex_pc = self._model.model["texPC"][:, :n_tp]  # (3*n_ver, n_tp)
        tex_ev = self._model.model["texEV"][:n_tp, :]  # (n_tp, 1)

        # generate vertices
        vertices = shape_mu + shape_pc.dot(shape_para) + exp_pc.dot(exp_para)  # (3*n_ver, 1)
        vertices = np.reshape(vertices, [-1, 3])  # (n_ver, 3)
        vertices = mesh_numpy.transform.similarity_transform(
            vertices, scale, rotation_matrix, translation_3d
        )  # (n_ver, 3)
        transformed_vertices = vertices
        transformed_vertices[:, 1] = np.shape(image)[0] - vertices[:, 1] - 1

        # Estimating light...
        colors = np.reshape(
            self._model.model["texMU"] + tex_pc.dot(tex_para * tex_ev), [-1, 3]
        )  # (n_ver, 3)
        return self.estimate_light(image, transformed_vertices, colors, light_by_channel)

    def fit_texture_light(
        self,
        image: Any,
        shape_para: Any,
        exp_para: Any,
        scale: float,
        rotation_matrix: Any,
        translation_3d: Sequence[float],
        n_tp: int,
        light_by_channel: bool = True,
    ):
        """Fit texture and light parameters.

        :param image: (h, w, 3)
        :param shape_para: (n_sp, 1)
        :param exp_para: (29, 1)
        :param scale: <float>
        :param rotation_matrix: (3, 3)
        :param translation_3d: (3,)
        :param n_tp: the number of texture parameters to be fitted
        :param light_by_channel: True -> sh_para:(9, 3). False -> sh_para:(9, 1)
        :returns tex_para: (n_tp, 1). fitted texture parameters
        :returns sh_light: fitted spherical harmonic lighting parameters. (9, 3) or (9, 1)
        """
        shape_pc = self._model.model["shapePC"][:, : shape_para.shape[0]]  # (3*n_ver, n_sp)
        exp_pc = self._model.model["expPC"][:, : exp_para.shape[0]]  # (3*n_ver, n_ep)

        tex_mu = self._model.model["texMU"]  # (3*n_ver, 1)
        tex_pc = self._model.model["texPC"][:, :n_tp]  # (3*n_ver, n_tp)
        tex_ev = self._model.model["texEV"][:n_tp, :]  # (n_tp, 1)

        # generate vertices
        vertices = (
            self._model.model["shapeMU"] + shape_pc.dot(shape_para) + exp_pc.dot(exp_para)
        )  # (3*n_ver, 1)
        vertices = np.reshape(vertices, [-1, 3])  # (n_ver, 3)
        vertices = mesh_numpy.transform.similarity_transform(
            vertices, scale, rotation_matrix, translation_3d
        )  # (n_ver, 3)
        transformed_vertices = vertices
        transformed_vertices[:, 1] = np.shape(image)[0] - vertices[:, 1] - 1

        # Estimating texture and light...
        # init
        sh_light = np.random.random([9, 3]) if light_by_channel else np.random.random([9, 1])
        for i in range(self._max_iter):
            tex_para = self.estimate_texture(image, vertices, n_tp, sh_light)
            colors = np.reshape(tex_mu + tex_pc.dot(tex_para * tex_ev), [-1, 3])  # (n_ver, 3)
            sh_light = self.estimate_light(image, vertices, colors, light_by_channel)
        return tex_para, sh_light

    def fit(
        self, image: Any, kpt_coor: Any, n_sp: int, n_tp: int, light_by_channel: bool = True
    ) -> Tuple[Any, Any, float, Any, Sequence[float], Any, Any]:
        """Fit all parameters, including shape, expression, pose, texture, light

        :param image: (h, w, 3). The image containing a face to be fitted.
        :param kpt_coor: (68, 2). face keypoint coordinates in the input image
        :param n_sp: the number of shape parameters
        :param n_tp: the number of texture parameters
        :param light_by_channel: True --> sh_para:(9, 3); False --> sh_para:(9, 1)
        :returns shape_para: (n_sp, 1). shape parameters
        :returns exp_para: (29, 1). expression parameters
        :returns scale: <float>
        :returns rotation_matrix: (3, 3).
        :returns translation_3d: (3,)
        :returns tex_para: (n_tp, 1). texture parameters
        :returns sh_light: spherical harmonic lighting parameters. If light_by_channel, (9, 3) else (9, 1).
        """
        shape_para, exp_para, scale, rotation_matrix, translation_3d = self.fit_geometry_pose(kpt_coor, n_sp)
        tex_para, sh_light = self.fit_texture_light(
            image, shape_para, exp_para, scale, rotation_matrix, translation_3d, n_tp, light_by_channel
        )
        return shape_para, exp_para, scale, rotation_matrix, translation_3d, tex_para, sh_light
