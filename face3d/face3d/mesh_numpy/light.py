#!/usr/bin/env python3
# Author: AnniTang

"""Functions about lighting mesh(changing colors/texture of mesh),
using the spherical harmonics lighting model.

Preparation knowledge:
lighting: https://cs184.eecs.berkeley.edu/lecture/pipeline
spherical harmonics in human face: '3D Face Reconstruction from a Single Image Using a Single Reference Face Shape'
"""
from typing import Any

import numpy as np


def get_normal(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Calculate normal direction of each vertex
    
    :param vertices: (n_ver, 3)
    :param triangles: (n_tri, 3)
    :return: (n_ver, 3)
    """
    pt0 = vertices[triangles[:, 0], :]  # (n_tri, 3)
    pt1 = vertices[triangles[:, 1], :]  # (n_tri, 3)
    pt2 = vertices[triangles[:, 2], :]  # (n_tri, 3)
    tri_normal = np.cross(pt0 - pt1, pt0 - pt2)  # (n_tri, 3). normal of each triangle

    normal = np.zeros_like(vertices)  # (n_ver, 3)
    for i in range(triangles.shape[0]):
        for j in range(3):
            normal[triangles[i, j], :] = normal[triangles[i, j], :] + tri_normal[i, :]

    # normalize to unit length
    mag = np.sum(normal ** 2, 1)  # (n_ver)
    zero_ind = mag == 0
    mag[zero_ind] = 1
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))
    return normal / np.sqrt(mag[:, np.newaxis])


def add_light_sh(
    vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray, sh_coeff: np.ndarray
) -> Any:
    """Add light to vertices based on spherical harmonics model.
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    --> can be expressed in terms of spherical harmonics(omit the lighting coefficients)
    I = albedo * (Y(n) x sh_coeff)

    n: normal direction of each vertex
    Y(n): spherical harmonics base function, number=4 if band=2, number=9 if band=3, etc.
    sh_coeff: 9 x 1(band=3), each channel has the same sh_coeff.
              9 x 3 if each channel has different sh_coeff.
    albedo: n x 1, each vertex has different albedo
    Y(n) = (1, n_x, n_y, n_z, n_xn_y, n_xn_z, n_yn_z, n_x^2 - n_y^2, 3n_z^2 - 1)': n x 9
    # Y(n) = (1, n_x, n_y, n_z)': n x 4

    :param vertices: (n_ver, 3)
    :param triangles: (n_tri, 3)
    :param colors: (n_ver, 3). albedo
    :param sh_coeff: (9, 1) or (9, 3). spherical harmonics coefficients
    :return: (n_ver, 3). lit colors
    """
    assert vertices.shape[0] == colors.shape[0], "Vertices and colors should have the same number!"
    n_ver = vertices.shape[0]
    normal = get_normal(vertices, triangles)  # (n_ver, 3)
    sh = np.array(
        (
            np.ones(n_ver),
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

    ref = sh.dot(sh_coeff)  # (n_ver, 1) or (n_ver, 3)
    lit_colors = colors * ref  # (n_ver, 3)
    return lit_colors
