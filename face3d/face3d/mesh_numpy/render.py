#!/usr/bin/env python3
# Author: AnniTang

"""
Functions about rendering mesh(from 3d obj to 2d image),
using rasterization render here.
Note that:
1. Generally, render func includes camera, light, raterize. Here no camera and light.
2. Generally, the input vertices are normalized to [-1,1] and centered on [0, 0]. (in world space)
   Here, the vertices are using image coords, which centers on [w/2, h/2] with the y-axis pointing to opposite direction.
Means: render here only conducts interpolation.(I just want to make the input flexible)

Preparation knowledge:
z-buffer: https://cs184.eecs.berkeley.edu/lecture/pipeline
"""

from typing import Sequence, Tuple

import numpy as np


def is_point_in_tri(point: Sequence[int], tri_points: np.ndarray) -> bool:
    """ Judge whether a point is in a triangle.
    Method: http://blackpawn.com/texts/pointinpoly/

    :param point: (2,). [u, v] or [x, y]
    :param tri_points: (3 vertices, 2 coordinates). three vertices(2d points) of a triangle.
    :return: whether the point is in the triangle
    """
    v0 = tri_points[2, :] - tri_points[0, :]
    v1 = tri_points[1, :] - tri_points[0, :]
    v2 = point - tri_points[0, :]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycen_tric coordinates
    in_ver_deno = 0 if dot00 * dot11 - dot01 * dot01 == 0 else 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * in_ver_deno
    v = (dot00 * dot12 - dot01 * dot02) * in_ver_deno
    return (u >= 0) & (v >= 0) & (u + v < 1)


def get_point_weight(point: Sequence[int], tri_points: np.ndarray) -> Tuple[float, float, float]:
    """ Get the weights of the point.
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycen_tric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    
    :param point: (2,). [u, v] or [x, y]
    :param tri_points: (3 vertices, 2 coordinates). three vertices(2d points) of a triangle.
    :returns: weights of the point position 
    """
    # vectors
    v0 = tri_points[2, :] - tri_points[0, :]
    v1 = tri_points[1, :] - tri_points[0, :]
    v2 = point - tri_points[0, :]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycen_tric coordinates
    in_ver_deno = 0 if dot00 * dot11 - dot01 * dot01 == 0 else 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * in_ver_deno
    v = (dot00 * dot12 - dot01 * dot02) * in_ver_deno
    return 1 - u - v, v, u


def rasterize_triangles(
    vertices: np.ndarray, triangles: np.ndarray, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize triangles.
    Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    h, w is the size of rendering

    :param vertices: (n_ver, 3)
    :param triangles: (n_tri, 3)
    :param h: the height of the rendering image
    :param w: the width of the rendering image
    :return depth_buffer: save the depth, here, the bigger the z, the frontier the point. (h, w)
    :return triangle_buffer: save the triangle id(-1 for no triangle). (h, w)
    :return barycen_tric_weight: save corresponding barycen_tric weight. (h, w, 3)
    """
    # initial
    depth_buffer = np.zeros([h, w]) - 999999.0
    # + np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer = np.zeros([h, w], dtype=np.int32) - 1
    # if tri id = -1, the pixel has no triangle correspondence
    barycen_tric_weight = np.zeros([h, w, 3], dtype=np.float32)

    for i in range(triangles.shape[0]):
        triangle = triangles[i, :]  # 3 vertex indices
        # the inner bounding box
        u_min = max(int(np.ceil(np.min(vertices[triangle, 0]))), 0)
        u_max = min(int(np.floor(np.max(vertices[triangle, 0]))), w - 1)
        v_min = max(int(np.ceil(np.min(vertices[triangle, 1]))), 0)
        v_max = min(int(np.floor(np.max(vertices[triangle, 1]))), h - 1)
        if u_max < u_min or v_max < v_min:
            continue

        for u in range(u_min, u_max + 1):
            for v in range(v_min, v_max + 1):
                if not is_point_in_tri([u, v], vertices[triangle, :2]):
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[triangle, :2])  # barycen_tric weight
                point_depth = (
                    w0 * vertices[triangle[0], 2]
                    + w1 * vertices[triangle[1], 2]
                    + w2 * vertices[triangle[2], 2]
                )
                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    triangle_buffer[v, u] = i
                    barycen_tric_weight[v, u, :] = np.array([w0, w1, w2])
    return depth_buffer, triangle_buffer, barycen_tric_weight


def render_mask(vertices: np.ndarray, triangles: np.ndarray, weights: np.ndarray, h: int, w: int) -> np.ndarray:
    """Render a mask image with different weights in each pixel.

    :param vertices: (n_ver, 3)
    :param triangles: (n_tri, 3)
    :param weights: (n_ver,)
    :param h: height
    :param w: width
    :return: a mask image with shape (h, w), each pixel has its own weight when calculating loss
    """
    assert (
        vertices.shape[0] == weights.shape[0]
    ), "Vertices and weights should have the same number!"

    mask = np.zeros([h, w])
    depth_buffer = np.zeros([h, w]) - 999999.0

    for i in range(triangles.shape[0]):
        tri = triangles[i, :]  # 3 vertex indices
        # the inner bounding box
        u_min = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        u_max = min(int(np.floor(np.max(vertices[tri, 0]))), w - 1)
        v_min = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        v_max = min(int(np.floor(np.max(vertices[tri, 1]))), h - 1)
        if u_max < u_min or v_max < v_min:
            continue

        for u in range(u_min, u_max + 1):
            for v in range(v_min, v_max + 1):
                if not is_point_in_tri([u, v], vertices[tri, :2]):
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = (
                    w0 * vertices[tri[0], 2] + w1 * vertices[tri[1], 2] + w2 * vertices[tri[2], 2]
                )

                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    mask[v, u] = w0 * weights[tri[0]] + w1 * weights[tri[1]] + w2 * weights[tri[2]]
    return mask


def render_colors(
    vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray, h: int, w: int
) -> np.ndarray:
    """ Render mesh with colors.

    :param vertices: (n_ver, 3)
    :param triangles: (n_tri, 3)
    :param colors: (n_ver, c)
    :param h: the height of the rendering image
    :param w: the width of the rendering image
    :return: the rendered color image with shape (h, w, c)
    """
    assert vertices.shape[0] == colors.shape[0]
    # initialize
    image = np.zeros((h, w, colors.shape[1]))
    depth_buffer = np.zeros([h, w]) - 999999.0

    for i in range(triangles.shape[0]):
        tri = triangles[i, :]  # 3 vertex indices
        # the inner bounding box
        u_min = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        u_max = min(int(np.floor(np.max(vertices[tri, 0]))), w - 1)
        v_min = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        v_max = min(int(np.floor(np.max(vertices[tri, 1]))), h - 1)
        if u_max < u_min or v_max < v_min:
            continue

        for u in range(u_min, u_max + 1):
            for v in range(v_min, v_max + 1):
                if not is_point_in_tri([u, v], vertices[tri, :2]):
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = (
                    w0 * vertices[tri[0], 2] + w1 * vertices[tri[1], 2] + w2 * vertices[tri[2], 2]
                )

                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    image[v, u, :] = (
                        w0 * colors[tri[0], :] + w1 * colors[tri[1], :] + w2 * colors[tri[2], :]
                    )
    return image


def render_colors_ras(
    vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray, h: int, w: int
) -> np.ndarray:
    """Render mesh with colors (rasterize triangle first)

    :param vertices: (n_ver, 3)
    :param triangles: (n_tri, 3)
    :param colors: (n_ver, 3)
    :param h: the height of the rendering image
    :param w: the width of the rendering image
    :return: (h, w, 3). rendering image.
    """
    assert vertices.shape[0] == colors.shape[0], "Vertices and colors should have the same shape!"

    _, triangle_buffer, barycen_tric_weight = rasterize_triangles(vertices, triangles, h, w)

    triangle_buffer_flat = np.reshape(triangle_buffer, [-1])  # (h*w,)
    barycen_tric_weight_flat = np.reshape(barycen_tric_weight, [-1, 3])  # (h*w, 3)
    weight = barycen_tric_weight_flat[:, :, np.newaxis]  # (h*w, 3(ver in tri), 1)
    colors_flat = colors[
        triangles[triangle_buffer_flat, :], :
    ]  # (h*w(tri id in pixel), 3(ver in tri), 3(color in ver))
    colors_flat = weight * colors_flat  # (h*w, 3, 3)
    colors_flat = np.sum(colors_flat, 1)  # (h*w, 3). add tri.
    return np.reshape(colors_flat, [h, w, 3])
