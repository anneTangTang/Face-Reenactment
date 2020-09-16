from typing import Any, List, Sequence, Tuple

import numpy as np


def is_point_in_polygon(vertices: np.ndarray, point: Sequence) -> bool:
    """Judge whether the point is in the polygon using ray method.
    A ray is emitted horizontally to the right from the point to be judged.
    If the number of intersections is odd(奇), return True else False.

    :param vertices: (n_ver, 2). the coordinates of the vertices of the polygon
    :param point: (2,). the coordinate of the point to be judged
    :return: whether the point is in the polygon
    """
    n_ver = vertices.shape[0]
    intersections: int = 0
    for i in range(n_ver):
        if is_ray_intersect_segment(point, vertices[i], vertices[(i + 1) % n_ver]):
            intersections += 1
    return intersections % 2 == 1


def is_ray_intersect_segment(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> bool:
    """Judge whether the ray intersects the segment.

    :param point: (2,). The start point of the ray, and that the ray is horizontal to the right.
    :param start: (2,). the coordinate of the start point of the segment
    :param end: (2,). the coordinate of the end point of the segment
    :return: whether the ray intersects the segment
    """
    if start[1] == end[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if start[1] > point[1] and end[1] > point[1]:  # 线段在射线上边
        return False
    if start[1] < point[1] and end[1] < point[1]:  # 线段在射线下边
        return False
    if start[1] == point[1] and end[1] > point[1]:  # 交点为下端点，对应start
        return False
    if end[1] == point[1] and start[1] > point[1]:  # 交点为下端点，对应end
        return False
    if start[0] < point[0] and end[1] < point[1]:  # 线段在射线左边
        return False

    intersect = end[0] - (end[0] - start[0]) * (end[1] - point[1]) / (end[1] - start[1])  # 求交点
    if intersect < point[0]:  # 交点在射线起点的左侧
        return False

    return True  # 排除上述情况之后


# def insertPoints(start, end):
#     """Calculate the coordinates of linear inserted points between two points.
#     Note that the x distance between start and end points are larger than 1.
#     Args:
#         start: (2,). the coordinate of the start point
#         end: (2,). the coordinate of the end point
#     Returns:
#         inserted_points: (n,2). n is the number of inserted points
#     """
#     x_distance = np.abs(end[0] - start[0])
#     assert x_distance > 1
#
#     slop = (end[1] - start[1]) / (end[0] - start[0]) # calculate the slop
#     # insert points
#     inserted_points = []
#     for i in range(x_distance - 1):
#         inserted_x = int(start[0] + i + 1)
#         inserted_y = int(start[1] + slop * (i + 1))
#         inserted_points.append([inserted_x, inserted_y])
#     return np.array(inserted_points)


def generate_one_eye(vertices: np.ndarray, center: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Get the coordinates of both white and blue pixels in one eye.
    
    :param vertices: (n_ver, 2). the coordinates of vertices of one eye
    :param center: (2,). the normalized coordinate of the center of one eye
    :returns whites: (n_white,2). the coordinates of white pixels
    :returns blues: (n_blue,2). the coordinates of blue pixels
    """
    u_max = np.max(vertices[:, 0])
    u_min = np.min(vertices[:, 0])
    v_max = np.max(vertices[:, 1])
    v_min = np.min(vertices[:, 1])
    center_u = int((u_max - u_min) * center[0] + u_min)
    center_v = int((v_max - v_min) * center[1] + v_min)

    blues: List[Any] = []
    whites: List[Any] = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if np.abs(i) != 2 or np.abs(j) != 2:
                point = [center_u + i, center_v + j]
                if is_point_in_polygon(vertices, point):
                    blues.append(point)

    for i in range(u_min, u_max + 1):
        for j in range(v_min, v_max + 1):
            point = [i, j]
            if is_point_in_polygon(vertices, point) and point not in blues:
                whites.append(point)

    return np.array(whites), np.array(blues)


def generate_eye_image(
    vertices: np.ndarray, left_center: Sequence[float], right_center: Sequence[float], height: int, width: int,
) -> np.ndarray:
    """Generate the corresponding eye gaze image.

    :param vertices: (12, 2). the outline of eyes
    :param left_center: (2,). the normalized coordinate of the center of left eye
    :param right_center: the normalized coordinate of the center of right eye
    :param height: the height of the desired eye image
    :param width: the width of the desired eye image
    :return: (h, w, 3). synthesized eye image
    """
    eye_image = np.zeros([height, width, 3])
    left_points = vertices[:6, :]
    right_points = vertices[6:, :]
    left_whites, left_blues = generate_one_eye(left_points, left_center)
    right_whites, right_blues = generate_one_eye(right_points, right_center)

    for i in range(left_whites.shape[0]):
        eye_image[left_whites[i, 1]][left_whites[i, 0]] += 1
    for i in range(right_whites.shape[0]):
        eye_image[right_whites[i, 1]][right_whites[i, 0]] += 1

    for i in range(left_blues.shape[0]):
        eye_image[left_blues[i, 1]][left_blues[i, 0]][2] += 1
    for i in range(right_blues.shape[0]):
        eye_image[right_blues[i, 1]][right_blues[i, 0]][2] += 1

    return eye_image


def find_center(image: np.ndarray, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find the center of left eye and right eye.

    :param image: (h, w, 3). range 0~255
    :param vertices: (12, 2). the outline of eyes
    :returns left_center: (2,). the center of left eye. normalized, such as (0.5, 0.5) -> (left_margin, top_margin)
    :returns right_center: (2,). the center of right eye, same as left
    """
    cropped_eye_image = np.zeros([image.shape[0], image.shape[1], 3]) + 255
    left_points = vertices[:6, :]  # (6,2)
    right_points = vertices[6:, :]  # (6,2)

    # left eye
    u_max_left = np.max(left_points[:, 0])
    u_min_left = np.min(left_points[:, 0])
    v_max_left = np.max(left_points[:, 1])
    v_min_left = np.min(left_points[:, 1])

    left_eye: List[Any] = []
    for i in range(left_points.shape[0]):
        left_eye.append(left_points.tolist()[i])

    for i in range(u_min_left, u_max_left + 1):
        for j in range(v_min_left, v_max_left + 1):
            point = np.array([i, j])
            if is_point_in_polygon(left_points, point):
                left_eye.append([i, j])

    for pixel in left_eye:
        u, v = pixel[0], pixel[1]
        cropped_eye_image[v][u] = image[v][u]

    # right eye
    u_max_right = np.max(right_points[:, 0])
    u_min_right = np.min(right_points[:, 0])
    v_max_right = np.max(right_points[:, 1])
    v_min_right = np.min(right_points[:, 1])

    right_eye: List[Any] = []
    for i in range(right_points.shape[0]):
        right_eye.append(right_points.tolist()[i])

    for i in range(u_min_right, u_max_right + 1):
        for j in range(v_min_right, v_max_right + 1):
            point = np.array([i, j])
            if is_point_in_polygon(right_points, point):
                right_eye.append([i, j])

    for pixel in right_eye:
        u, v = pixel[0], pixel[1]
        cropped_eye_image[v][u] = image[v][u]

    # ------------- save
    cropped_eye_image /= 255

    # -----find center from cropped_eye_image
    # left eye
    n_dim_left = v_max_left - v_min_left
    if n_dim_left % 2 == 0:
        n_dim_left -= 1  # n_dim should be odd.

    kernel_left = np.zeros([n_dim_left, n_dim_left, 3]) - 1

    left_max = -99999
    for l in left_eye:
        conv = convolution_3d(cropped_eye_image, l, kernel_left)
        if conv > left_max:
            left_max = conv
            left_center = l

    # right eye
    n_dim_right = v_max_right - v_min_right
    if n_dim_right % 2 == 0:  # n_dim should be odd.
        n_dim_right -= 1

    kernel_right = np.zeros([n_dim_right, n_dim_right, 3])
    kernel_right -= 1

    right_max = -99999
    for r in right_eye:
        conv = convolution_3d(cropped_eye_image, r, kernel_right)
        if conv > right_max:
            right_max = conv
            right_center = r

    # normalize
    if right_max > left_max:
        right_center[0] = (right_center[0] - u_min_right) / (u_max_right - u_min_right)
        right_center[1] = (right_center[1] - v_min_right) / (v_max_right - v_min_right)
        left_center = right_center
    else:
        left_center[0] = (left_center[0] - u_min_left) / (u_max_left - u_min_left)
        left_center[1] = (left_center[1] - v_min_left) / (v_max_left - v_min_left)
        right_center = left_center

    return left_center, right_center


def convolution_3d(image: np.ndarray, point: Sequence[float], kernel: np.ndarray) -> float:
    """Calculate the value of 3d convolution.
    
    :param image: (h, w, 3). 
    :param point: (u, v)
    :param kernel: (n,n,3)
    :return: the convolution result
    """
    assert image.shape[2] == kernel.shape[2]
    assert 0 <= point[0] < image.shape[1]
    assert 0 <= point[1] < image.shape[0]
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1

    height = image.shape[0]
    width = image.shape[1]
    n_dim = kernel.shape[0]
    is_padded = False

    if point[0] < int(n_dim / 2) or point[1] < int(n_dim / 2):
        is_padded = True
        image_padding = np.ones([(height + n_dim - 1), (width + n_dim - 1), 3])
        for v in range(height):
            for u in range(width):
                image_padding[int(n_dim / 2) + v][int(n_dim / 2) + u] = image[v][u]
        image = image_padding

    if is_padded:
        center_u = int(n_dim / 2) + point[0]
        center_v = int(n_dim / 2) + point[1]
    else:
        center_u = point[0]
        center_v = point[1]

    conv = 0
    for i in range(n_dim):
        for j in range(n_dim):
            tmp_v = center_v - int(n_dim / 2) + i
            tmp_u = center_u - int(n_dim / 2) + j
            conv += np.sum(image[tmp_v][tmp_u] * kernel[i][j])
    return conv
