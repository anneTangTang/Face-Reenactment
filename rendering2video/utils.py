import cv2
import numpy as np
import torch


def bilinear_interpolation(src_img, dst_h, dst_w):
    """Use bilinear interpolation method to downsample the source image.

    :param src_img: (n, 9Nw, h, w). cuda Tensor
    :param dst_h: <int>. the height of the destination image
    :param dst_w: <int>. the width of the destination image
    :return: (n, 9Nw, dst_h, dst_w). cuda Tensor
    """
    src_img = src_img.cpu().numpy()
    n = src_img.shape[0]
    dst = []

    for i in range(n):
        src_tmp = src_img[i]  # (9Nw, h, w)
        src_tmp = np.swapaxes(src_tmp, 0, 1)  # (h, 9Nw, w)
        src_tmp = np.swapaxes(src_tmp, 1, 2)  # (h, w, 9Nw)
        dst_tmp = cv2.resize(src_tmp, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)  # (dst_h, dst_w, 9Nw)
        dst.append(dst_tmp.tolist())
    dst_img = np.array(dst)  # (n, dst_h, dst_w, 9Nw)
    dst_img = np.swapaxes(dst_img, 1, 3)  # (n, 9Nw, dst_w, dst_h)
    dst_img = np.swapaxes(dst_img, 2, 3)  # (n, 9Nw, dst_h, dst_w)
    dst_img = torch.from_numpy(dst_img).type(torch.cuda.FloatTensor)

    return dst_img
