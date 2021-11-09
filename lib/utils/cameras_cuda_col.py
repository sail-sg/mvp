# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import torch
import cv2


def project_pose(x, camera=None, **kwargs):
    """
    Args
        x: 3xN points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: 2 Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel: 2xN points in pixel space
    """
    if camera:
        device = x.device
        R = torch.as_tensor(camera['R'], device=device, dtype=torch.float32)
        T = torch.as_tensor(camera['T'], device=device, dtype=torch.float32)
        f = torch.as_tensor([[camera['fx']], [camera['fy']]],
                            device=device, dtype=torch.float32)
        c = torch.as_tensor([[camera['cx']], [camera['cy']]],
                            device=device, dtype=torch.float32)
    else:
        R = kwargs['R']
        T = kwargs['T']
        f = kwargs['f']
        c = kwargs['c']

    xcam = torch.mm(R, x - T)
    y = xcam[:2] / xcam[2]
    ypixel = (f * y) + c
    return ypixel


def world_to_camera_frame(x, camera):
    """
    Args
        x: 3xN 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: 3xN 3d points in camera coordinates
    """
    R = camera['R']
    T = camera['T']
    xcam = torch.mm(R, x - T)
    return xcam


def camera_to_world_frame(x, camera):
    """
    Args
        x: 3xN points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: 3xN points in world coordinates
    """

    R = torch.as_tensor(camera['R'], device=x.device, dtype=torch.float32)
    T = torch.as_tensor(camera['T'], device=x.device, dtype=torch.float32)
    xcam = torch.mm(torch.t(R), x)
    xcam = xcam + T  # rotate and translate
    return xcam


def uv_to_image_frame(uv, camera):
    """

    :param uv: (2, N)
    :param f: scalar
    :param c: (2, 1)
    :param k:
    :param p:
    :return:
    """
    device = uv.device
    f = torch.as_tensor([[camera['fx']], [camera['fy']]],
                        device=device, dtype=torch.float32)
    c = torch.as_tensor([[camera['cx']], [camera['cy']]],
                        device=device, dtype=torch.float32)
    xy = (uv - c) / f
    return xy


def backproject_uv_to_depth(uv, camera, depth):
    """
    assume there are several layers of
    different depth, this function back project uv to all this layers
    :param uv: (2, N)
    :param camera:
    :param depth: 1d tensor
    :return: (n_dep, 3, N)
    """
    depth = torch.as_tensor(depth, device=uv.device, dtype=torch.float32)
    norm_xy = uv_to_image_frame(uv, camera)
    norm_xy1 = torch.cat(
        (norm_xy, torch.ones_like(torch.unsqueeze(norm_xy[0], dim=0))), dim=0)
    xyz_all_depth = []
    for dep in depth:
        xyz = norm_xy1 * dep
        xyz_all_depth.append(xyz)
    return xyz_all_depth


def get_affine_transform(center, scale, patch_size, inv=0):
    """
    :param center: (2,)
    :param scale: (2,)
    :param patch_size:
    :param inv: inv=0 image->crop_img
    :return:
    """
    half_scale = scale * 100.0
    dst_w = patch_size[0]
    dst_h = patch_size[1]

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])

    src[1, :] = center + half_scale
    dst[1, :] = np.array([dst_w, dst_h])

    src[2:, :] = src[0, :] + np.array([half_scale[0], -half_scale[1]])
    dst[2:, :] = np.array([dst_w, 0])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)
    return trans


def affine_transform_pts(pts, trans):
    """

    :param pts: (2, N)
    :param trans: (2, 3)
    :return:
    """
    trans = torch.as_tensor(trans, device=pts.device, dtype=torch.float32)
    xy1 = torch.stack((pts[0], pts[1], torch.ones_like(pts[0]))).contiguous()
    return torch.mm(trans, xy1)


if __name__ == '__main__':
    import lib.utils.transforms as trans
    center = np.array([100., 100.], dtype=np.float32)
    scale = np.array([100., 120.], dtype=np.float32)/200
    patch_size = [50, 60]

    trans_1 = trans.get_affine_transform(
        center=center, scale=scale, rot=0, inv=0, output_size=patch_size)
    trans_2 = get_affine_transform(
        center=center, scale=scale, inv=0, patch_size=patch_size)
    print(trans_1, trans_2)
    print(trans_1.shape, trans_1.dtype, trans_2.dtype)
    print(np.isclose(trans_1, trans_2, atol=1e-7))

    cords = torch.randn(2, 10)
    print(trans.affine_transform_pts(cords.numpy().T, trans_1).T)
    print(affine_transform_pts(cords, trans_1))
    cords = cords.cuda()
    print(affine_transform_pts(cords, trans_1))

    affin_t = get_affine_transform(
        center=center, scale=scale, inv=0, patch_size=patch_size)
    inv_affin_t = get_affine_transform(
        center=center, scale=scale, inv=1, patch_size=patch_size)
    print(affin_t, inv_affin_t)
