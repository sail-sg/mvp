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
import torch


def unfold_camera_param(camera, device=None):
    R = torch.as_tensor(camera['R'],
                        dtype=torch.float, device=device).reshape(3, 3)
    T = torch.as_tensor(camera['T'],
                        dtype=torch.float, device=device).reshape(3, 1)
    fx = torch.as_tensor(camera['fx'], dtype=torch.float, device=device)
    fy = torch.as_tensor(camera['fy'], dtype=torch.float, device=device)
    cx = torch.as_tensor(camera['cx'], dtype=torch.float, device=device)
    cy = torch.as_tensor(camera['cy'], dtype=torch.float, device=device)
    f = torch.tensor([fx, fy], dtype=torch.float, device=device).reshape(2, 1)
    c = torch.as_tensor(
        [[cx], [cy]],
        dtype=torch.float,
        device=device).reshape(2, 1)
    k = torch.as_tensor(camera['k'],
                        dtype=torch.float, device=device).reshape(3, 1)
    p = torch.as_tensor(camera['p'],
                        dtype=torch.float, device=device).reshape(2, 1)
    return R, T, f, c, k, p


def unfold_camera_param_batch(camera, device=None):
    R = torch.as_tensor(camera['R'], dtype=torch.float, device=device)
    bs, nview, _, _ = R.shape
    T = torch.as_tensor(camera['T'], dtype=torch.float, device=device)
    fx = torch.as_tensor(camera['fx'], dtype=torch.float, device=device)
    fy = torch.as_tensor(camera['fy'], dtype=torch.float, device=device)
    f = torch.tensor(torch.stack([fx, fy], dim=2),
                     dtype=torch.float, device=device).view(bs, nview, 2, 1)
    c = torch.as_tensor(
        torch.stack([camera['cx'], camera['cy']], dim=2),
        dtype=torch.float,
        device=device).view(bs, nview, 2, 1)
    k = torch.as_tensor(camera['k'], dtype=torch.float, device=device)
    p = torch.as_tensor(camera['p'], dtype=torch.float, device=device)
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: 2x1 Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    xcam = torch.mm(R, torch.t(x) - T)
    y = xcam[:2] / (xcam[2] + 1e-5)

    kexp = k.repeat((1, n))
    r2 = torch.sum(y**2, 0, keepdim=True)
    r2exp = torch.cat([r2, r2**2, r2**3], 0)
    radial = 1 + torch.einsum('ij,ij->j', kexp, r2exp)

    tan = p[0] * y[1] + p[1] * y[0]
    corr = (radial + 2 * tan).repeat((2, 1))

    y = y * corr + torch.ger(torch.cat([p[1], p[0]]).view(-1), r2.view(-1))
    ypixel = (f * y) + c
    return torch.t(ypixel)


def project_point_radial_batch(x_bs, R_bs, T_bs, f_bs, c_bs, k_bs, p_bs):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: 2x1 Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """

    # x_bs, R_bs, T_bs, f_bs, c_bs, k_bs, p_bs
    # = x.unsqueeze(0).expand(2,-1,-1), R.unsqueeze(0).expand(2,-1,-1),
    # T.unsqueeze(0).expand(2,-1,-1), f.unsqueeze(0).expand(2,-1,-1),
    # c.unsqueeze(0).expand(2,-1,-1), k.unsqueeze(0).expand(2,-1,-1),
    # p.unsqueeze(0).expand(2,-1,-1)

    bs, nview, nbins, _ = x_bs.shape
    xcam_bs = torch.matmul(R_bs, x_bs.transpose(2, 3) - T_bs)

    y_bs = xcam_bs[:, :, :2] / (xcam_bs[:, :, 2:] + 1e-5)

    kexp_bs = k_bs.repeat(1, 1, 1, nbins)
    r2_bs = torch.sum(y_bs**2, 2, keepdim=True)
    r2exp_bs = torch.cat([r2_bs, r2_bs**2, r2_bs**3], 2)
    radial_bs = 1 + torch.einsum('bvij,bvij->bvj', kexp_bs, r2exp_bs)

    tan_bs = p_bs[:, :, 0]*y_bs[:, :, 1] + p_bs[:, :, 1]*y_bs[:, :, 0]
    corr_bs = (radial_bs + 2 * tan_bs).unsqueeze(2).expand(-1, -1, 2, -1)

    y_bs = (y_bs * corr_bs + torch.matmul(
        torch.stack([p_bs[:, :, 1], p_bs[:, :, 0]], dim=2), r2_bs))
    ypixel_bs = (f_bs * y_bs) + c_bs
    return ypixel_bs.transpose(2, 3)


def project_pose_batch(x, camera):
    R, T, f, c, k, p = unfold_camera_param_batch(camera, device=x.device)
    return project_point_radial_batch(x, R, T, f, c, k, p)


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera, device=x.device)
    return project_point_radial(x, R, T, f, c, k, p)


def world_to_camera_frame(x, R, T):
    """
    Args
        x: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 3d points in camera coordinates
    """

    R = torch.as_tensor(R, device=x.device, dtype=torch.float32)
    T = torch.as_tensor(T, device=x.device, dtype=torch.float32)
    xcam = torch.mm(R, torch.t(x) - T)
    return torch.t(xcam)


def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """

    R = torch.as_tensor(R, device=x.device, dtype=torch.float32)
    T = torch.as_tensor(T, device=x.device, dtype=torch.float32)
    xcam = torch.mm(torch.t(R), torch.t(x))
    xcam = xcam + T  # rotate and translate
    return torch.t(xcam)


def uv_to_image_frame(uv, camera):
    """

    :param uv: (N, 2)
    :param f: scalar
    :param c: (2, 1)
    :param k:
    :param p:
    :return:
    """
    R, T, f, c, k, p = unfold_camera_param(camera, device=uv.device)
    xy = (uv.t() - c) / f
    return xy.t()
