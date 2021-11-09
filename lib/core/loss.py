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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F
from lib.utils.cameras import project_pose
# from utils.transforms import get_affine_transform as get_transform
# from utils.transforms import affine_transform_pts_cuda as do_transform


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss


class PerJointMSELoss(nn.Module):
    def __init__(self):
        super(PerJointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target,
                use_target_weight=False, target_weight=None):
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)

            heatmap_pred = output.reshape((batch_size, num_joints, -1))
            heatmap_gt = target.reshape((batch_size, num_joints, -1))
            loss = self.criterion(heatmap_pred.mul(target_weight),
                                  heatmap_gt.mul(target_weight))
        else:
            loss = self.criterion(output, target)

        return loss


class PerJointL1Loss(nn.Module):
    def __init__(self, loss_type):
        super(PerJointL1Loss, self).__init__()
        self.loss_type = loss_type
        # self.criterion = nn.L1Loss(reduction='none')

    def forward(self, output, target,
                use_target_weight=False, target_weight=None, num_boxes=None):
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)

            pred = output.reshape((batch_size, num_joints, -1))
            gt = target.reshape((batch_size, num_joints, -1))
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred.mul(target_weight),
                                 gt.mul(target_weight), reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred.mul(target_weight),
                                  gt.mul(target_weight), reduction='none')
            elif self.loss_type == 'mpjpe':
                loss = (((pred - gt) ** 2).sum(-1) ** (1 / 2))\
                           .mul(target_weight.squeeze(-1)).sum(-1) / \
                           target_weight.squeeze(-1).sum(-1)
                loss = loss.sum()/num_boxes
            else:
                raise NotImplementedError
        else:
            if self.loss_type == 'l1':
                loss = F.l1_loss(output, target, reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(output, target, reduction='none')
            else:
                raise NotImplementedError

        return loss


class PerJointAlignedL1Loss(nn.Module):
    def __init__(self, loss_type):
        super(PerJointAlignedL1Loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, gt, use_target_weight=False, target_weight=None):
        # Do procrustes alignment
        pred_ = compute_similarity_transform(pred, gt)
        if use_target_weight:
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred_.mul(target_weight),
                                 gt.mul(target_weight), reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred_.mul(target_weight),
                                  gt.mul(target_weight), reduction='none')
            else:
                raise NotImplementedError
        else:
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred_, gt, reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred_, gt, reduction='none')
            else:
                raise NotImplementedError
        return loss


class PerBoneL1Loss(nn.Module):
    def __init__(self, loss_type):
        super(PerBoneL1Loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, gt, use_target_weight=False, target_weight=None):
        LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
                   [10, 11], [2, 6], [2, 12],
                   [6, 7], [7, 8], [12, 13], [13, 14]]
        gt_bone_vector = gt[:, LIMBS15]
        gt_bone_vector[:, :, 0] = \
            gt_bone_vector[:, :, 0] - gt_bone_vector[:, :, 1]
        gt_bone_vector = gt_bone_vector[:, :, 0]

        pred_bone_vector = pred[:, LIMBS15]
        pred_bone_vector[:, :, 0] = \
            pred_bone_vector[:, :, 0] - pred_bone_vector[:, :, 1]
        pred_bone_vector = pred_bone_vector[:, :, 0]

        target_weight_bone_vector = target_weight[:, LIMBS15]
        target_weight_bone_vector[:, :, 0] = \
            target_weight_bone_vector[:, :, 0] * \
            target_weight_bone_vector[:, :, 1]
        target_weight_bone_vector = target_weight_bone_vector[:, :, 0]

        if use_target_weight:
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred_bone_vector.
                                 mul(target_weight_bone_vector),
                                 gt_bone_vector.mul(target_weight_bone_vector),
                                 reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred_bone_vector.
                                  mul(target_weight_bone_vector),
                                  gt_bone_vector.
                                  mul(target_weight_bone_vector),
                                  reduction='none')
            else:
                raise NotImplementedError
        else:
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred_bone_vector,
                                 gt_bone_vector, reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred_bone_vector,
                                  gt_bone_vector, reduction='none')
            else:
                raise NotImplementedError
        return loss


class PerProjectionL1Loss(nn.Module):
    def __init__(self, loss_type):
        super(PerProjectionL1Loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, gt, cameras, center, scale,
                img_size, use_target_weight=False, target_weight=None):
        assert pred.size(0) == gt.size(0)
        pred_multi_view = pred[None].repeat(len(cameras), 1, 1, 1)
        gt_multi_view = gt[None].repeat(len(cameras), 1, 1, 1)
        n_joints = pred.size(1)

        projection_pred = torch.cat(
            [project_pose(pred_view.view(-1, 3), cam)
             for pred_view, cam in zip(pred_multi_view, cameras)], 0)
        projection_gt = torch.cat(
            [project_pose(gt_view.view(-1, 3), cam)
             for gt_view, cam in zip(gt_multi_view, cameras)], 0)
        weights_2d = torch.cat(
            [weight for weight in target_weight], 0)

        projection_pred = projection_pred.view(-1, n_joints, 2)
        projection_gt = projection_gt.view(-1, n_joints, 2)

        if use_target_weight:
            if self.loss_type == 'l1':
                loss = F.l1_loss(projection_pred.mul(weights_2d),
                                 projection_gt.mul(weights_2d),
                                 reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(projection_pred.mul(weights_2d),
                                  projection_gt.mul(weights_2d),
                                  reduction='none')
            else:
                raise NotImplementedError
        else:
            if self.loss_type == 'l1':
                loss = F.l1_loss(projection_pred,
                                 projection_gt, reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(projection_pred,
                                  projection_gt, reduction='none')
            else:
                raise NotImplementedError

        return loss


def compute_similarity_transform(S1, S2):
    '''
    A port of MATLAB's `procrustes` function to PyTorch.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[1] != 3 and S1.shape[1] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = torch.mean(S1, dim=-1, keepdim=True)
    mu2 = torch.mean(S2, dim=-1, keepdim=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1)) + 1e-8

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det((U.bmm(V.permute(0, 2, 1)))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
