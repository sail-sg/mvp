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

# ------------------------------------------------------------------------
# Multi-view Pose transformer
# ----------------------------------------------------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# and Deformable Detr
# (https://github.com/fundamentalvision/Deformable-DETR)
# ----------------------------------------------------------------------------------------------------------------------------------------

import copy
# from typing import Optional, List
# import math

import torch
import torch.nn.functional as F
from torch import nn
# from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from lib.models.util.misc import inverse_sigmoid
from lib.models.ops.modules import ProjAttn

import lib.utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
# from utils.transforms import affine_transform_pts_cuda as do_transform
from utils.transforms import \
    affine_transform_pts_cuda_batch as do_transform_batch

import time


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class MvPDecoderLayer(nn.Module):
    def __init__(self, space_size, space_center,
                 img_size, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 detach_refpoints_cameraprj=True,
                 fuse_view_feats='mean', n_views=5,
                 projattn_posembed_mode='use_rayconv'):
        super().__init__()

        # projective attention
        self.proj_attn = ProjAttn(d_model, n_levels,
                                  n_heads, n_points,
                                  projattn_posembed_mode)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model,
                                               n_heads,
                                               dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.grid_size = torch.tensor(space_size)
        self.grid_center = torch.tensor(space_center)

        self.img_size = img_size

        self.detach_refpoints_cameraprj = detach_refpoints_cameraprj

        self.fuse_view_feats = fuse_view_feats
        if self.fuse_view_feats == 'cat_proj':
            self.fuse_view_projction = nn.Linear(d_model*n_views,
                                                 d_model)
        elif self.fuse_view_feats == 'cat_catcoord_proj':
            self.fuse_view_projction = nn.Linear((d_model+2) * n_views,
                                                 d_model)
        elif self.fuse_view_feats == 'cat_catcoord_catref_proj':
            self.fuse_view_projction = nn.Linear((d_model+2) * n_views+3,
                                                 d_model)
        elif self.fuse_view_feats == 'sum_proj':
            self.fuse_view_projction = nn.Linear(d_model, d_model)
        elif self.fuse_view_feats == 'attn_fuse_subtract':
            self.attn_proj = nn.Sequential(*[nn.ReLU(),
                                             nn.Linear(d_model, d_model)])
        elif self.fuse_view_feats == 'cat_attn_proj':
            raise NotImplementedError
        elif self.fuse_view_feats == 'attn_fuse_dot_prod_proj':
            self.fuse_view_projction = nn.Linear(d_model, d_model)
        elif self.fuse_view_feats == 'attn_fuse_subtract_proj':
            self.attn_proj = nn.Sequential(*[nn.ReLU(),
                                             nn.Linear(d_model, d_model)])
            self.fuse_view_projction = nn.Linear(d_model, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def norm2absolute(self, norm_coords):
        device = norm_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = norm_coords * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, tgt, query_pos, reference_points, src_views,
                src_views_with_rayembed, src_spatial_shapes,
                level_start_index, meta, src_padding_mask=None):

        batch_size = query_pos.shape[0]
        device = query_pos.device
        nviews = len(src_views[0])
        # h, w = src_spatial_shapes[0]
        nfeat_level = len(src_views)
        nbins = reference_points.shape[1]
        # bounding = torch.zeros(batch_size,nviews, nbins, device=device)
        # tgt_batch = []
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1),
                              k.transpose(0, 1),
                              tgt.transpose(0, 1))[0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt_expand = tgt.unsqueeze(1).\
            expand((-1, nviews, -1, -1)).flatten(0, 1)
        query_pos_expand = query_pos.unsqueeze(1).\
            expand((-1, nviews, -1, -1)).flatten(0, 1)
        src_padding_mask_expand = torch.cat(src_padding_mask, dim=1)

        ref_points_proj2d_xy_norm = []

        if self.detach_refpoints_cameraprj:
            reference_points = reference_points.detach()

        cam_batch = {}
        for k in meta[0]['camera'].keys():
            cam_batch[k] = []
        for v in range(nviews):
            for k, v in meta[v]['camera'].items():
                cam_batch[k].append(v)
        for k in meta[0]['camera'].keys():
            cam_batch[k] = torch.stack(cam_batch[k], dim=1)

        reference_points_expand = reference_points.\
            unsqueeze(1).expand(-1, nviews, -1, -1, -1)
        reference_points_expand_flatten \
            = reference_points_expand\
            .contiguous().view(batch_size, nviews, nbins, 3)

        reference_points_absolute = self.\
            norm2absolute(reference_points_expand_flatten)
        reference_points_projected2d_xy = \
            cameras.project_pose_batch(reference_points_absolute, cam_batch)

        trans_batch = []
        for i in range(batch_size):
            temp = []
            for v in range(nviews):
                temp.append(
                    torch.as_tensor(
                        get_transform(meta[v]['center'][i],
                                      meta[v]['scale'][i],
                                      0, self.img_size),
                        dtype=torch.float,
                        device=device))
            trans_batch.append(torch.stack(temp))
        trans_batch = torch.stack(trans_batch)
        wh = torch.stack([meta[v]['center'] for v in range(nviews)], dim=1)*2
        bounding \
            = (reference_points_projected2d_xy[..., 0] >= 0) \
            & (reference_points_projected2d_xy[..., 1] >= 0) \
            & (reference_points_projected2d_xy[..., 0] < wh[..., 0:1]) \
            & (reference_points_projected2d_xy[..., 1] < wh[..., 1:2])
        reference_points_projected2d_xy \
            = torch.clamp(reference_points_projected2d_xy, -1.0, wh.max())
        reference_points_projected2d_xy \
            = do_transform_batch(reference_points_projected2d_xy, trans_batch)
        reference_points_projected2d_xy \
            = reference_points_projected2d_xy \
            / torch.tensor(self.img_size, dtype=torch.float, device=device)

        ref_points_expand = reference_points_projected2d_xy\
            .flatten(0, 1).unsqueeze(2)

        ref_points_expand \
            = ref_points_expand.expand(-1, -1, nfeat_level, -1) \
            * src_spatial_shapes.flip(-1).float() \
            / (src_spatial_shapes.flip(-1)-1).float()
        tgt2 = self.proj_attn(
            self.with_pos_embed(tgt_expand, query_pos_expand),
            ref_points_expand, src_views, src_views_with_rayembed,
            src_spatial_shapes, level_start_index, src_padding_mask_expand)
        for id, m in enumerate(meta):
            if 'padding' in m:
                bounding[:, id] = False

        tgt2 = (bounding.unsqueeze(-1) *
                tgt2.view(batch_size, nviews, nbins, -1))
        # various ways to fuse the multi-view feats
        if self.fuse_view_feats == 'mean':
            tgt2 = tgt2.mean(1)
        elif self.fuse_view_feats == 'cat_proj':
            tgt2 = tgt2.permute(0, 2, 1, 3).contiguous()\
                .view(batch_size, nbins, -1)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'cat_catcoord_proj':
            tgt2 = torch.cat(
                [
                    tgt2,
                    torch.stack(
                        ref_points_proj2d_xy_norm).squeeze(-2)], dim=-1)
            tgt2 = tgt2.permute(0, 2, 1, 3)\
                .contiguous()\
                .view(batch_size, nbins, -1)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'cat_catcoord_catref_proj':
            tgt2 = \
                torch.cat(
                    [
                        tgt2,
                        torch.stack(ref_points_proj2d_xy_norm).squeeze(-2)],
                    dim=-1)
            tgt2 = tgt2.permute(0, 2, 1, 3)\
                .contiguous().view(batch_size, nbins, -1)
            tgt2 = torch.cat([tgt2, reference_points.squeeze(-2)], dim=-1)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'sum_proj':
            tgt2 = self.fuse_view_projction(tgt2.sum(1))
        elif self.fuse_view_feats == 'attn_fuse_dot_prod':
            attn_weight = \
                torch.matmul(
                    tgt2.permute(0, 2, 1, 3),
                    tgt.unsqueeze(-1)).softmax(-2)
            tgt2 = (tgt2.transpose(1, 2)*attn_weight).sum(-2)
        elif self.fuse_view_feats == 'attn_fuse_subtract':
            attn_weight = self.attn_proj(tgt2 - tgt.unsqueeze(1))
            tgt2 = (attn_weight*tgt2).sum(1)
        elif self.fuse_view_feats == 'attn_fuse_dot_prod_proj':
            attn_weight = \
                torch.matmul(
                    tgt2.permute(0, 2, 1, 3),
                    tgt.unsqueeze(-1)).softmax(-2)
            tgt2 = (tgt2.transpose(1, 2)*attn_weight).sum(-2)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'attn_fuse_subtract_proj':
            attn_weight = self.attn_proj(tgt2 - tgt.unsqueeze(1))
            tgt2 = (attn_weight*tgt2).sum(1)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'cat_attn_proj':
            raise NotImplementedError
        else:
            raise NotImplementedError
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class MvPDecoder(nn.Module):
    def __init__(self, cfg, decoder_layer,
                 num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.pose_embed = None
        self.class_embed = None

        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)

    def absolute2norm(self, absolute_coords):
        device = absolute_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        norm_coords = (absolute_coords
                       - grid_center
                       + grid_size / 2.0) / grid_size
        return norm_coords

    def norm2absolute(self, norm_coords):
        device = norm_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = norm_coords * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, tgt, reference_points, src_views,
                src_views_with_rayembed, meta, src_spatial_shapes,
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):

        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            output = layer(output, query_pos, reference_points_input,
                           src_views, src_views_with_rayembed,
                           src_spatial_shapes,
                           src_level_start_index, meta, src_padding_mask)

            # hack implementation for iterative pose refinement
            if self.pose_embed is not None:
                tmp = self.pose_embed[lid](output)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), \
                   torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
