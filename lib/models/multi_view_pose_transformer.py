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
# and Deformable Detr (https://github.com/fundamentalvision/Deformable-DETR)
# ----------------------------------------------------------------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from models import pose_resnet
# from core.loss import PerJointMSELoss

from models.mvp_decoder import MvPDecoderLayer, MvPDecoder
from models.ops.modules import ProjAttn
from torch.nn.init import xavier_uniform_, constant_, normal_
from models.position_encoding import PositionEmbeddingSine, \
    get_rays_new, get_2d_coords
from models.util.misc import (
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import copy
import torch.nn.functional as F
# import h5py
from models.matcher import HungarianMatcher
from core.loss import PerJointL1Loss, PerBoneL1Loss, PerProjectionL1Loss

# import os
# import os.path as osp
# import numpy as np

import math


def sigmoid_focal_loss(inputs, targets, num_samples,
                       alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection:
    https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as
        inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_samples


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    Args:
        input_dim: The dimension of input feature.
        hidden_dim: The dimension of intermediate feature.
        output_dim: The dimension of output.
        num_layers: number of layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h,
                                            h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiviewPosetransformer(nn.Module):
    """
    Multi-view Pose Transformer Module
    Args:
        cfg: the config file
    """
    def __init__(self, backbone, cfg):
        super(MultiviewPosetransformer, self).__init__()
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_instance = cfg.DECODER.num_instance
        self.backbone = backbone
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET
        self.reference_points = nn.Linear(cfg.DECODER.d_model, 3)
        self.reference_feats = nn.Linear(
            cfg.DECODER.d_model * len(cfg.DECODER.use_feat_level)
            * cfg.DATASET.CAMERA_NUM,
            cfg.DECODER.d_model)  # 256*feat_level*num_views

        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)

        decoder_layer = MvPDecoderLayer(cfg.MULTI_PERSON.SPACE_SIZE,
                                        cfg.MULTI_PERSON.SPACE_CENTER,
                                        cfg.NETWORK.IMAGE_SIZE,
                                        cfg.DECODER.d_model,
                                        cfg.DECODER.dim_feedforward,
                                        cfg.DECODER.dropout,
                                        cfg.DECODER.activation,
                                        cfg.DECODER.num_feature_levels,
                                        cfg.DECODER.nhead,
                                        cfg.DECODER.dec_n_points,
                                        cfg.DECODER.
                                        detach_refpoints_cameraprj_firstlayer,
                                        cfg.DECODER.fuse_view_feats,
                                        cfg.DATASET.CAMERA_NUM,
                                        cfg.DECODER.projattn_posembed_mode)
        self.decoder = MvPDecoder(cfg, decoder_layer,
                                  cfg.DECODER.num_decoder_layers,
                                  cfg.DECODER.return_intermediate_dec)

        num_queries = cfg.DECODER.num_instance*cfg.DECODER.num_keypoints

        self.query_embed_type = cfg.DECODER.query_embed_type
        if self.query_embed_type == 'person_joint':
            self.joint_embedding = nn.Embedding(cfg.DECODER.num_keypoints,
                                                cfg.DECODER.d_model * 2)
            self.instance_embedding = nn.Embedding(cfg.DECODER.num_instance,
                                                   cfg.DECODER.d_model * 2)
        elif self.query_embed_type == 'image_person_joint':
            self.image_embedding = nn.Embedding(1, cfg.DECODER.d_model * 2)
            self.joint_embedding = nn.Embedding(cfg.DECODER.num_keypoints,
                                                cfg.DECODER.d_model * 2)
            self.instance_embedding = nn.Embedding(cfg.DECODER.num_instance,
                                                   cfg.DECODER.d_model * 2)
        elif self.query_embed_type == 'per_joint':
            self.query_embed = nn.Embedding(num_queries,
                                            cfg.DECODER.d_model * 2)

        N_steps = cfg.DECODER.d_model // 2
        self.pos_encoding = PositionEmbeddingSine(N_steps, normalize=True)
        self.view_embed = nn.Parameter(torch.Tensor(cfg.DATASET.CAMERA_NUM,
                                                    cfg.DECODER.d_model))
        self._reset_parameters()

        # We can use gt camera for projection,
        # so we dont need to regress camera param
        num_pred = self.decoder.num_layers
        num_classes = 2

        self.class_embed = nn.Linear(cfg.DECODER.d_model, num_classes)
        self.pose_embed = MLP(cfg.DECODER.d_model, cfg.DECODER.d_model,
                              3, cfg.DECODER.pose_embed_layer)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.pose_embed.layers[-1].bias.data, 0)

        if cfg.DECODER.with_pose_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.pose_embed = _get_clones(self.pose_embed, num_pred)
            self.decoder.pose_embed = self.pose_embed
        else:
            nn.init.constant_(self.pose_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed
                                              for _ in range(num_pred)])
            self.pose_embed = nn.ModuleList([self.pose_embed
                                             for _ in range(num_pred)])
            self.decoder.pose_embed = None

        matcher = HungarianMatcher(match_coord=cfg.DECODER.match_coord,
                                   cost_class=2.,
                                   cost_pose=5.)

        weight_dict = {
            'loss_ce': cfg.DECODER.loss_weight_loss_ce,
            'loss_pose_perjoint': cfg.DECODER.loss_pose_perjoint,
            'loss_pose_perbone': cfg.DECODER.loss_pose_perbone}

        focal_alpha = 0.25
        losses = ['joints', 'labels', 'cardinality']
        self.aux_loss = cfg.DECODER.aux_loss

        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(num_pred - 1):
                aux_weight_dict.update({k + f'_{i}': v
                                        for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v
                                    for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.criterion = SetCriterion(num_classes, matcher, weight_dict,
                                      losses, cfg,
                                      focal_alpha=focal_alpha,
                                      root_idx=self.root_id)
        device = torch.device('cuda')
        self.criterion.to(device)
        self.pred_conf_threshold = cfg.DECODER.pred_conf_threshold
        self.pred_class_fuse = cfg.DECODER.pred_class_fuse
        self.num_joints = cfg.DECODER.num_keypoints

        self.level_embed = nn.Parameter(torch.Tensor(3, cfg.DECODER.d_model))

        self.projattn_posembed_mode = cfg.DECODER.projattn_posembed_mode

        self.use_feat_level = cfg.DECODER.use_feat_level
        self.query_adaptation = cfg.DECODER.query_adaptation

        self.convert_joint_format_indices = \
            cfg.DECODER.convert_joint_format_indices

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, ProjAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.view_embed)

    # convert abosolute joint coordinates to normalized (0-1) coordinates
    def absolute2norm(self, absolute_coords):
        device = absolute_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        norm_coords = (absolute_coords - grid_center +
                       grid_size / 2.0) / grid_size
        return norm_coords

    # convert normalized (0-1) joint coordinates to abosolute coordinates
    def norm2absolute(self, norm_coords):
        device = norm_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = norm_coords * grid_size + grid_center - grid_size / 2.0
        return loc

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_poses': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def collate_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)

    def forward(self, views=None, meta=None):
        if views is not None:
            all_feats = self.backbone(torch.cat(views, dim=0),
                                      self.use_feat_level)
            all_feats = all_feats[::-1]
        batch, _, imageh, imagew = views[0].shape
        nview = len(views)

        cam_R = torch.stack([m['camera_R'] for m in meta], dim=1)
        cam_T = torch.stack([m['camera_standard_T'] for m in meta], dim=1)
        cam_K = torch.stack([m['camera_Intri'] for m in meta], dim=1)
        affine_trans = torch.stack([m['affine_trans'] for m in meta], dim=1)
        cam_K_crop = \
            torch.bmm(
                self.collate_first_two_dims(affine_trans),
                self.collate_first_two_dims(cam_K)).view(batch, nview, 3, 3)
        nfeat_level = len(all_feats)
        camera_rays = []
        # get pos embed, camera ray or 2d coords
        for lvl in range(nfeat_level):
            # this can be compute only once, without iterating over views
            if self.projattn_posembed_mode == 'use_rayconv':
                camera_rays.append(
                    get_rays_new(self.image_size,
                                 all_feats[lvl].shape[2],
                                 all_feats[lvl].shape[3],
                                 cam_K_crop, cam_R, cam_T).flatten(0, 1))
            elif self.projattn_posembed_mode == 'use_2d_coordconv':
                camera_rays.append(
                    get_2d_coords(self.image_size,
                                  all_feats[lvl].shape[2],
                                  all_feats[lvl].shape[3],
                                  cam_K_crop, cam_R, cam_T).flatten(0, 1))

        src_flatten_views = []
        mask_flatten_views = []
        spatial_shapes_views = []

        for lvl, src in enumerate(all_feats):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_views.append(spatial_shape)
            mask = src.new_zeros(bs, h, w).bool()
            mask_flatten_views.append(mask)
            mask = mask.flatten(1)
            src_flatten_views.append(src)

        spatial_shapes_views = \
            torch.as_tensor(spatial_shapes_views,
                            dtype=torch.long,
                            device=mask.device)
        level_start_index_views = \
            torch.cat((mask.new_zeros((1, ), dtype=torch.long),
                       torch.as_tensor(spatial_shapes_views,
                                       dtype=torch.long,
                                       device=mask.device)
                       .prod(1).cumsum(0)[:-1]))
        valid_ratios_views = torch.stack([self.get_valid_ratio(m)
                                          for m in mask_flatten_views], 1)
        mask_flatten_views = [m.flatten(1) for m in mask_flatten_views]

        # query embedding scheme
        if self.query_embed_type == 'person_joint':
            # person embedding + joint embedding
            joint_embeds = self.joint_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            query_embeds = (joint_embeds + instance_embeds).flatten(0, 1)

        if self.query_embed_type == 'image_person_joint':
            # image_embedding + person embedding + joint embedding
            joint_embeds = self.joint_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            query_embeds = (joint_embeds + instance_embeds).flatten(0, 1)
            query_embeds += self.image_embedding.weight

        elif self.query_embed_type == 'per_joint':
            # per joint embedding
            query_embeds = self.query_embed.weight

        query_embed, tgt = torch.split(query_embeds, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch, -1, -1)
        tgt = tgt.unsqueeze(0).expand(batch, -1, -1)

        # query adaptation
        if self.query_adaptation:
            feats_0 = F.adaptive_avg_pool2d(all_feats[0], (1, 1))
            feats_1 = F.adaptive_avg_pool2d(all_feats[1], (1, 1))
            feats_2 = F.adaptive_avg_pool2d(all_feats[2], (1, 1))
            feats = torch.cat((feats_0, feats_1, feats_2),
                              dim=1).squeeze().view(1, -1)
            ref_feats = self.reference_feats(feats).unsqueeze(0)
            reference_points = self.reference_points(
                query_embed + ref_feats).sigmoid()
        else:
            reference_points = self.reference_points(query_embed).sigmoid()

        init_reference = reference_points  # B x 150 x 3

        hs, inter_references = \
            self.decoder(tgt, reference_points, src_flatten_views,
                         camera_rays,
                         meta=meta, src_spatial_shapes=spatial_shapes_views,
                         src_level_start_index=level_start_index_views,
                         src_valid_ratios=valid_ratios_views,
                         query_pos=query_embed,
                         src_padding_mask=mask_flatten_views)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # mean after sigmoid
            if self.pred_class_fuse == 'mean':
                outputs_class = self.class_embed[lvl](hs[lvl]).\
                    view(batch, self.num_instance, self.num_joints, -1).\
                    sigmoid().mean(2)
                outputs_class = inverse_sigmoid(outputs_class)
            elif self.pred_class_fuse == 'feat_mean_pool':
                outputs_class = self.class_embed[lvl](hs[lvl])\
                    .view(batch, self.num_instance, self.num_joints, -1)\
                    .mean(2)
            elif self.pred_class_fuse == 'feat_max_pool':
                outputs_class = \
                    self.class_embed[lvl](
                        hs[lvl].view(batch,
                                     self.num_instance,
                                     self.num_joints, -1).max(2)[0])
            else:
                raise NotImplementedError
            tmp = self.pose_embed[lvl](hs[lvl])
            tmp += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)

            # convert panoptic joints to shelf/campus
            if self.convert_joint_format_indices is not None:
                outputs_coord = \
                    outputs_coord.view(batch,
                                       self.num_instance,
                                       self.num_joints, -1)
                outputs_coord \
                    = outputs_coord[..., self.convert_joint_format_indices, :]
                outputs_coord = outputs_coord.flatten(1, 2)

            outputs_coords.append({'outputs_coord': outputs_coord})

        out = {'pred_logits': outputs_classes[-1],
               'pred_poses': outputs_coords[-1]}

        if self.aux_loss:
            out['aux_outputs'] = \
                self._set_aux_loss(outputs_classes, outputs_coords)

        if self.training and 'joints_3d' in meta[0] \
                and 'joints_3d_vis' in meta[0]:
            meta[0]['roots_3d_norm'] = \
                self.absolute2norm(meta[0]['roots_3d'].float())
            meta[0]['joints_3d_norm'] = \
                self.absolute2norm(meta[0]['joints_3d'].float())
            loss_dict = self.criterion(out, meta)
            return out, loss_dict

        return out


class SetCriterion(nn.Module):
    """
    The process happens in two steps:
        1) we compute hungarian assignment
        between ground truth poses and the outputs of the model
        2) we supervise each pair of matched
        ground-truth / prediction (supervise class and pose)
    """
    def __init__(self, num_classes, matcher,
                 weight_dict, losses, cfg, focal_alpha=0.25, root_idx=2):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories,
            omitting the special no-object category

            matcher: module able to compute a
            matching between targets and proposals

            weight_dict: dict containing as key the names of
            the losses and as values their relative weight.

            losses: list of all the losses to be applied.
            See get_loss for list of available losses.

            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.matcher.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.matcher.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.root_idx = root_idx

        self.eos_coef = 0.1
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.criterion_pose_perjoint = \
            PerJointL1Loss(cfg.DECODER.loss_joint_type)

        self.use_loss_pose_perbone = cfg.DECODER.use_loss_pose_perbone
        self.use_loss_pose_perprojection = \
            cfg.DECODER.use_loss_pose_perprojection

        if self.use_loss_pose_perbone:
            # loss for bone vector prediction
            self.criterion_pose_perbone = \
                PerBoneL1Loss(cfg.DECODER.loss_joint_type)

        if self.use_loss_pose_perprojection:
            # loss for projected joint
            self.criterion_pose_perprojection = \
                PerProjectionL1Loss(cfg.DECODER.loss_joint_type)

        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)

        self.loss_pose_normalize = cfg.DECODER.loss_pose_normalize
        self.pred_conf_threshold = cfg.DECODER.pred_conf_threshold

        self.num_person = cfg.DECODER.num_instance

    def absolute2norm(self, absolute_coords):
        device = absolute_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        norm_coords = (absolute_coords -
                       grid_center + grid_size / 2.0) / grid_size
        return norm_coords

    def norm2absolute(self, norm_coords):
        device = norm_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = norm_coords * grid_size + grid_center - grid_size / 2.0
        return loc

    def loss_labels(self, outputs, meta, indices, num_samples, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key
        "labels" containing a tensor of dim [nb_target_poses]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([src_logits.new([1]*i).long()
                                      for i in meta[0]['num_person']])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = \
            torch.zeros([src_logits.shape[0],
                         src_logits.shape[1],
                         src_logits.shape[2] + 1],
                        dtype=src_logits.dtype,
                        layout=src_logits.layout,
                        device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits,
                                     target_classes_onehot,
                                     num_samples,
                                     alpha=self.focal_alpha,
                                     gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, meta, indices, num_samples):
        """ Compute the cardinality error, ie the
        absolute error in the number of predicted non-empty poses
        This is not really a loss, it is intended for
        logging purposes only. It doesn't propagate gradients
        """
        threshold = self.pred_conf_threshold
        # gt_3d = meta[0]['joints_3d_norm'].float()
        # num_joints = gt_3d.shape[2]
        # num_cand = gt_3d.shape[1]
        # bs = outputs['pred_logits'].shape[0]
        # num_person = meta[0]['num_person']

        pred_logits = outputs['pred_logits']
        # device = pred_logits.device
        tgt_lengths = meta[0]['num_person']
        # Count the number of predictions that
        # are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.sigmoid()[:, :, 1] > threshold).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_poses(self, outputs,
                   meta, indices, num_samples, output_abs_coord=False):
        """Compute the losses related to the bounding poses,
        the L1 regression loss and the GIoU loss
           targets dicts must contain the key "poses"
           containing a tensor of dim [nb_target_poses, 4]
           The target poses are expected in format
           (center_x, center_y, h, w), normalized by the image size.
        """
        # assert 'pred_poses' in outputs
        idx = self._get_src_permutation_idx(indices)
        idx_target = self._get_tgt_permutation_idx(indices)

        gt_3d = meta[0]['joints_3d_norm'].float()
        num_joints = gt_3d.shape[2]
        bs = outputs.shape[0]
        num_queries = self.num_person

        src_poses = outputs.view(bs, num_queries, num_joints, 3)[idx]
        target_poses = \
            torch.cat([t[i]
                       for t, (_, i) in
                       zip(meta[0]['joints_3d_norm'], indices)], dim=0)

        weights_3d = meta[0]['joints_3d_vis'][idx_target][:, :, 0:1].float()

        if not self.loss_pose_normalize:
            target_poses = self.norm2absolute(target_poses)
            if not output_abs_coord:
                src_poses = self.norm2absolute(src_poses)

        loss_cord = \
            self.criterion_pose_perjoint(
                src_poses, target_poses, True, weights_3d, num_samples)
        losses = {}

        if not self.criterion_pose_perjoint.loss_type == 'mpjpe':
            losses['loss_pose_perjoint'] = \
                (loss_cord.sum(0)/num_samples).mean()
        else:
            losses['loss_pose_perjoint'] = loss_cord

        if self.use_loss_pose_perbone:
            loss_cord = self.criterion_pose_perbone(
                src_poses, target_poses, True, weights_3d)
            losses['loss_pose_perbone'] = \
                (loss_cord.sum(0) / num_samples).mean()

        if self.use_loss_pose_perprojection:
            idx_target = [idx_target] * len(meta)
            weights_2d = [meta_view['joints_vis'][idx_view][:, :, 0:1]
                          for meta_view, idx_view in zip(meta, idx_target)]
            cameras = [meta_view['camera'] for meta_view in meta]

            if self.loss_pose_normalize:
                src_poses = self.norm2absolute(src_poses)
                target_poses = self.norm2absolute(target_poses)
            loss_cord = \
                self.criterion_pose_perprojection(
                    src_poses,
                    target_poses,
                    cameras,
                    meta[0]['center'][0], meta[0]['scale'][0],
                    self.img_size,
                    True, weights_2d)
            num_views = len(cameras)
            loss_cord \
                = loss_cord.view(-1, num_views, num_joints, 2)[
                  :, loss_cord.new(['padding' not in m for m in meta]).bool()]
            loss_cord = loss_cord.flatten(0, 1)
            loss_pose_perprojection = (loss_cord.sum(0) /
                                       (num_samples * num_views)).mean()
            if loss_pose_perprojection.item() > 1e5:
                loss_pose_perprojection = loss_pose_perprojection * 0.0
            losses['loss_pose_perprojection'] = loss_pose_perprojection

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets,
                 indices, num_samples, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'joints': self.loss_poses,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return \
            loss_map[loss](outputs, targets, indices, num_samples, **kwargs)

    def forward(self, outputs, meta):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors,
             see the output specification of the model for the format

             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the
                      losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items()
                               if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the
        # outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, meta)

        # Compute the average number of target
        # poses accross all nodes, for normalization purposes
        num_samples = sum(meta[0]['num_person'])
        num_samples = \
            torch.as_tensor([num_samples],
                            dtype=torch.float,
                            device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_samples)
        num_samples = torch.clamp(num_samples /
                                  get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'joints':
                losses.update(
                    self.get_loss(loss,
                                  outputs["pred_poses"]['outputs_coord'],
                                  meta, indices, num_samples, **kwargs))
            else:
                losses.update(self.get_loss(loss, outputs, meta,
                                            indices, num_samples, **kwargs))

        # In case of auxiliary losses, we repeat
        # this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, meta)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False

                    if loss == 'joints':
                        l_dict = \
                            self.get_loss(
                                loss,
                                aux_outputs["pred_poses"]['outputs_coord'],
                                meta, indices,
                                num_samples,
                                **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

                    else:
                        l_dict = \
                            self.get_loss(
                                loss,
                                aux_outputs,
                                meta,
                                indices,
                                num_samples,
                                **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(meta)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss,
                                       enc_outputs,
                                       bin_targets,
                                       indices,
                                       num_samples,
                                       **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def get_mvp(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(
            cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiviewPosetransformer(backbone, cfg)
    return model
