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

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the
    targets and the predictions of the network

    For efficiency reasons, the targets don't include the
    no_object. Because of this, in general,
    there are more predictions than targets. In this case,
    we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 match_coord,
                 cost_class: float = 1,
                 cost_pose: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the
            classification error in the matching cost
            cost_pose: This is the relative weight of the L1
            error of the pose coordinates in the matching cost
            cost_giou: This is the relative weight of the giou
            loss of the pose in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_pose = cost_pose
        self.cost_giou = cost_giou
        self.match_coord = match_coord
        assert cost_class != 0 or cost_pose != 0 or cost_giou != 0, \
            "all costs cant be 0"

    def pose_dist(self, x1, x2, dist='per_joint_mean'):
        if dist == 'per_joint_mean':
            return torch.cdist(x1, x2, p=1)

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

    def forward(self, outputs, meta):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim
                 [batch_size, num_queries, num_classes]
                 with the classification logits

                 "pred_poses": Tensor of dim
                 [batch_size, num_queries, 4]
                 with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size),
            where each target is a dict containing:

                 "labels": Tensor of dim [num_target_poses]
                 (where num_target_poses is the number of ground-truth
                 objects in the target) containing the class labels

                 "poses": Tensor of dim [num_target_poses, 4]
                 containing the target box coordinates

        Returns:
            A list of size batch_size, containing
            tuples of (index_i, index_j) where:
                - index_i is the indices of the
                selected predictions (in order)

                - index_j is the indices of the
                corresponding selected targets (in order)

            For each batch element, it holds:
                len(index_i) = len(index_j)
                = min(num_queries, num_target_poses)
        """
        # gt_3d_root = meta[0]['roots_3d_norm'].float()
        gt_3d = meta[0]['joints_3d_norm'].float()
        num_person_gt = meta[0]['num_person']
        num_joints = gt_3d.shape[2]
        bs, num_queries = outputs["pred_logits"].shape[:2]
        with torch.no_grad():

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            # [batch_size * num_queries, 4]
            out_pose = outputs["pred_poses"]['outputs_coord'].flatten(0, 1)
            # [batch_size * num_cand, num_joints 4]
            out_pose = out_pose.view(bs * num_queries, num_joints, -1)
            # convert to absolute coord for matching
            if self.match_coord == 'abs':
                out_pose = self.norm2absolute(out_pose)

            # Also concat the target labels and poses
            tgt_pose = torch.cat([gt_3d[i, :num_person_gt[i]].
                                 reshape(num_person_gt[i], num_joints, -1)
                                  for i in range(len(num_person_gt))])
            tgt_ids = torch.cat([tgt_pose.new([1]*num_person_gt[i])
                                 for i in range(len(num_person_gt))]).long()
            # convert to absolute coord for matching
            if self.match_coord == 'abs':
                tgt_pose = self.norm2absolute(tgt_pose)

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                    (1 - alpha)
                    * (out_prob ** gamma)
                    * (-(1 - out_prob + 1e-8).log()))
            pos_cost_class = (
                    alpha
                    * ((1 - out_prob) ** gamma)
                    * (-(out_prob + 1e-8).log()))
            cost_class \
                = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between poses
            cost_pose = self.pose_dist(out_pose.view(bs * num_queries, -1),
                                       tgt_pose.view(sum(num_person_gt), -1))
            # scale down to match cls cost
            if self.match_coord == 'abs':
                cost_pose = 0.01*cost_pose

            # Final cost matrix
            C = self.cost_pose * cost_pose + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [v.item() for v in num_person_gt]
            indices = [linear_sum_assignment(c[i])
                       for i, c in enumerate(C.split(sizes, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64),
                     torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_pose=args.set_cost_pose,
                            cost_giou=args.set_cost_giou)
