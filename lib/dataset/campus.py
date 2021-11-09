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

import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import scipy.io as scio
import logging
import copy
import os
from collections import OrderedDict

from dataset.JointsDataset import JointsDataset
from utils.cameras_cpu import project_pose

CAMPUS_JOINTS_DEF = {
    'Right-Ankle': 0,
    'Right-Knee': 1,
    'Right-Hip': 2,
    'Left-Hip': 3,
    'Left-Knee': 4,
    'Left-Ankle': 5,
    'Right-Wrist': 6,
    'Right-Elbow': 7,
    'Right-Shoulder': 8,
    'Left-Shoulder': 9,
    'Left-Elbow': 10,
    'Left-Wrist': 11,
    'Bottom-Head': 12,
    'Top-Head': 13
}

LIMBS = [
    [0, 1],
    [1, 2],
    [3, 4],
    [4, 5],
    [2, 3],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [2, 8],
    [3, 9],
    [8, 12],
    [9, 12],
    [12, 13]
]


class Campus(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        self.pixel_std = 200.0
        self.joints_def = CAMPUS_JOINTS_DEF
        super().__init__(cfg, image_set, is_train, transform)
        self.limbs = LIMBS
        self.num_joints = len(CAMPUS_JOINTS_DEF)
        self.cam_list = [0, 1, 2]
        self.num_views = len(self.cam_list)

        if self.is_train:
            # training set
            # self.frame_range = list(range(0, 350)) \
            #                    + list(range(471, 650)) \
            #                    + list(range(751, 1900))
            # augmented training set
            self.frame_range = list(range(0, 350)) + list(range(471, 650)) + list(range(751, 1900)) + list(
                range(471, 520)) * 2 + list(range(751, 1200)) * 2
        else:
            # self.frame_range = list(range(0, 350))
            # + list(range(471, 650)) + list(range(751, 1900))
            self.frame_range = list(range(350, 471)) \
                               + list(range(650, 751))

        # self.pred_pose2d = self._get_pred_pose2d()
        self.db = self._get_db(osp.join(
            './data/CampusSeq1/pesudo_gt/', cfg.DATASET.PESUDO_GT))

        self.db_size = len(self.db)

    # def _get_pred_pose2d(self):
    #     file = os.path.join(self.dataset_root,
    #                         "pred_campus_maskrcnn_hrnet_coco.pkl")
    #     with open(file, "rb") as pfile:
    #         logging.info("=> load {}".format(file))
    #         pred_2d = pickle.load(pfile)
    #
    #     return pred_2d

    def _get_db(self, pesudo_gt_path):
        width = 360
        height = 288

        db = []
        cameras = self._get_cam()

        datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
        data = scio.loadmat(datafile)
        actor_3d = np.array(
            np.array(data['actor3D'].tolist()).tolist()).squeeze()

        num_person = len(actor_3d)

        if self.is_train:
            with open(pesudo_gt_path, 'rb') as handle:
                gt_voxelpose_infered = pickle.load(handle)
            print(dir)
            filtered_frame_range = []
            for i in self.frame_range:
                image = osp.join("Camera" + '0',
                                 "campus4-c{0}-{1:05d}.png".format(0, i))
                if image.split('/')[-1] in gt_voxelpose_infered:
                    filtered_frame_range.append(i)
            self.frame_range = filtered_frame_range

        for i in self.frame_range:
            for k, cam in cameras.items():
                image = osp.join("Camera" + k,
                                 "campus4-c{0}-{1:05d}.png".format(k, i))

                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_poses_vis = []
                all_poses_conf = []
                if self.is_train:
                    for pose3d in \
                            gt_voxelpose_infered[
                                "campus4-c{0}-{1:05d}.png".format(0, i)]:
                        if len(pose3d[0]) > 0:
                            conf = pose3d[:, 3]
                            pose3d = pose3d[:, :3]
                            all_poses_conf.append(conf)
                            all_poses_3d.append(pose3d)
                            all_poses_vis_3d.append(
                                np.ones((self.num_joints, 3)))

                            pose2d = project_pose(pose3d, cam)

                            x_check = \
                                np.bitwise_and(pose2d[:, 0] >= 0,
                                               pose2d[:, 0] <= width - 1)
                            y_check = \
                                np.bitwise_and(pose2d[:, 1] >= 0,
                                               pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)

                            joints_vis = np.ones((len(pose2d), 1))
                            joints_vis[np.logical_not(check)] = 0
                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(
                                        joints_vis, (-1, 1)), 2, axis=1))
                else:
                    for person in range(num_person):
                        pose3d = actor_3d[person][i] * 1000.0
                        if len(pose3d[0]) > 0:
                            all_poses_3d.append(pose3d)
                            all_poses_conf.append(
                                np.array([person]*14))
                            all_poses_vis_3d.append(
                                np.ones((self.num_joints, 3)))

                            pose2d = project_pose(pose3d, cam)

                            x_check = \
                                np.bitwise_and(pose2d[:, 0] >= 0,
                                               pose2d[:, 0] <= width - 1)
                            y_check = \
                                np.bitwise_and(pose2d[:, 1] >= 0,
                                               pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)

                            joints_vis = np.ones((len(pose2d), 1))
                            joints_vis[np.logical_not(check)] = 0
                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(
                                        joints_vis, (-1, 1)), 2, axis=1))

                # pred_index = '{}_{}'.format(k, i)
                # preds = self.pred_pose2d[pred_index]
                # preds = [np.array(p["pred"]) for p in preds]

                # add standard T
                cam['standard_T'] = np.dot(-cam['R'], cam['T'])

                db.append({
                    'image': osp.join(self.dataset_root, image),
                    'joints_3d': all_poses_3d,
                    'joints_3d_vis': all_poses_vis_3d,
                    'joints_2d': all_poses,
                    'joints_2d_vis': all_poses_vis,
                    'camera': cam,
                    # 'pred_pose2d': preds
                })
        return db

    def _get_cam(self):
        cam_file = osp.join(self.dataset_root, "calibration_campus.json")
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

        return cameras

    def __getitem__(self, idx):
        input, meta = [], []
        for k in range(self.num_views):
            i, m = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            meta.append(m)
        return input, meta

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds, recall_threshold=500):
        datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
        data = scio.loadmat(datafile)
        actor_3d = np.array(np.array(
            data['actor3D'].tolist()).tolist()).squeeze()
        num_person = len(actor_3d)
        total_gt = 0
        match_gt = 0

        limbs = [[0, 1], [1, 2], [3, 4], [4, 5],
                 [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
        correct_parts = np.zeros(num_person)
        total_parts = np.zeros(num_person)
        alpha = 0.5
        bone_correct_parts = np.zeros((num_person, 10))

        for i, fi in enumerate(self.frame_range):
            pred_coco = preds[i].copy()
            pred_coco = pred_coco[pred_coco[:, 0, 3] >= 0, :, :3]
            pred = np.stack([p for p in
                             copy.deepcopy(pred_coco[:, :, :3])])

            for person in range(num_person):
                gt = actor_3d[person][fi] * 1000.0
                if len(gt[0]) == 0:
                    continue

                mpjpes = np.mean(np.sqrt(
                    np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person] += 1
                        bone_correct_parts[person, j] += 1
                pred_hip = (pred[min_n, 2, 0:3] + pred[min_n, 3, 0:3]) / 2.0
                gt_hip = (gt[2] + gt[3]) / 2.0
                total_parts[person] += 1
                error_s = np.linalg.norm(pred_hip - gt_hip)
                error_e = np.linalg.norm(pred[min_n, 12, 0:3] - gt[12])
                limb_length = np.linalg.norm(gt_hip - gt[12])
                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person] += 1
                    bone_correct_parts[person, 9] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        bone_group = OrderedDict(
            [('Head', [8]), ('Torso', [9]), ('Upper arms', [5, 6]),
             ('Lower arms', [4, 7]),
             ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])
        bone_person_pcp = OrderedDict()
        for k, v in bone_group.items():
            bone_person_pcp[k] = \
                np.sum(
                    bone_correct_parts[:, v], axis=-1) \
                / (total_parts / 10 * len(v) + 1e-8)

        return \
            actor_pcp, \
            avg_pcp, \
            bone_person_pcp, \
            match_gt / (total_gt + 1e-8)

    @staticmethod
    def coco2campus3D(coco_pose):
        """
        transform coco order(our method output)
        3d pose to shelf dataset order with interpolation
        :param coco_pose: np.array with shape 17x3
        :return: 3D pose in campus order with shape 14x3
        """
        campus_pose = np.zeros((14, 3))
        coco2campus = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        campus_pose[0: 12] += coco_pose[coco2campus]

        mid_sho = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
        head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear

        head_bottom = (mid_sho + head_center) / 2  # nose and head center
        head_top = head_bottom + (head_center - head_bottom) * 2
        campus_pose[12] += head_bottom
        campus_pose[13] += head_top

        return campus_pose
