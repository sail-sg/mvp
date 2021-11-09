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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
# import json_tricks as json
import pickle
# import scipy.io as scio
# import logging
import copy
import os
# from collections import OrderedDict

from dataset.JointsDataset import JointsDataset
from lib.utils.cameras_cpu import camera_to_world_frame, project_pose
# import cv2

INF = 1e8
JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]


H36M_TO_PANOPTIC = [8, 9, 0, 11, 12, 13, 4, 5, 6, 14, 15, 16, 1, 2, 3]


class H36M(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)

        self.db_file = 'h36m_quickload_{}.pkl'.format(self.image_set)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                grouping_db = pickle.load(f)
                self.grouping = grouping_db['grouping']
                self.db = grouping_db['db']
        else:
            self.db = self._get_db()
            self.grouping = self._get_group(self.db)
            grouping_db_to_dump = {'grouping': self.grouping, 'db': self.db}
            with open(self.db_file, 'wb') as f:
                pickle.dump(grouping_db_to_dump, f)

        if self.is_train:
            self.grouping = self.grouping[::5]
        else:
            self.grouping = self.grouping[::64]

        self.group_size = len(self.grouping)
        self.selected_cam = [0, 1, 2, 3]
        self.num_views = len(self.selected_cam)

    def _get_db(self):
        anno_file = osp.join(self.dataset_root, 'annot',
                             'h36m_{}.pkl'.format(self.image_set))
        with open(anno_file, 'rb') as f:
            dataset = pickle.load(f)
        # process all item to Panoptic Format!!!
        nitems = len(dataset)
        for i in range(nitems):
            all_poses_3d = []
            all_poses_vis_3d = []
            all_poses = []
            all_poses_vis = []

            camera = self._get_cam(dataset[i]['camera'])
            # NOTE: note that h36m joints_3d is in camera frame
            joints_3d = \
                camera_to_world_frame(
                    dataset[i]['joints_3d'],
                    camera['R'],
                    camera['T'])[H36M_TO_PANOPTIC]
            img_path = osp.join(self.dataset_root,
                                'images', dataset[i]['image'])

            if True:  # use projected 2d pose
                joints_2d = project_pose(joints_3d, camera)
            else:  # use original 2d pose
                joints_2d = dataset[i]['joints_2d'][H36M_TO_PANOPTIC]
            # """ This for 2D joints visualization
            # import cv2
            # img = cv2.imread(img_path)
            # for joint in joints_2d:
            # cv2.circle(img, (int(joint[0]),
            # int(joint[1])), 3, [0, 0, 255], -1)
            # cv2.imwrite('test.jpg', img)
            # import pdb; pdb.set_trace()
            # """

            joints_3d_vis = dataset[i]['joints_vis'][H36M_TO_PANOPTIC]
            all_poses_3d.append(joints_3d)
            all_poses_vis_3d.append(joints_3d_vis)

            joints_2d_vis = joints_3d_vis[:, :2]
            all_poses.append(joints_2d)
            all_poses_vis.append(joints_2d_vis)

            dataset[i]['joints_2d_ori'] = dataset[i]['joints_2d']
            dataset[i]['joints_3d'] = all_poses_3d
            dataset[i]['joints_3d_vis'] = all_poses_vis_3d
            dataset[i]['joints_2d'] = all_poses
            dataset[i]['joints_2d_vis'] = all_poses_vis

            our_cam = {}
            our_cam['R'] = camera['R']
            our_cam['T'] = camera['T']
            our_cam['standard_T'] = -np.dot(camera['R'], camera['T'])
            our_cam['K'] = camera['K']
            our_cam['fx'] = camera['fx'][0]
            our_cam['fy'] = camera['fy'][0]
            our_cam['cx'] = camera['cx'][0]
            our_cam['cy'] = camera['cy'][0]
            our_cam['k'] = camera['k'].reshape(3, 1)
            our_cam['p'] = camera['p'].reshape(2, 1)

            dataset[i]['camera_ori'] = dataset[i]['camera']
            dataset[i]['camera'] = our_cam

            dataset[i]['image_name'] = dataset[i]['image']
            dataset[i]['image'] = img_path

        return dataset

    def _get_cam(self, camera):
        fx, fy = camera['fx'], camera['fy']
        cx, cy = camera['cx'], camera['cy']
        K = np.eye(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        camera['K'] = K
        return camera

    def _get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            subject = db[i]['subject']
            action = db[i]['action']
            subaction = db[i]['subaction']
            # filter out damaged actions
            if subject == 9 and \
                    ((action == 5 and subaction == 2)
                     or (action == 10 and subaction == 2)
                     or (action == 13 and subaction == 1)):
                continue
            keystr = self._get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            # remove all samples without full views
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        return filtered_grouping

    def _get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap \
            = [], [], [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, t3, m, ih = super().__getitem__(item)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)

        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.group_size

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.group_size
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i, items in enumerate(self.grouping):
            db_rec = copy.deepcopy(self.db[items[0]])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(
                        np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return \
            aps, \
            recs, \
            self._eval_list_to_mpjpe(eval_list), \
            self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt


if __name__ == "__main__":
    import argparse
    from core.config import config
    from core.config import update_config
    import torchvision.transforms as transforms

    def parse_args():
        parser = argparse.ArgumentParser(description='Train keypoints network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            required=True, type=str)

        args, rest = parser.parse_known_args()
        update_config(args.cfg)

        return args

    args = parse_args()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = H36M(
        config, config.DATASET.TRAIN_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    dataset.__getitem__(1)
