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

import time
import logging
import os
import torch

from utils.vis import save_debug_3d_images

from models.util.misc import get_total_grad_norm, is_main_process


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch,
             output_dir, device=torch.device('cuda'), num_views=5):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_ce = AverageMeter()
    class_error = AverageMeter()
    loss_pose_perjoint = AverageMeter()
    loss_pose_perbone = AverageMeter()
    loss_pose_perprojection = AverageMeter()
    cardinality_error = AverageMeter()

    model.train()

    if model.module.backbone is not None:
        # Comment out this line if you want to train 2D backbone jointly
        model.module.backbone.eval()

    threshold = model.module.pred_conf_threshold

    end = time.time()
    for i, (inputs, meta) in enumerate(loader):
        assert len(inputs) == num_views
        inputs = [i.to(device) for i in inputs]
        meta = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()} for t in meta]
        data_time.update(time_synchronized() - end)
        end = time_synchronized()

        out, loss_dict = model(views=inputs, meta=meta)

        gt_3d = meta[0]['joints_3d'].float()
        num_joints = gt_3d.shape[2]
        bs, num_queries = out["pred_logits"].shape[:2]

        src_poses = out['pred_poses']['outputs_coord'].\
            view(bs, num_queries, num_joints, 3)
        src_poses = model.module.norm2absolute(src_poses)
        score = out['pred_logits'][:, :, 1:2].sigmoid()
        score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
        temp = (score > threshold).float() - 1

        pred = torch.cat([src_poses, temp, score], dim=-1)

        weight_dict = model.module.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        loss_ce.update(loss_dict['loss_ce'].sum().item())
        class_error.update(loss_dict['class_error'].sum().item())

        loss_pose_perjoint.update(loss_dict['loss_pose_perjoint'].sum().item())
        if 'loss_pose_perbone' in loss_dict:
            loss_pose_perbone.update(
                loss_dict['loss_pose_perbone'].sum().item())
        if 'loss_pose_perprojection' in loss_dict:
            loss_pose_perprojection.update(
                loss_dict['loss_pose_perprojection'].sum().item())

        cardinality_error.update(
            loss_dict['cardinality_error'].sum().item())

        if losses > 0:
            optimizer.zero_grad()
            losses.backward()
            if config.TRAIN.clip_max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.clip_max_norm)
            else:
                grad_total_norm = get_total_grad_norm(
                    model.parameters(), config.TRAIN.clip_max_norm)

            optimizer.step()

        batch_time.update(time_synchronized() - end)
        end = time_synchronized()

        if i % config.PRINT_FREQ == 0 and is_main_process():
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = \
                'Epoch: [{0}][{1}/{2}]\t' \
                'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.3f}s ' '({data_time.avg:.3f}s)\t' \
                'loss_ce: {loss_ce.val:.7f} ' '({loss_ce.avg:.7f})\t' \
                'class_error: {class_error.val:.7f} ' \
                '({class_error.avg:.7f})\t' \
                'loss_pose_perjoint: {loss_pose_perjoint.val:.6f} ' \
                '({loss_pose_perjoint.avg:.6f})\t' \
                'loss_pose_perbone: {loss_pose_perbone.val:.6f} ' \
                '({loss_pose_perbone.avg:.6f})\t' \
                'loss_pose_perprojection: {loss_pose_perprojection.val:.6f} ' \
                '({loss_pose_perprojection.avg:.6f})\t' \
                'cardinality_error: {cardinality_error.val:.6f} ' \
                '({cardinality_error.avg:.6f})\t' \
                'Memory {memory:.1f}\t'\
                'gradnorm {gradnorm:.2f}'.format(
                  epoch, i, len(loader),
                  batch_time=batch_time,
                  speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                  data_time=data_time,
                  loss_ce=loss_ce,
                  class_error=class_error,
                  loss_pose_perjoint=loss_pose_perjoint,
                  loss_pose_perbone=loss_pose_perbone,
                  loss_pose_perprojection=loss_pose_perprojection,
                  cardinality_error=cardinality_error,
                  memory=gpu_memory_usage,
                  gradnorm=grad_total_norm)
            logger.info(msg)

            prefix2 = '{}_{:08}'.format(
                os.path.join(output_dir, 'train'), i)
            save_debug_3d_images(config, meta[0], pred, prefix2)


def validate_3d(config, model, loader, output_dir, threshold, num_views=5):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()

    preds = []
    meta_image_files = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, meta) in enumerate(loader):
            data_time.update(time.time() - end)
            assert len(inputs) == num_views

            output = model(views=inputs, meta=meta)

            meta_image_files.append(meta[0]['image'])
            gt_3d = meta[0]['joints_3d'].float()
            num_joints = gt_3d.shape[2]
            bs, num_queries = output["pred_logits"].shape[:2]

            src_poses = output['pred_poses']['outputs_coord'].\
                view(bs, num_queries, num_joints, 3)
            src_poses = model.module.norm2absolute(src_poses)
            score = output['pred_logits'][:, :, 1:2].sigmoid()
            score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
            temp = (score > threshold).float() - 1

            pred = torch.cat([src_poses, temp, score], dim=-1)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])

            batch_time.update(time.time() - end)
            end = time.time()
            if (i % config.PRINT_FREQ == 0 or i == len(loader) - 1) \
                    and is_main_process():
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                logger.info(msg)

                prefix2 = '{}_{:08}'.format(
                    os.path.join(output_dir, 'validation'), i)
                save_debug_3d_images(config, meta[0], pred, prefix2)
    return preds, meta_image_files


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
