# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

from mmdet.models.losses import sigmoid_focal_loss1




from mmdet.registry import MODELS
from .accuracy import accuracy
from .utils import weight_reduce_loss


@MODELS.register_module()
class SalienceCriterion(nn.Module):
    def __init__(
        self,
        limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
        noise_scale: float = 0.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        loss_weight=2.0
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, foreground_mask, targets, feature_strides, image_sizes):
        gt_boxes_list = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            boxes = bbox_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(boxes * scale_factor)

        mask_targets = []
        for level_idx, (mask, feature_stride) in enumerate(zip(foreground_mask, feature_strides)):
            feature_shape = mask.shape[-2:]
            coord_x, coord_y = self.get_pixel_coordinate(feature_shape, feature_stride, device=mask.device)
            masks_per_level = []
            for gt_boxes in gt_boxes_list:
                mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, level_idx)
                masks_per_level.append(mask)

            masks_per_level = torch.stack(masks_per_level)
            mask_targets.append(masks_per_level)
        mask_targets = torch.cat(mask_targets, dim=1)
        foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
        foreground_mask = foreground_mask.squeeze(1)
        num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale).clamp_(min=1)
        salience_loss = (
            sigmoid_focal_loss1(
                foreground_mask,
                mask_targets,
                num_pos,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * foreground_mask.shape[1]
        )
        # return {"loss_salience": salience_loss}
        return {"loss_salience": salience_loss * self.loss_weight}

    def get_pixel_coordinate(self, feature_shape, stride, device):
        height, width = feature_shape
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
            indexing="ij",
        )
        coord_y = coord_y.reshape(-1)
        coord_x = coord_x.reshape(-1)
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
        # gt_label: (m,) gt_boxes: (m, 4)
        # coord_x: (h*w, )
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )  # [h*w, m, 4]

        # the foreground queries must satisfy two requirements:
        # 1. the quereis located in bounding boxes
        # 2. the distance from queries to the box center match the feature map stride
        min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m]
        max_border_distances = torch.max(border_distances, dim=-1)[0]
        mask_in_gt_boxes = min_border_distances > 0
        min_limit, max_limit = self.limit_range[level_idx]
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
        mask_pos = mask_in_gt_boxes & mask_in_level

        # scale-independent salience confidence
        row_factor = left_border_distance + right_border_distance
        col_factor = top_border_distance + bottom_border_distance
        delta_x = (left_border_distance - right_border_distance) / row_factor
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

        confidence_per_box = 1 - confidence
        confidence_per_box[~mask_in_gt_boxes] = 0

        # process positive coordinates
        if confidence_per_box.numel() != 0:
            mask = confidence_per_box.max(-1)[0]
        else:
            mask = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

        # process negative coordinates
        mask_pos = mask_pos.long().sum(dim=-1) >= 1
        mask[~mask_pos] = 0

        # add noise to add randomness
        mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
        return mask
