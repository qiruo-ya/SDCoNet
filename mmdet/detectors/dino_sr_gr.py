# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .dino import DINO
from ..generator import SRNet



@MODELS.register_module()
class DINO_SR_GR(DINO):
    def __init__(self, *args, sr_cfg=None,sr_loss_weight=5.0,freeze_sr=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.sr_loss_weight = sr_loss_weight
        self.freeze_sr = freeze_sr
        # Initialize the super-resolution module
        if sr_cfg is not None:
            self.sr_module = MODELS.build(sr_cfg)
        else:
            self.sr_module = SRNet(in_channels=self.embed_dims, out_channels=self.embed_dims, scale_factor=2)

        if self.freeze_sr:
            for p in self.sr_module.parameters():
                p.requires_grad = False

        # Define L1 loss function for super resolution task
        self.sr_loss_fn = nn.L1Loss()

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        res_x, x,hw_shapes = self.backbone(batch_inputs)
        if self.with_neck:
            x, res_4 = self.neck(x)
        return res_x, res_4,hw_shapes, x

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples: OptSampleList) -> Dict:

        low_res_inputs = F.interpolate(batch_inputs, scale_factor=0.5, mode='bicubic', align_corners=False)

        res_x, res_4,hw_shapes,img_feats = self.extract_feat(low_res_inputs)

        # === Super-resolution branch ===
        sr_loss = None
        if self.training and not self.freeze_sr:
            sr_feats = self.sr_module(res_x, res_4, hw_shapes)
            _, _, H_gt, W_gt = batch_inputs.shape
            sr_feats = F.interpolate(sr_feats, size=(H_gt, W_gt), mode='bilinear', align_corners=False)
            sr_loss = self.sr_loss_fn(sr_feats, batch_inputs)* self.sr_loss_weight

        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        bbox_inputs = {k: v for k, v in head_inputs_dict.items() if k != 'memory'}
        # Call bbox_head loss
        losses = self.bbox_head.loss(**bbox_inputs, batch_data_samples=batch_data_samples)

        if sr_loss is not None:
            losses['sr_loss'] = sr_loss

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: OptSampleList,
                rescale: bool = True) -> OptSampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        low_res_inputs = F.interpolate(batch_inputs, scale_factor=0.5, mode='bicubic', align_corners=False)
        res_x, res_4,hw_shapes,img_feats  = self.extract_feat(low_res_inputs)

        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        bbox_inputs = {k: v for k, v in head_inputs_dict.items() if k != 'memory'}
        results_list = self.bbox_head.predict(
            **bbox_inputs, rescale=rescale, batch_data_samples=batch_data_samples)
        # results_list = self.bbox_head.predict(
        #     **head_inputs_dict,
        #     rescale=rescale,
        #     batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

