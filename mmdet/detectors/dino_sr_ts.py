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
class DINO_SR_TS(DINO):
    def __init__(self,
                 backbone,
                 neck=None,
                 *args,
                 sr_cfg=None,
                 sr_loss_weight=5.0,
                 sr_backbone=None,
                 sr_neck=None,
                 **kwargs):

        import copy
        # 保存配置用于创建独立的SR分支
        saved_backbone_cfg = copy.deepcopy(backbone)
        saved_neck_cfg = copy.deepcopy(neck) if neck is not None else None

        # 初始化父类（检测分支）
        super().__init__(backbone=backbone, neck=neck, *args, **kwargs)

        self.sr_loss_weight = sr_loss_weight

        # 创建独立的SR Backbone
        if sr_backbone is not None:
            self.sr_backbone = MODELS.build(sr_backbone)
        else:
            self.sr_backbone = MODELS.build(saved_backbone_cfg)

        # 添加这一行！设置是否有SR neck的标志
        self.with_sr_neck = sr_neck is not None or saved_neck_cfg is not None

        # 创建独立的SR Neck
        if sr_neck is not None:
            self.sr_neck = MODELS.build(sr_neck)
        elif saved_neck_cfg is not None:
            self.sr_neck = MODELS.build(saved_neck_cfg)
        else:
            self.sr_neck = None

        # SR Decoder
        if sr_cfg is not None:
            self.sr_module = MODELS.build(sr_cfg)
        else:
            self.sr_module = SRNet(
                in_channels=self.embed_dims,
                out_channels=3,  # RGB image
                scale_factor=2
            )

        self.sr_loss_fn = nn.L1Loss()

        # Note: Detection branch uses self.backbone and self.neck (inherited from DINO)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

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

        # Initialize detection components
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

        # Initialize SR backbone and neck
        if hasattr(self.sr_backbone, 'init_weights'):
            self.sr_backbone.init_weights()
        if self.sr_neck is not None and hasattr(self.sr_neck, 'init_weights'):
            self.sr_neck.init_weights()

    def extract_feat(self, batch_inputs: Tensor, branch: str = 'det') -> Tuple[Tensor]:
        """Unified feature extraction function for both SR and Det branches.

        Same code logic, different parameters based on branch selection.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, 3, H, W).
            branch (str): 'sr' for SR branch, 'det' for detection branch.

        Returns:
            tuple[Tensor]: Feature maps from backbone and neck.
        """
        # Select backbone and neck based on branch
        if branch == 'sr':
            backbone = self.sr_backbone
            neck = self.sr_neck if self.with_sr_neck else None
        elif branch == 'det':
            backbone = self.backbone
            neck = self.neck if self.with_neck else None
        else:
            raise ValueError(f"Invalid branch: {branch}. Must be 'sr' or 'det'.")

        # Unified extraction logic
        res_x, x, hw_shapes = backbone(batch_inputs)

        if neck is not None:
            x, res_4 = neck(x)
        else:
            res_4 = None

        return res_x, res_4, hw_shapes, x

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples: OptSampleList) -> Dict:
        """Two-stage loss computation with independent encoders.

        Stage 1: Low-res → SR Encoder → SR Decoder → High-res image
        Stage 2: High-res image → Det Encoder → Transformer → Predictions
        """

        # ============ Stage 1: Super-Resolution ============
        # Downsample to create low-resolution input
        low_res_inputs = F.interpolate(
            batch_inputs,
            scale_factor=0.5,
            mode='bicubic',
            align_corners=False
        )

        # SR Encoder: extract features from low-res images (using SR branch)
        res_x, res_4, hw_shapes, sr_feats = self.extract_feat(low_res_inputs, branch='sr')

        # SR Decoder: generate high-resolution images
        sr_images = self.sr_module(res_x, res_4, hw_shapes)

        # Ensure SR output matches ground truth size
        _, _, H_gt, W_gt = batch_inputs.shape
        if sr_images.shape[2:] != (H_gt, W_gt):
            sr_images = F.interpolate(
                sr_images,
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            )

        # Compute SR loss
        sr_loss = self.sr_loss_fn(sr_images, batch_inputs) * self.sr_loss_weight

        # ============ Stage 2: Object Detection ============
        if self.training:
            # Option 1: Detach to train SR and Det independently
            det_inputs = sr_images.detach()
            # Option 2: Keep gradient flow for joint training
            # det_inputs = sr_images
        else:
            det_inputs = sr_images

        # Det Encoder: extract features from SR images (using detection branch)
        _, _, _, det_feats = self.extract_feat(det_inputs, branch='det')

        # Forward through detection transformer
        head_inputs_dict = self.forward_transformer(det_feats, batch_data_samples)
        bbox_inputs = {k: v for k, v in head_inputs_dict.items() if k != 'memory'}

        # Compute detection loss
        losses = self.bbox_head.loss(
            **bbox_inputs,
            batch_data_samples=batch_data_samples
        )

        # Add SR loss to total losses
        if self.training:
            losses['sr_loss'] = sr_loss

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: OptSampleList,
                rescale: bool = True) -> OptSampleList:
        """Two-stage prediction with independent encoders.

        Stage 1: Low-res → SR Encoder → SR Decoder → High-res image
        Stage 2: High-res image → Det Encoder → Transformer → Predictions
        """

        # ============ Stage 1: Super-Resolution ============
        # Downsample to create low-resolution input
        low_res_inputs = F.interpolate(
            batch_inputs,
            scale_factor=0.5,
            mode='bicubic',
            align_corners=False
        )

        # SR Encoder: extract features from low-res images (using SR branch)
        res_x, res_4, hw_shapes, sr_feats = self.extract_feat(low_res_inputs, branch='sr')

        # SR Decoder: generate high-resolution images
        sr_images = self.sr_module(res_x, res_4, hw_shapes)

        # Ensure SR output matches input size
        _, _, H_gt, W_gt = batch_inputs.shape
        if sr_images.shape[2:] != (H_gt, W_gt):
            sr_images = F.interpolate(
                sr_images,
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            )

        # ============ Stage 2: Object Detection ============
        # Det Encoder: extract features from SR images (using detection branch)
        _, _, _, det_feats = self.extract_feat(sr_images, branch='det')

        # Forward through detection transformer
        head_inputs_dict = self.forward_transformer(
            det_feats,
            batch_data_samples
        )
        bbox_inputs = {k: v for k, v in head_inputs_dict.items() if k != 'memory'}

        # Get predictions
        results_list = self.bbox_head.predict(
            **bbox_inputs,
            rescale=rescale,
            batch_data_samples=batch_data_samples
        )

        # Add predictions to data samples
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples,
            results_list
        )

        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None) -> Tuple:
        """Forward function for inference and FLOPs calculation.

        Two-stage forward pass:
        Stage 1: Low-res → SR Encoder → SR Decoder → High-res image
        Stage 2: High-res image → Det Encoder → Transformer → Predictions

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional):
                The Data Samples. Defaults to None.

        Returns:
            tuple: A tuple of features from bbox_head forward.
        """

        # ============ Stage 1: Super-Resolution ============
        # Downsample to create low-resolution input
        low_res_inputs = F.interpolate(
            batch_inputs,
            scale_factor=0.5,
            mode='bicubic',
            align_corners=False
        )

        # SR Encoder: extract features from low-res images
        res_x, res_4, hw_shapes, sr_feats = self.extract_feat(
            low_res_inputs,
            branch='sr'
        )

        # SR Decoder: generate high-resolution images
        sr_images = self.sr_module(res_x, res_4, hw_shapes)

        # Ensure SR output matches input size
        _, _, H_gt, W_gt = batch_inputs.shape
        if sr_images.shape[2:] != (H_gt, W_gt):
            sr_images = F.interpolate(
                sr_images,
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            )

        # ============ Stage 2: Object Detection ============
        # Det Encoder: extract features from SR images
        _, _, _, det_feats = self.extract_feat(sr_images, branch='det')

        # 关键修复：确保 det_feats 是正确的格式
        # 如果 det_feats 是嵌套列表，需要展平
        if isinstance(det_feats, list) and len(det_feats) > 0:
            if isinstance(det_feats[0], list):
                # 如果是嵌套列表，提取第一层
                det_feats = det_feats[0]

            # 确保所有元素都是 Tensor
            det_feats = tuple([feat for feat in det_feats if isinstance(feat, Tensor)])

        # Forward through detection transformer
        head_inputs_dict = self.forward_transformer(
            det_feats,
            batch_data_samples
        )

        # Prepare inputs for bbox_head
        bbox_inputs = {
            k: v for k, v in head_inputs_dict.items()
            if k != 'memory'
        }

        # Forward through detection head
        results = self.bbox_head.forward(**bbox_inputs)

        return results