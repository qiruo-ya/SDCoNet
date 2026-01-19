# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder, SADetrTransformerEncoder_BG,SADetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .sa_detr_bg import SA_DeformableDETR_BG
from ..losses import SalienceCriterion
from .dino import DINO
from ..generator import SRNet



@MODELS.register_module()
class SA_DINO_SR(SA_DeformableDETR_BG):
    def __init__(self, *args, dn_cfg: OptConfigType = None, sr_cfg=None, sr_loss_weight=5.0, salience_criterion=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sr_loss_weight = sr_loss_weight

        
        if sr_cfg is not None:
            self.sr_module = MODELS.build(sr_cfg)
        else:
            self.sr_module = SRNet(in_channels=self.embed_dims, out_channels=self.embed_dims, scale_factor=2)

        self.sr_loss_fn = nn.L1Loss()

        
        self.salience_criterion = MODELS.build(salience_criterion) if salience_criterion else None
        assert self.as_two_stage, 'as_two_stage 必须为 True 才能使用 DINO'
        assert self.with_box_refine, 'with_box_refine 必须为 True 才能使用 DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and 'num_queries' not in dn_cfg and 'hidden_dim' not in dn_cfg, \
                '请不要在 dn_cfg 配置中设置 `num_classes`, `embed_dims`, `num_matching_queries`，这三个参数已在 `detector.__init__()` 中设置。'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """初始化除 backbone, neck 和 bbox_head 之外的层"""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = SADetrTransformerEncoder_BG(**self.encoder)
        # self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims 应该是 num_feats 的两倍, 目前 embed_dims 为 {self.embed_dims}，num_feats 为 {num_feats}'

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """初始化 Transformer 和其他组件的权重"""
        super(SA_DeformableDETR_BG, self).init_weights()
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
        self.alpha.data.uniform_(-0.3, 0.3)
      

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """提取特征"""
        res_x, x, hw_shapes = self.backbone(batch_inputs)
        if self.with_neck:
            x, res_4 = self.neck(x)
        return res_x, res_4, hw_shapes, x

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples: OptSampleList) -> Dict:
        """计算损失"""
        low_res_inputs = F.interpolate(batch_inputs, scale_factor=0.5, mode='bicubic', align_corners=False)
        res_x, res_4, hw_shapes, img_feats = self.extract_feat(low_res_inputs)

        # res_x, res_4, hw_shapes, img_feats = self.extract_feat(batch_inputs)

        if self.training:
            sr_feats = self.sr_module(res_x, res_4, hw_shapes)
            _, _, H_gt, W_gt = batch_inputs.shape
            sr_feats = F.interpolate(sr_feats, size=(H_gt, W_gt), mode='bilinear', align_corners=False)
            sr_loss = self.sr_loss_fn(sr_feats, batch_inputs) * self.sr_loss_weight
        else:
            sr_feats = img_feats

        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        bbox_inputs = {k: v for k, v in head_inputs_dict.items() if k != 'memory'}

       
        losses = self.bbox_head.loss(**bbox_inputs, batch_data_samples=batch_data_samples)

        if self.training:
            losses['sr_loss'] = sr_loss

        
        foreground_score = head_inputs_dict.get('memory', None)
        mlvl_feats = self.last_feat

        if foreground_score is not None and mlvl_feats is not None:
            img_h, img_w = batch_data_samples[0].img_shape
            feature_strides = [
                (img_h / feat.shape[-2], img_w / feat.shape[-1])
                for feat in mlvl_feats
            ]
            image_sizes = [s.img_shape for s in batch_data_samples]

            
            targets = []
            for sample in batch_data_samples:
                gt_instances = sample.gt_instances
                boxes = gt_instances.bboxes  # [num_gt, 4] in xyxy
                img_h, img_w = sample.img_shape

                
                cxcywh = torch.stack([
                    (boxes[:, 0] + boxes[:, 2]) / 2 / img_w,  # cx
                    (boxes[:, 1] + boxes[:, 3]) / 2 / img_h,  # cy
                    (boxes[:, 2] - boxes[:, 0]) / img_w,  # w
                    (boxes[:, 3] - boxes[:, 1]) / img_h  # h
                ], dim=-1)

                targets.append({'boxes': cxcywh})

           
            salience_loss_dict = self.salience_criterion(
                foreground_mask=foreground_score,
                targets=targets,
                feature_strides=feature_strides,
                image_sizes=image_sizes
            )

           
            losses.update(salience_loss_dict)

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: OptSampleList, rescale: bool = True) -> OptSampleList:
       
        low_res_inputs = F.interpolate(batch_inputs, scale_factor=0.5, mode='bicubic', align_corners=False)
        res_x, res_4, hw_shapes, img_feats = self.extract_feat(low_res_inputs)

        # res_x, res_4, hw_shapes, img_feats = self.extract_feat(batch_inputs)

        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        bbox_inputs = {k: v for k, v in head_inputs_dict.items() if k != 'memory'}
        results_list = self.bbox_head.predict(**bbox_inputs, rescale=rescale, batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
       
        encoder_inputs_dict, decoder_inputs_dict, mlvl_masks = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_inputs_dict,salience_score = self.forward_sa(**encoder_inputs_dict, mlvl_masks=mlvl_masks)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict['memory'] = salience_score
        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
       
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        # print(f"enc_outputs_class shape: {enc_outputs_class.shape}")
        # print(f"self.num_queries: {self.num_queries}")

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
       
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process for tensor mode.
        This method is typically used for FLOPs calculation.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (OptSampleList, optional): The batch data samples.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from bbox_head forward.
        """
        low_res_inputs = F.interpolate(
            batch_inputs, scale_factor=0.5, mode='bicubic', align_corners=False)

       
        res_x, res_4, hw_shapes, img_feats = self.extract_feat(low_res_inputs)

        
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)

       
        bbox_inputs = {k: v for k, v in head_inputs_dict.items() if k != 'memory'}
        results = self.bbox_head.forward(**bbox_inputs)

        return results


