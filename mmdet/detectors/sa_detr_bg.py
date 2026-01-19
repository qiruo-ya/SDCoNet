# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (DeformableDetrTransformerDecoder,SADetrTransformerEncoder,
                      SADetrTransformerEncoder_BG, SinePositionalEncoding)
from .base_detr import DetectionTransformer

class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out
@MODELS.register_module()
class SA_DeformableDETR_BG(DetectionTransformer):
    r"""Implementation of `Deformable DETR: Deformable Transformers for
    End-to-End Object Detection <https://arxiv.org/abs/2010.04159>`_

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    Args:
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        with_box_refine (bool, optional): Whether to refine the references
            in the decoder. Defaults to `False`.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
        num_feature_levels (int, optional): Number of feature levels.
            Defaults to 4.
    """

    def __init__(self,
                 *args,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 num_feature_levels: int = 4,
                 # level_filter_ratio=(0.4, 0.8, 1.0, 1.0),
                 # layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
                 level_filter_ratio=(1.0, 1.0, 1.0, 1.0),
                 layer_filter_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 # level_filter_ratio=(0.6, 0.8, 1.0, 1.0),
                 # layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.6, 0.4),
                 **kwargs) -> None:
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        if bbox_head is not None:
            assert 'share_pred_layer' not in bbox_head and \
                   'num_pred_layer' not in bbox_head and \
                   'as_two_stage' not in bbox_head, \
                'The two keyword args `share_pred_layer`, `num_pred_layer`, ' \
                'and `as_two_stage are set in `detector.__init__()`, users ' \
                'should not set them in `bbox_head` config.'
            # The last prediction layer is used to generate proposal
            # from encode feature map when `as_two_stage` is `True`.
            # And all the prediction layers should share parameters
            # when `with_box_refine` is `True`.
            bbox_head['share_pred_layer'] = not with_box_refine
            bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
                if self.as_two_stage else decoder['num_layers']
            bbox_head['as_two_stage'] = as_two_stage

            self.num_classes = bbox_head['num_classes']

        super().__init__(*args, decoder=decoder, bbox_head=bbox_head, **kwargs)
        self.embed_dim = self.encoder.embed_dims
        # self.encoder_class_head = nn.Linear(self.embed_dim, self.num_classes)
        # self.encoder.enhance_mcsp = self.encoder_class_head
        self.encoder.enhance_mcsp = nn.Linear(self.embed_dim, self.num_classes)
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = SADetrTransformerEncoder_BG(**self.encoder)
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)
        self.alpha.data.uniform_(-0.3, 0.3)


    def flatten_multi_level(self, multi_level_elements):
        multi_level_elements = torch.cat([e.flatten(-2) for e in multi_level_elements], -1)  # (b, [c], s)
        if multi_level_elements.ndim == 3:
            multi_level_elements.transpose_(1, 2)
        return multi_level_elements

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        # batch_size = mlvl_feats[0].size(0)
        if isinstance(mlvl_feats[0], list):
            batch_size = len(mlvl_feats[0])
        else:
            batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        # if torch.onnx.is_in_onnx_export() or same_shape_flag:
        #     mlvl_masks = []
        #     mlvl_pos_embeds = []
        #     for feat in mlvl_feats:
        #         mlvl_masks.append(None)
        #         mlvl_pos_embeds.append(
        #             self.positional_encoding(None, input=feat))

        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            # for idx, feat in enumerate(mlvl_feats):
            #     # 输出每个特征图的尺寸
            #     print(f"shape: {mlvl_feats.shape}")
            #     print(f"Feature map {idx} shape: {feat.shape}")



            for feat in mlvl_feats:
                # 构造全为 False 的 mask，表示所有位置都有效
                mask = torch.zeros(
                    (batch_size, feat.shape[-2], feat.shape[-1]),
                    dtype=torch.bool,
                    device=feat.device
                )
                mlvl_masks.append(mask)
                mlvl_pos_embeds.append(
                    self.positional_encoding(mask))


        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))
                mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        self.last_feat = mlvl_feats
        return encoder_inputs_dict, decoder_inputs_dict, mlvl_masks

    def forward_sa(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor, mlvl_masks) -> Dict:
        backbone_output_memory = self.gen_encoder_output_proposals(
            feat + feat_pos, feat_mask, spatial_shapes
        )[0]

        # calculate filtered tokens numbers for each feature map
        reverse_multi_level_masks = [~m for m in mlvl_masks]
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)

        #调试
        # total_original_tokens = valid_token_nums.sum(-1)
        #
        # print(f"\n{'=' * 70}")
        # print(f"[DEBUG] Token Filtering Analysis")
        # print(f"{'=' * 70}")
        # print(f"[1] Original valid tokens per level: {valid_token_nums[0].cpu().numpy()}")
        # print(f"[1] Total original tokens: {total_original_tokens[0].item():.0f}")

        # ============= 插入计算代码 - 开始 =============
        # 计算原始有效token总数
        # reverse_multi_level_masks = [~m for m in mlvl_masks]
        # valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        # total_original_tokens = valid_token_nums.sum(-1)  # [batch_size]
        #
        # print(f"\n{'=' * 60}")
        # print(f"Effective Query Ratio Calculation")
        # print(f"{'=' * 60}")
        # print(f"Original valid tokens per level: {valid_token_nums[0].cpu().numpy()}")
        # print(f"Total original tokens: {total_original_tokens[0].item():.0f}")
        # ============= 插入计算代码 - 结束 =============

        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        level_token_nums = focus_token_nums.max(0)[0]
        # focus_token_nums = focus_token_nums.sum(-1)
        # focus_token_nums_sum = focus_token_nums.sum(-1)
        focus_token_nums = focus_token_nums.sum(-1)

        #调试
        # print(f"\n[2] Level filter ratio: {self.level_filter_ratio.cpu().numpy()}")
        # print(f"[2] Tokens after level filtering per level: {focus_token_nums[0].cpu().numpy()}")
        # print(f"[2] Total after level filtering: {focus_token_nums_sum[0].item():.0f}")
        # print(
        #     f"[2] Level filtering retention rate: {focus_token_nums_sum[0].item() / total_original_tokens[0].item() * 100:.2f}%")



        # ============= 插入计算代码 - 开始 =============
        # Level过滤后的token数
        # print(f"After level filtering: {focus_token_nums[0].item():.0f}")
        # level_ratio = focus_token_nums[0].item() / total_original_tokens[0].item()
        # print(f"Level filtering ratio: {level_ratio:.4f} ({level_ratio * 100:.2f}%)")

        # from high level to low level
        batch_size = feat.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            mask = feat_mask[:, start_index:end_index]
            # update the memory using the higher-level score_prediction
            if level_idx != spatial_shapes.shape[0] - 1:
                upsample_score = torch.nn.functional.interpolate(
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                upsample_score = upsample_score.view(batch_size, -1, spatial_shapes[level_idx].prod())
                upsample_score = upsample_score.transpose(1, 2)
                level_memory = level_memory + level_memory * upsample_score * self.alpha[level_idx]
            # predict the foreground score of the current layer
            score = self.enc_mask_predictor(level_memory)
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())
            score = score.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx])

            # get the topk salience index of the current feature map level
            level_score, level_inds = valid_score.topk(level_token_nums[level_idx], dim=1)
            level_inds = level_inds + level_start_index[level_idx]
            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)

        selected_score = torch.cat(selected_score[::-1], 1)
        index = torch.sort(selected_score, dim=1, descending=True)[1]
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)  # selected_inds 中的 token 索引会按照显著性得分的降序排列

        # create layer-wise filtering
        num_inds = selected_inds.shape[1]
        # change dtype to avoid shape inference error during exporting ONNX
        cast_dtype = torch.int64
        layer_filter_ratio = (num_inds * self.layer_filter_ratio).to(cast_dtype)
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]


        #调试
        # print(f"\n[3] Layer filter ratio: {self.layer_filter_ratio.cpu().numpy()}")
        # print(f"[3] Selected indices shape: {num_inds}")
        # print(f"[3] Tokens for each encoder layer: {layer_filter_ratio.cpu().numpy()}")
        # print(f"[3] Average tokens across layers: {layer_filter_ratio.float().mean():.0f}")
        # print(
        #     f"[3] Final retention rate (avg): {layer_filter_ratio.float().mean() / total_original_tokens[0].item() * 100:.2f}%")
        # print(f"{'=' * 70}\n")

        # ============= 插入计算代码 - 开始 =============
        # Layer过滤后的token数（取平均或最后一层）
        # avg_layer_tokens = sum(layer_filter_ratio.float()) / len(layer_filter_ratio)
        #
        # print(f"\nLayer-wise filtering ratios: {self.layer_filter_ratio.cpu().numpy()}")
        # print(f"Tokens after each layer: {layer_filter_ratio.cpu().numpy()}")
        # print(f"Average tokens across layers: {avg_layer_tokens:.0f}")
        #
        # # 计算最终的Effective Query Ratio
        # # 方法1: 使用平均值
        # avg_effective_ratio = avg_layer_tokens / total_original_tokens[0].item()
        # print(f"\nEffective Query Ratio (average): {avg_effective_ratio:.4f} ({avg_effective_ratio * 100:.2f}%)")
        #
        # print(f"{'=' * 60}\n")
        # ============= 插入计算代码 - 结束 =============


        salience_score = salience_score[::-1]
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        foreground_score = foreground_score.masked_fill(feat_mask, foreground_score.min())  # 未经过 level layer过滤的fae

        encoder_inputs_dict = dict(
            feat=feat,
            feat_mask=feat_mask,
            feat_pos=feat_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # salience input
            foreground_score=foreground_score,
            focus_token_nums=focus_token_nums,
            foreground_inds=selected_inds,
            multi_level_masks=mlvl_masks,
        )
        return encoder_inputs_dict,salience_score


    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,foreground_score: Tensor,
                        focus_token_nums: Tensor,foreground_inds: Tensor,
                        multi_level_masks: Tensor,) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # salience input
            foreground_score=foreground_score,
            focus_token_nums=focus_token_nums,
            foreground_inds=foreground_inds,
            multi_level_masks=multi_level_masks,
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). It will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                It will only be used when `as_two_stage` is `True`.

        Returns:
            tuple[dict, dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and `reference_points`. The reference_points of
              decoder input here are 4D boxes when `as_two_stage` is `True`,
              otherwise 2D points, although it has `points` in its name.
              The reference_points in encoder is always 2D points.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `enc_outputs_class` and
              `enc_outputs_coord`. They are both `None` when 'as_two_stage'
              is `False`. The dict is empty when `self.training` is `False`.
        """
        batch_size, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                    output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers](output_memory) + output_proposals
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            # We only use the first channel in enc_outputs_class as foreground,
            # the other (num_classes - 1) channels are actually not used.
            # Its targets are set to be 0s, which indicates the first
            # class (foreground) because we use [0, num_classes - 1] to
            # indicate class labels, background class is indicated by
            # num_classes (similar convention in RPN).
            # See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/deformable_detr_head.py#L241 # noqa
            # This follows the official implementation of Deformable DETR.
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            pos_trans_out = self.pos_trans_fc(
                self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            enc_outputs_class, enc_outputs_coord = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, reference_points: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged as
                (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)
        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references)
        return decoder_outputs_dict

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """

        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    @staticmethod
    def get_proposal_pos_embed(proposals: Tensor,
                               num_pos_feats: int = 128,
                               temperature: int = 10000) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos
