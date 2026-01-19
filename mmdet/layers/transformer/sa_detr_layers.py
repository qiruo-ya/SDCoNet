# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor, nn

from .sabase_detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
from ..positional_encoding import PositionEmbeddingLearned
from .utils import inverse_sigmoid

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class SADetrTransformerEncoder(DetrTransformerEncoder):
    """Transformer encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            SADetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                # salience input
                foreground_score=None,
                focus_token_nums=None,
                foreground_inds=None,
                multi_level_masks=None,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)

        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query

        for layer_id, layer in enumerate(self.layers):
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dims)
            query = torch.gather(output, 1, inds_for_query)
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            score_tgt = self.enhance_mcsp(query)

        # for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points,
                score_tgt=score_tgt,
                foreground_pre_layer=foreground_pre_layer,
                **kwargs)

            outputs = []
            for i in range(foreground_inds[layer_id].shape[0]):
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                query_no_pad = query[i][:focus_token_nums[i]]
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),
                        query_no_pad,
                    )
                )
            output = torch.stack(outputs)

        return output

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

class SADetrTransformerEncoder_BG(DetrTransformerEncoder):
    """Transformer encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            SADetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims
        self.background_embedding = PositionEmbeddingLearned(
            num_embeddings=200,  # 可根据最大 H/W 调整
            num_pos_feats=self.embed_dims // 2
        )

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                # salience input
                foreground_score=None,
                focus_token_nums=None,
                foreground_inds=None,
                multi_level_masks=None,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        # 调试
        # print(f"\n{'=' * 70}")
        # print(f"[ENCODER] Token Filtering Verification")
        # print(f"{'=' * 70}")
        # print(f"Input query shape: {query.shape}")  # 例如: [2, 20000, 256]
        # print(f"Focus token nums (per sample): {focus_token_nums}")
        # print(f"Number of layer-wise filter groups: {len(foreground_inds)}")
        #
        # for i, inds in enumerate(foreground_inds):
        #     reduction = (1 - inds.shape[1] / query.shape[1]) * 100
        #     print(f"  Layer {i}: using {inds.shape[1]:5d} / {query.shape[1]} tokens (Reduction: {reduction:5.2f}%)")
        # print(f"{'=' * 70}\n")

        # 结束




        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)

        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query

        for layer_id, layer in enumerate(self.layers):

            #调试
            # print(f"\n{'=' * 60}")
            # print(f"[Layer {layer_id}] Token Selection Details")
            # print(f"{'=' * 60}")
            #
            # # ============= 验证点2: Token选择前 =============
            # print(f"[Before Selection]")
            # print(f"  Output (full) shape: {output.shape}")
            # print(f"  Indices shape: {foreground_inds[layer_id].shape}")
            # print(f"  Will select {foreground_inds[layer_id].shape[1]} tokens")

            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dims)
            query = torch.gather(output, 1, inds_for_query)
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            # ============= 验证点3: Token选择后 =============
            # print(f"\n[After Selection]")
            # print(f"  Selected query shape: {query.shape}")
            # print(f"  Selected query_pos shape: {query_pos.shape}")
            # print(f"  Selected reference_points shape: {reference_points.shape}")
            # print(f"  Reduction: {(1 - query.shape[1] / output.shape[1]) * 100:.2f}%")
            #
            # # 验证选择的indices是否有效
            # print(f"\n[Index Validation]")
            # print(f"  Min index: {foreground_inds[layer_id].min().item()}")
            # print(f"  Max index: {foreground_inds[layer_id].max().item()}")
            # print(f"  Max should be < {output.shape[1]}: {foreground_inds[layer_id].max().item() < output.shape[1]}")
            #
            # # 检查是否有重复的indices（不应该有）
            # unique_inds = torch.unique(foreground_inds[layer_id][0])
            # print(f"  Unique indices count: {len(unique_inds)} / {foreground_inds[layer_id].shape[1]}")
            # if len(unique_inds) != foreground_inds[layer_id].shape[1]:
            #     print(f"  ⚠️ WARNING: Found duplicate indices!")

            score_tgt = self.enhance_mcsp(query)

            # ============= 验证点4: Layer处理前 =============
            # print(f"\n[Before Layer Processing]")
            # print(f"  Input to layer: {query.shape}")


        # for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points,
                score_tgt=score_tgt,
                foreground_pre_layer=foreground_pre_layer,
                **kwargs)

            #== == == == == == = 验证点5: Layer处理后 == == == == == == =
            # print(f"\n[After Layer Processing]")
            # print(f"  Output from layer: {query.shape}")

            outputs = []
            for i in range(foreground_inds[layer_id].shape[0]):
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                query_no_pad = query[i][:focus_token_nums[i]]

                # ============= 验证点6: Scatter操作 =============
                # if i == 0 and layer_id == 0:  # 只打印第一个样本的第一层
                #     print(f"\n[Scatter Operation - Sample 0]")
                #     print(f"  Tokens to scatter back: {focus_token_nums[i]}")
                #     print(f"  Target shape: {output[i].shape}")
                #     print(f"  Source shape: {query_no_pad.shape}")


                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),
                        query_no_pad,
                    )
                )
            output = torch.stack(outputs)

            # print(f"\n[After Scatter]")
            # print(f"  Restored output shape: {output.shape}")

            # add learnt embedding for background
            if multi_level_masks is not None:
                background_embedding = [
                    self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
                ]
                background_embedding = torch.cat(background_embedding, dim=1)
                background_embedding.scatter_(1, inds_for_query, 0)
                background_embedding *= (~key_padding_mask).unsqueeze(-1)
                output = output + background_embedding
        #         print(f"  Added background embedding")
        #
        #     print(f"{'=' * 60}\n")
        #
        # print(f"\n{'=' * 70}")
        # print(f"[ENCODER] Final output shape: {output.shape}")
        # print(f"{'=' * 70}\n")

        return output

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points



class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


# class SADetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
#     """Encoder layer of Deformable DETR."""
#
#     def _init_layers(self) -> None:
#         """Initialize self_attn, ffn, and norms."""
#         self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = FFN(**self.ffn_cfg)
#         norms_list = [
#             build_norm_layer(self.norm_cfg, self.embed_dims)[1]
#             for _ in range(2)
#         ]
#         self.norms = ModuleList(norms_list)

class SADetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of SA-Deformable-DETR with salience filtering."""

    def __init__(self, *args, topk_sa=100, **kwargs):
        self.topk_sa = topk_sa  # 👈 添加 salience 参数
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize deformable attention, FFN, and LayerNorms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        self.dropout = self.self_attn_cfg['dropout']

        self.pre_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.self_attn_cfg['num_heads'],
            dropout=self.dropout,
            batch_first=True
        )
        self.pre_dropout = nn.Dropout(self.dropout)
        self.pre_norm = build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        self.dropout1 = nn.Dropout(self.dropout)

        # 2 LayerNorms: after self-attn, after FFN
        self.norms = ModuleList([
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ])

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query,
                query_pos,
                value,
                reference_points,
                spatial_shapes,
                level_start_index,
                key_padding_mask=None,
                score_tgt=None,
                foreground_pre_layer=None):
        """Forward with salience-aware top-k selection."""
        # 1. Top-k pre-attention selection
        if score_tgt is not None and foreground_pre_layer is not None:
            mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
            select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
            select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dims)
            select_tgt = torch.gather(query, 1, select_tgt_index)
            select_pos = torch.gather(query_pos, 1, select_tgt_index)
            query_with_pos = self.with_pos_embed(select_tgt, select_pos)

            tgt2 = self.pre_attention(
                query_with_pos,
                query_with_pos,  # key = query
                select_tgt,
            )[0]
            select_tgt = select_tgt + self.pre_dropout(tgt2)
            select_tgt = self.pre_norm(select_tgt)

            # scatter back to full query
            query = query.scatter(1, select_tgt_index, select_tgt)

        # 2. Deformable Self-Attention
        query2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norms[0](query)

        # 3. FFN
        query = self.ffn(query, identity=query)
        query = self.norms[1](query)

        return query



class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
