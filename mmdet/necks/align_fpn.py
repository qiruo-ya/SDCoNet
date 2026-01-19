# -----------------------------------------------------------------------------
# Licensed under the MIT License.
# The code is based on MMDetection (https://github.com/open-mmlab/mmdetection).
# @Author  : Jixiang Wu
# @Time    : 2022/5/21 下午4:54
# Migrated to MMDetection 3.x
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init, constant_init
from mmdet.registry import MODELS


class FeatureAlign(nn.Module):
    def __init__(self,
                 out_channels=256,
                 kernel_size=3):
        super(FeatureAlign, self).__init__()
        self.out_channels = out_channels

        self.depthwise_conv = nn.Conv2d(
            self.out_channels * 2,
            self.out_channels * 2,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
            groups=self.out_channels * 2)
        self.pointwise_conv = nn.Conv2d(
            self.out_channels * 2,
            2,
            kernel_size=1,
            bias=False)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1.0)
                if m.bias is not None:
                    constant_init(m.bias, 0)

    def forward(self, low_feature, high_feature):
        b, c, h, w = low_feature.size()
        high_feature_up = F.interpolate(
            high_feature, size=(h, w), mode='bilinear', align_corners=True)
        concat_feature = torch.cat([low_feature, high_feature_up], dim=1)
        high_offset = self.pointwise_conv(self.depthwise_conv(concat_feature))
        high_feature_new = self.grid_sample(high_feature, high_offset, (h, w))
        out = low_feature + high_feature_new
        return out

    def grid_sample(self, input, offset, size):
        b, _, h, w = input.size()
        out_h, out_w = size
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(b, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + offset.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output


@MODELS.register_module()
class AlignFPN(BaseModule):
    """Feature Pyramid Network with Feature Alignment.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_ with feature alignment.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 1.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        kernel_size (int): Kernel size for feature alignment. Default: 3.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 num_outs=3,
                 start_level=1,
                 end_level=-1,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(AlignFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1 or end_level is None:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.c3_align = FeatureAlign(
            out_channels=out_channels, kernel_size=kernel_size)
        self.c4_align = FeatureAlign(
            out_channels=out_channels, kernel_size=kernel_size)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple[Tensor]: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        m4 = self.c4_align(laterals[1], laterals[2])
        m3 = self.c3_align(laterals[0], m4)

        p5 = self.fpn_convs[0](laterals[2])
        p4 = self.fpn_convs[1](m4)
        p3 = self.fpn_convs[2](m3)

        outs = [p3, p4, p5]
        return tuple(outs)