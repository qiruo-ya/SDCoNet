# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Callable, List, Optional
import warnings
# from typing import Callable, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from mmdet.registry import MODELS
# from models.bricks.basic import SqueezeAndExcitation
# from models.bricks.misc import Conv2dNormActivation


class RepVggPluXBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation_layer: nn.Module = nn.ReLU,
            inplace: bool = True,
            groups: int = 4,
            alpha: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_layer(inplace=True)

        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            activation_layer=None,
            inplace=inplace,
        )
        self.conv2 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            activation_layer=None,
            inplace=inplace,
        )
        self.alpha = nn.Parameter(torch.tensor(1.0)) if alpha else 1.0

        self.se_module = SqueezeAndExcitation(channels=out_channels, )

        if self.in_channels != self.out_channels:
            self.identity = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x) + self.alpha * self.conv2(x)
        y = self.se_module(self.activation(y))
        return y + self.identity(x)


class CSPRepPluXLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int = 3,
            expansion: float = 1.0,
            groups: int = 4,
            norm_layer: nn.Module = nn.BatchNorm2d,
            activation_layer: nn.Module = nn.SiLU,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=True,
        )
        self.conv2 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=True,
        )
        self.bottlenecks = nn.Sequential(
            *[
                RepVggPluXBlock(
                    hidden_channels,
                    hidden_channels,
                    groups=groups,
                    activation_layer=activation_layer,
                ) for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = Conv2dNormActivation(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.bottlenecks(self.conv1(x)) + self.conv2(x)
        x = self.conv3(x)
        return x


@MODELS.register_module()
class RepVGGPluXNetwork(nn.Module):
    def __init__(
            self,
            in_channels_list: List[int],
            out_channels_list: List[int],
            groups: int = 4,
            norm_layer: nn.Module = nn.BatchNorm2d,
            activation: nn.Module = nn.SiLU,
            extra_block: bool = False,
    ):
        """The implementation RepVGGPluXNetwork, the network is basically built with RepVGGPluxBlock
        upon PathAggregationNetwork.

        :param in_channels_list: input channels list, example: [256, 512, 1024, 2048]
        :param out_channels_list: output channel list, example: [256, 512, 1024, 2048]
        :param groups: number of groups used on GroupConvolution in RepVGGPluXBlock, defaults to 4
        :param norm_layer: norm layer type, defaults to nn.BatchNorm2d
        :param activation: activation layer type, defaults to nn.SiLU
        :param extra_block: whether to add an extra block, defaults to False
        """

        super(RepVGGPluXNetwork, self).__init__()
        for idx in range(len(in_channels_list)):
            if in_channels_list[idx] == 0:
                raise ValueError("in_channels=0 is currently not supported")

        self.lateral_convs = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for idx in range(1, len(out_channels_list)):
            lateral_conv_module = Conv2dNormActivation(
                out_channels_list[idx],
                out_channels_list[idx - 1],
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation,
                inplace=True,
            )
            layer_block_module = CSPRepPluXLayer(
                out_channels_list[idx - 1] * 2,
                out_channels_list[idx - 1],
                groups=groups,
                norm_layer=norm_layer,
                activation_layer=activation,
            )
            self.lateral_convs.append(lateral_conv_module)
            self.layer_blocks.append(layer_block_module)

        self.downsample_blocks = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(in_channels_list) - 1):
            downsample_block_module = Conv2dNormActivation(
                out_channels_list[idx],
                out_channels_list[idx + 1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=activation,
                inplace=True,
            )
            pan_block_module = CSPRepPluXLayer(
                out_channels_list[idx + 1] * 2,
                out_channels_list[idx + 1],
                groups=groups,
                norm_layer=norm_layer,
                activation_layer=activation,
            )
            self.downsample_blocks.append(downsample_block_module)
            self.pan_blocks.append(pan_block_module)
        self.extra_block = extra_block

        self.init_weights()

    def init_weights(self):
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: OrderedDict):
        keys = list(x.keys())
        x = list(x.values())
        assert len(x) == len(self.layer_blocks) + 1

        # top down path
        results = x
        inner_outs = [results[-1]]
        for idx in range(len(results) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = results[idx - 1]
            feat_high = self.lateral_convs[idx - 1](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(
                feat_high,
                size=feat_low.shape[-2:],
                mode="nearest",
            )
            inner_out = self.layer_blocks[idx - 1](torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # bottom up path
        results = [inner_outs[0]]
        for idx in range(len(inner_outs) - 1):
            feat_low = results[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_blocks[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_high], dim=1))
            results.append(out)

        # output layer
        output = OrderedDict()
        for idx in range(len(x)):
            output[keys[idx]] = results[idx]
        # extra block
        if self.extra_block:
            output["pool"] = F.max_pool2d(list(output.values())[-1], 1, 2, 0)

        return output


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.se_module = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        nn.init.kaiming_normal_(self.conv_mask.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        batch, channel, height, width = x.shape
        # spatial pool
        # b, 1, c, h * w
        input_x = x.view(batch, channel, height * width).unsqueeze(1)
        # b, 1, h * w, 1
        context_mask = self.conv_mask(x).view(batch, 1, height * width)
        context_mask = self.softmax(context_mask).unsqueeze(-1)
        # b, 1, c, 1
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)
        return self.se_module(context) * x



class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )