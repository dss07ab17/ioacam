"""RTMPose-t rebuilt in plain PyTorch, so the checkpoint can be exported.

This file exists for one reason: to turn the published `.pth` into an ONNX
graph without adding MMPose, MMDetection, MMEngine and MMCV to the build. That
stack pins its own torch and numpy versions, does not install cleanly on the
board, and would be dragged in only to run `torch.onnx.export` once. Every
width here is read off the checkpoint's own tensor shapes rather than copied
from a config, and `load_rtmpose_tiny` refuses any mismatch -- if a single name
or shape were wrong the load would fail rather than quietly leave a layer at
its random initialisation and return 17 plausible-looking keypoints.

Nothing on the inference path imports this. `rtmpose.py` runs the exported ONNX
through onnxruntime; this module is used by `perception/tools/export_rtmpose.py`
and, as a fallback, by the test, so the decode path can be checked against real
weights on a machine that has torch but no exported graph yet.

The architecture, from the 218 tensors in
`rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192`:

    backbone   CSPNeXt-P5, deepen 0.167, widen 0.375: stem 12/12/24, then
               24->48->96->192->384, one CSPNeXt block per stage, SPP in
               stage 4, channel attention on every CSP layer
    head       RTMCC: 7x7 conv to 17 channels, 6x8 feature map flattened to
               48, ScaleNorm+Linear to 256, one gated attention unit, then
               two bias-free Linears to 384 x-bins and 512 y-bins

Licence: MMPose is Apache-2.0 and so is this reimplementation of its module
graph. The weights carry their training data's terms -- COCO and AI Challenger.
See LICENCE-NOTES.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# MMCV's ConvModule defaults, not torch's. Small, but it changes the
# activations and costs nothing to get right.
BN_EPS = 1e-3
BN_MOMENTUM = 0.03


class ConvModule(nn.Sequential):
    """conv -> bn -> SiLU, under the submodule names the checkpoint uses."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.activate = nn.SiLU(inplace=True)


class DepthwiseSeparableConvModule(nn.Sequential):
    """kxk depthwise then 1x1 pointwise, each with its own BN and SiLU."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)


class ChannelAttention(nn.Module):
    """Squeeze-excite with a hard sigmoid, over the concatenated CSP halves."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.act(self.fc(x.mean((2, 3), keepdim=True)))


class CSPNeXtBlock(nn.Module):
    """3x3 then 5x5-depthwise, residual where the stage allows it."""

    def __init__(self, channels: int, add_identity: bool = True) -> None:
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(channels, channels, 5, padding=2)
        self.add_identity = add_identity

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    """Split, transform one half, attend over the concatenation, project.

    The attention runs on the concatenated tensor *before* `final_conv`, which
    is what the checkpoint's `attention.fc` width says: it is the layer's
    output width, not the half width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        add_identity: bool = True,
        expand_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        mid = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid, 1)
        self.short_conv = ConvModule(in_channels, mid, 1)
        self.final_conv = ConvModule(2 * mid, out_channels, 1)
        self.blocks = nn.Sequential(
            *[CSPNeXtBlock(mid, add_identity) for _ in range(num_blocks)]
        )
        self.attention = ChannelAttention(2 * mid)

    def forward(self, x: Tensor) -> Tensor:
        short = self.short_conv(x)
        main = self.blocks(self.main_conv(x))
        return self.final_conv(self.attention(torch.cat((main, short), dim=1)))


class SPPBottleneck(nn.Module):
    """Parallel 5/9/13 max-pools concatenated with the input."""

    def __init__(self, channels: int, kernel_sizes=(5, 9, 13)) -> None:
        super().__init__()
        mid = channels // 2
        self.conv1 = ConvModule(channels, mid, 1)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes]
        )
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


class CSPNeXt(nn.Module):
    """CSPNeXt-tiny. Only the last stage is consumed; RTMPose has no neck."""

    # in, out, blocks, add_identity, spp -- P5 at widen 0.375, deepen 0.167.
    ARCH = (
        (24, 48, 1, True, False),
        (48, 96, 1, True, False),
        (96, 192, 1, True, False),
        (192, 384, 1, False, True),
    )

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvModule(3, 12, 3, stride=2, padding=1),
            ConvModule(12, 12, 3, stride=1, padding=1),
            ConvModule(12, 24, 3, stride=1, padding=1),
        )
        for i, (cin, cout, blocks, identity, spp) in enumerate(self.ARCH, start=1):
            layers = [ConvModule(cin, cout, 3, stride=2, padding=1)]
            if spp:
                layers.append(SPPBottleneck(cout))
            layers.append(CSPLayer(cout, cout, blocks, identity))
            setattr(self, f"stage{i}", nn.Sequential(*layers))

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for i in range(1, len(self.ARCH) + 1):
            x = getattr(self, f"stage{i}")(x)
        return x


class ScaleNorm(nn.Module):
    """L2 norm over the last axis with one learned scalar. Cheaper than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    """Per-channel learned scale on the residual branch."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class RTMCCBlock(nn.Module):
    """The gated attention unit the SimCC head runs over the 17 keypoint tokens.

    One projection produces the gate, the values, and a shared 128-d base; the
    base is scaled and shifted into a query and a key by `gamma` and `beta`.
    The attention kernel is a squared ReLU rather than a softmax, which is part
    of why the exported graph stays cheap enough to be worth putting on an NPU.

    This checkpoint has no relative-bias tensors and no positional encoding, so
    neither is implemented. Carrying code no weights can exercise would only
    invite someone to enable it against a model that was never trained with it.
    """

    def __init__(
        self,
        in_dims: int = 256,
        out_dims: int = 256,
        s: int = 128,
        expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        self.s = s
        self.e = int(in_dims * expansion_factor)
        self.uv = nn.Linear(in_dims, 2 * self.e + s, bias=False)
        self.o = nn.Linear(self.e, out_dims, bias=False)
        self.gamma = nn.Parameter(torch.rand((2, s)))
        self.beta = nn.Parameter(torch.rand((2, s)))
        self.ln = ScaleNorm(in_dims)
        self.act_fn = nn.SiLU(inplace=True)
        self.res_scale = Scale(in_dims)
        self.sqrt_s = math.sqrt(s)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        uv = self.uv(self.ln(x))
        u, v, base = torch.split(self.act_fn(uv), [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        q, k = torch.unbind(base, dim=2)
        kernel = torch.square(F.relu(torch.bmm(q, k.permute(0, 2, 1)) / self.sqrt_s))
        return self.res_scale(shortcut) + self.o(u * torch.bmm(kernel, v))


class RTMCCHead(nn.Module):
    """SimCC head: one 1-D classification over x bins, another over y bins.

    Coordinate regression as classification is the trick. Predicting which of
    384 columns and which of 512 rows a joint falls in gives sub-pixel-capable
    output from a 6x8 feature map, and the winning bin's value doubles as the
    confidence this pipeline needs -- `posetube.py` uses it as blob amplitude,
    so an uncertain joint has to come back faint rather than crisp.
    """

    def __init__(
        self,
        in_channels: int = 384,
        num_keypoints: int = 17,
        input_size: tuple[int, int] = (192, 256),
        in_featuremap_size: tuple[int, int] = (6, 8),
        simcc_split_ratio: float = 2.0,
        hidden_dims: int = 256,
    ) -> None:
        super().__init__()
        flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]
        self.final_layer = nn.Conv2d(in_channels, num_keypoints, 7, stride=1, padding=3)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims), nn.Linear(flatten_dims, hidden_dims, bias=False)
        )
        self.gau = RTMCCBlock(hidden_dims, hidden_dims)
        self.cls_x = nn.Linear(
            hidden_dims, int(input_size[0] * simcc_split_ratio), bias=False
        )
        self.cls_y = nn.Linear(
            hidden_dims, int(input_size[1] * simcc_split_ratio), bias=False
        )

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        feats = torch.flatten(self.final_layer(feats), 2)
        feats = self.gau(self.mlp(feats))
        return self.cls_x(feats), self.cls_y(feats)


class RTMPose(nn.Module):
    """Backbone plus head: a batch of 256x192 crops in, (simcc_x, simcc_y) out."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = CSPNeXt()
        self.head = RTMCCHead()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.head(self.backbone(x))


def load_rtmpose_tiny(checkpoint_path: str, device: str = "cpu") -> RTMPose:
    """Build the network and load the published checkpoint, refusing any mismatch.

    Strictness is the point. A near-miss on a module name would otherwise leave
    a layer at its random initialisation, and the model would still run and
    still return 17 plausible-looking keypoints -- exactly the silent failure
    this repo keeps designing against.
    """
    model = RTMPose()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state, strict=False)
    # num_batches_tracked is a BN counter, not a parameter, and means nothing
    # at eval. Anything else is a real architecture mismatch.
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match this architecture.\n"
            f"  missing:    {missing[:8]}\n"
            f"  unexpected: {unexpected[:8]}\n"
            "This module was written against rtmpose-tiny_simcc-aic-coco at "
            "256x192. A different variant has different widths -- read them "
            "with tools/inspect_checkpoint.py rather than guessing."
        )
    return model.eval().to(device)
