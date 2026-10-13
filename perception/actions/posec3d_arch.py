"""PoseC3D's pose pathway rebuilt in plain PyTorch, so the checkpoint exports.

Same reason as `rtmpose_arch.py`, and the same method. MMAction2's exporter
needs MMAction2, MMEngine and MMCV; on this machine MMCV will not build at all
(no wheel for this Python, and it needs a toolchain plus a torch version it
supports), and mmaction2 pins `numpy<2`, which would downgrade the numpy the
rest of the perception layer runs on. That is a lot of collateral damage to run
`torch.onnx.export` once. So the module graph is rebuilt here from the
checkpoint's own 260 tensors, and `load_pose_only` refuses any mismatch.

Nothing on the inference path imports this. `posec3d.py` runs the exported ONNX
through onnxruntime; this module is used by
`perception/tools/export_posec3d.py` and by the test.

The architecture, read off `pose_only_20230228-fa40054e.pth`:

    ResNet3dSlowOnly, 3 stages, base 32 channels, blocks (4, 6, 3)
      conv1        (32, 17, 1, 7, 7)  -- 17 input channels, one per COCO joint,
                                         and a spatial-only stem: the first
                                         layer must not blur time
      layer1       4 bottlenecks, 32 -> 128, no temporal conv (inflate 0)
      layer2       6 bottlenecks, 64 -> 256, conv1 is (3,1,1): time enters here
      layer3       3 bottlenecks, 128 -> 512, same
      cls_head     avg-pool -> dropout -> Linear(512, 60)   NTU-60

    Input  (N, 17, 32, 56, 56)   Output  (N, 60) logits

    +--------------------------------------------------------------------+
    |  LICENCE: these weights are NTU RGB+D trained, which is research-   |
    |  only under an academic use agreement, and a fine-tune inherits it. |
    |  See LICENCE-NOTES.md. This is the licence the repo flagged first   |
    |  and it is still unresolved.                                        |
    +--------------------------------------------------------------------+
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvModule3d(nn.Sequential):
    """conv3d -> bn -> relu, under the submodule names the checkpoint uses.

    `activate=False` for the layers MMAction2 builds with `act_cfg=None` --
    a bottleneck's third convolution and the downsample shortcut, both of
    which are summed before the ReLU rather than after it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        if activate:
            self.activate = nn.ReLU(inplace=True)


class Bottleneck3d(nn.Module):
    """The standard 1x1 / 3x3 / 1x1 bottleneck, optionally inflated in time.

    Two details that are invisible in the tensor shapes and wrong by default:

    `style='pytorch'` puts the spatial stride on the 3x3 convolution rather
    than the first 1x1. Putting it on the 1x1 throws away three quarters of
    the activations before they are ever convolved.

    `inflate_style='3x1x1'` makes the temporal kernel a separate (3,1,1)
    convolution on conv1, with conv2 staying purely spatial (1,3,3). The
    alternative would be a full 3x3x3, which is a different, larger model --
    and the checkpoint says (3,1,1), so that is what this is.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        inflate: bool = True,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        conv1_kernel = (3, 1, 1) if inflate else (1, 1, 1)
        conv1_padding = (1, 0, 0) if inflate else (0, 0, 0)
        self.conv1 = ConvModule3d(in_channels, planes, conv1_kernel,
                                  stride=(1, 1, 1), padding=conv1_padding)
        self.conv2 = ConvModule3d(
            planes, planes, (1, 3, 3),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
        )
        self.conv3 = ConvModule3d(planes, planes * self.expansion, (1, 1, 1),
                                  activate=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(out + identity)


class ResNet3dSlowOnly(nn.Module):
    """The pose pathway: heatmap volume in, 512-channel features out.

    Note what the stem does *not* do. `conv1` is (1, 7, 7) and both it and the
    pooling have stride 1 in time, so 32 frames stay 32 frames all the way
    through -- every temporal stride is 1. A volume that got downsampled in
    time here would blur exactly the short, fast movements the model exists to
    tell apart.
    """

    def __init__(
        self,
        in_channels: int = 17,
        base_channels: int = 32,
        stage_blocks: tuple[int, ...] = (4, 6, 3),
        # Stage 1 is spatial only. Time is introduced once the features mean
        # something; convolving it at the stem is expensive and adds nothing.
        inflate: tuple[int, ...] = (0, 1, 1),
        spatial_strides: tuple[int, ...] = (2, 2, 2),
        temporal_strides: tuple[int, ...] = (1, 1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = ConvModule3d(in_channels, base_channels, (1, 7, 7),
                                  stride=(1, 1, 1), padding=(0, 3, 3))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                    padding=(0, 1, 1))

        planes = base_channels
        channels = base_channels
        for i, blocks in enumerate(stage_blocks):
            layers = []
            for b in range(blocks):
                downsample = None
                if b == 0:
                    downsample = ConvModule3d(
                        channels, planes * Bottleneck3d.expansion, (1, 1, 1),
                        stride=(temporal_strides[i], spatial_strides[i],
                                spatial_strides[i]),
                        activate=False,
                    )
                layers.append(
                    Bottleneck3d(
                        channels, planes,
                        spatial_stride=spatial_strides[i] if b == 0 else 1,
                        temporal_stride=temporal_strides[i] if b == 0 else 1,
                        inflate=bool(inflate[i]),
                        downsample=downsample,
                    )
                )
                channels = planes * Bottleneck3d.expansion
            setattr(self, f"layer{i + 1}", nn.Sequential(*layers))
            planes *= 2

        self.num_stages = len(stage_blocks)
        self.out_channels = channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.conv1(x))
        for i in range(1, self.num_stages + 1):
            x = getattr(self, f"layer{i}")(x)
        return x


class I3DHead(nn.Module):
    """Global average pool over space and time, then one linear layer.

    The pool is why the clip length is a training-time fact rather than a
    shape constraint the graph would catch: a volume of 24 frames instead of
    32 pools to the same 512 numbers and classifies without complaint, just
    less accurately. That is precisely the silent failure `posec3d.py`
    validates the tube shape to prevent.
    """

    def __init__(self, in_channels: int = 512, num_classes: int = 60,
                 dropout_ratio: float = 0.5) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.dropout(x)
        return self.fc_cls(x.flatten(1))


class PoseC3D(nn.Module):
    """Backbone plus head: one heatmap volume in, 60 NTU logits out.

    Logits, not probabilities. `AbstentionPolicy` applies its own softmax with
    a calibration temperature, and baking a softmax into the graph would hide
    the temperature it needs to divide by.
    """

    def __init__(self, num_classes: int = 60) -> None:
        super().__init__()
        self.backbone = ResNet3dSlowOnly()
        self.cls_head = I3DHead(self.backbone.out_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.cls_head(self.backbone(x))


def load_pose_only(checkpoint_path: str, device: str = "cpu") -> PoseC3D:
    """Build the network and load the published checkpoint, refusing any mismatch.

    Same reasoning as the pose model: a near-miss on a module name leaves a
    layer randomly initialised, and a 60-class classifier with one broken stage
    still returns a confident-looking label. It has to fail here or not at all.
    """
    model = PoseC3D()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match this architecture.\n"
            f"  missing:    {missing[:8]}\n"
            f"  unexpected: {unexpected[:8]}\n"
            "This module was written against pose_only_20230228-fa40054e "
            "(RGBPose-Conv3D pose pathway, NTU-60). A different PoseC3D "
            "variant has different widths or stage counts -- read them with "
            "tools/inspect_checkpoint.py rather than guessing."
        )
    return model.eval().to(device)
