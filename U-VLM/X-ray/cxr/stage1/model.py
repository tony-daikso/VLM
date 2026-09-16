"""
共用 2D U-Net 風格 encoder（ResNet34, ImageNet 預訓練）+ 單一 lesion decoder head。

跟 CT 版（U-VLM/CT/stage1/model.py）架構家族完全一致（同款 ResNet34Encoder/DecoderBlock/
SegmentationHead），但只有一個 head，不是三個 -- PadChest-GR 只有 finding 的 box，沒有器官/
血管的 pixel mask（見 STAGE1_PLAN.md 第 1、2 節）。這是完全獨立的一份程式碼，不 import CT 版的
model.py、不共用權重。

Encoder 輸出 5 個 stage 的 multi-scale feature maps {f0..f4}，stride {2,4,8,16,32} -- 這組
feature 之後會被 CXR Stage 2/3 使用，stage 劃分跟 CT 版對齊，方便之後比較兩條 track 的 encoder。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _adapt_stem_to_grayscale(conv1):
    """ImageNet resnet34 的 conv1 是 (64, 3, 7, 7)，改成 (64, 1, 7, 7)，
    權重取 RGB 三個 channel 的平均，保留預訓練特徵而不是重新初始化。"""
    new_conv1 = nn.Conv2d(
        1, conv1.out_channels, kernel_size=conv1.kernel_size,
        stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None,
    )
    with torch.no_grad():
        new_conv1.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
    return new_conv1


class ResNet34Encoder(nn.Module):
    """輸出 5 個 stage 的 feature map，channel 數 [64, 64, 128, 256, 512]，stride [2,4,8,16,32]。"""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet34(weights=weights)

        self.stem = nn.Sequential(_adapt_stem_to_grayscale(backbone.conv1), backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_channels = [64, 64, 128, 256, 512]

    def forward(self, x):
        f0 = self.stem(x)  # stride 2, ch 64
        f1 = self.layer1(self.maxpool(f0))  # stride 4, ch 64
        f2 = self.layer2(f1)  # stride 8, ch 128
        f3 = self.layer3(f2)  # stride 16, ch 256
        f4 = self.layer4(f3)  # stride 32, ch 512
        return [f0, f1, f2, f3, f4]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationHead(nn.Module):
    """對稱 U-Net decoder：從 encoder 最深層 f4 開始，逐層上採樣 + skip connection 接回
    f3/f2/f1/f0，最後再多上採樣一次回到輸入解析度，1x1 conv 輸出。"""

    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super().__init__()
        c0, c1, c2, c3, c4 = encoder_channels
        d0, d1, d2, d3, d4 = decoder_channels

        self.block4 = DecoderBlock(c4, c3, d0)
        self.block3 = DecoderBlock(d0, c2, d1)
        self.block2 = DecoderBlock(d1, c1, d2)
        self.block1 = DecoderBlock(d2, c0, d3)
        self.final_conv = nn.Sequential(
            nn.Conv2d(d3, d4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(d4, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(d4, out_channels, kernel_size=1)

    def forward(self, features, output_size):
        f0, f1, f2, f3, f4 = features
        x = self.block4(f4, f3)
        x = self.block3(x, f2)
        x = self.block2(x, f1)
        x = self.block1(x, f0)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        return self.out_conv(x)


class Stage1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        self.encoder = ResNet34Encoder(pretrained=model_cfg["encoder_pretrained"])
        decoder_channels = model_cfg["decoder_channels"]
        self.lesion_head = SegmentationHead(self.encoder.out_channels, decoder_channels, out_channels=1)

    def forward(self, x):
        output_size = x.shape[-2:]
        features = self.encoder(x)
        return {"lesion_logits": self.lesion_head(features, output_size)}

    def encoder_state_dict(self):
        """給 Stage 2/3 用：只存 encoder 權重，不帶 decoder。"""
        return self.encoder.state_dict()
