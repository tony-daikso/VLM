"""
Encoder 沿用 CXR Stage 1 的 ResNet34Encoder（跟 stage1/model.py 逐層一致，獨立複製一份維持每個
stage 資料夾自包含）。DetectionHead 只用 f3（256 channel, stride 16），CenterNet 風格的三個
輕量 branch：heatmap（26ch, sigmoid）/ size（2ch, w-h）/ offset（2ch, dx-dy），設計理由見
STAGE2_5_PLAN.md 第 4 節。
"""
import torch
import torch.nn as nn
import torchvision


def _adapt_stem_to_grayscale(conv1):
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
        f0 = self.stem(x)
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


def _head_branch(in_channels, hidden_channels, out_channels):
    """Conv3x3+ReLU+Conv1x1，跟 CenterNet 原論文的 head 設計一致，刻意不加 normalization layer。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
    )


class DetectionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.heatmap_branch = _head_branch(in_channels, hidden_channels, num_classes)
        self.size_branch = _head_branch(in_channels, hidden_channels, 2)
        self.offset_branch = _head_branch(in_channels, hidden_channels, 2)

        # heatmap 最後一層 bias 初始化成負值（CenterNet/RetinaNet 的標準技巧）：一開始網路對每個
        # class 都輸出低機率，避免訓練初期 focal loss 被大量的負樣本 pixel 主導、梯度爆走
        nn.init.constant_(self.heatmap_branch[-1].bias, -2.19)

    def forward(self, f3):
        heatmap = torch.sigmoid(self.heatmap_branch(f3))
        size = self.size_branch(f3)
        offset = self.offset_branch(f3)
        return heatmap, size, offset


class Stage2_5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        num_classes = len(config["labels"])
        # pretrained=False：encoder 權重會在 train.py 裡從 Stage 1 的 best_encoder.pt 載入覆蓋
        self.encoder = ResNet34Encoder(pretrained=False)
        self.head = DetectionHead(
            self.encoder.out_channels[3], model_cfg["head_hidden_channels"], num_classes
        )

    def forward(self, x):
        features = self.encoder(x)
        f3 = features[3]
        heatmap, size, offset = self.head(f3)
        return {"heatmap": heatmap, "size": size, "offset": offset}

    def set_encoder_frozen(self, frozen):
        for p in self.encoder.parameters():
            p.requires_grad = not frozen
        self.encoder.train(not frozen)
