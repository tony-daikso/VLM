"""
Encoder 沿用 Stage 1 的 ResNet34Encoder 架構定義（跟 U-VLM/stage1/model.py 逐層一致，獨立複製
一份是為了維持每個 stage 資料夾自包含、方便單獨搬去遠端機器訓練，見 HANDOFF.md/REMOTE_SETUP.md
的搬機器流程——如果要改 encoder 架構，這份跟 stage1/model.py 要一起改，否則
best_encoder.pt 的 state_dict 會對不上）。

新增的 ClassificationHead 只用 encoder 最深層 f4（512 channel, 全域語義最抽象）做 GAP + 2 層 FC，
輸出 18 維 sigmoid logits（不用 softmax，因為同一筆樣本可以同時有多個陽性標籤），
設計理由見 STAGE2_PLAN.md 第 4 節。
"""
import torch
import torch.nn as nn
import torchvision


def _adapt_stem_to_grayscale(conv1):
    """ImageNet resnet34 的 conv1 是 (64, 3, 7, 7)，改成 (64, 1, 7, 7)，
    權重取 RGB 三個 channel 的平均，保留預訓練特徵而不是重新初始化。"""
    new_conv1 = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size,
                           stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None)
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


class ClassificationHead(nn.Module):
    """GAP(f4) -> Linear(512->256) -> LeakyReLU -> Dropout -> Linear(256->num_labels)。
    刻意做得很輕量，不讓 head 自己就能記住訓練集，逼 encoder 的 feature 要真的有用。"""

    def __init__(self, in_channels, hidden_dim, num_labels, dropout):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, f4):
        x = self.pool(f4).flatten(1)
        return self.fc(x)


class Stage2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        num_labels = len(config["labels"])
        # pretrained=False：encoder 權重會在 train.py 裡從 Stage 1 的 best_encoder.pt 載入覆蓋，
        # 這裡的隨機初始化只是佔位，實際跑訓練前一定要載入 Stage 1 權重
        self.encoder = ResNet34Encoder(pretrained=False)
        self.head = ClassificationHead(
            self.encoder.out_channels[-1], model_cfg["head_hidden_dim"], num_labels, model_cfg["dropout"]
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features[-1])

    def set_encoder_frozen(self, frozen):
        """凍結時連 BatchNorm 也切成 eval，不讓 running mean/var 跟著訓練 batch 漂移
        （否則就算權重凍結，BN 的統計量還是會被訓練資料影響，等於沒真的凍結）。"""
        for p in self.encoder.parameters():
            p.requires_grad = not frozen
        self.encoder.train(not frozen)
