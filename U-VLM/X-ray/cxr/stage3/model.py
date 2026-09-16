"""
Encoder 沿用 CXR Stage 2 fine-tuned 過的 ResNet34Encoder（跟 stage1/stage2 逐層一致，獨立複製
一份維持每個 stage 資料夾自包含）。只取 f4（512 channel, stride 32, 16x16=256 個空間位置）投影
成 GPT-2 hidden size 的 256 個 visual token + 學到的 2D 位置編碼，當作 GPT-2 每一層
cross-attention 的 key/value 來源（見 STAGE3_PLAN.md 第 2、5 節）。

Cross-attention 是用 `transformers` GPT2 原生支援的 `add_cross_attention=True` 機制（不用手動
改 GPT2Block），載入 pretrained GPT-2 權重時 self-attention/MLP/embedding 都從 checkpoint 正常
載入，新增的 12 層 crossattention 權重是隨機初始化，靠 Stage 3 的訓練學起來。
"""
import torch
import torch.nn as nn
import torchvision
from transformers import GPT2Config, GPT2LMHeadModel


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


class VisualTokenizer(nn.Module):
    """f4 (B, 512, H, W) -> (B, H*W, lm_hidden) visual token 序列，加上學到的位置編碼。"""

    def __init__(self, in_channels, lm_hidden, max_tokens):
        super().__init__()
        self.proj = nn.Linear(in_channels, lm_hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, lm_hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, f4):
        B, C, H, W = f4.shape
        tokens = f4.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.proj(tokens)
        return tokens + self.pos_embed[:, : H * W]


class Stage3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        self.freeze_encoder = model_cfg["freeze_encoder"]

        # pretrained=False：encoder 權重會在 train.py 裡從 Stage 2 的 best_encoder.pt 載入覆蓋
        self.encoder = ResNet34Encoder(pretrained=False)

        lm_config = GPT2Config.from_pretrained(model_cfg["lm_name"])
        lm_config.add_cross_attention = True
        self.lm = GPT2LMHeadModel.from_pretrained(model_cfg["lm_name"], config=lm_config)

        num_visual_tokens = (config["image"]["resolution"] // 32) ** 2  # f4 stride=32
        self.visual_tokenizer = VisualTokenizer(
            self.encoder.out_channels[-1], lm_config.n_embd, num_visual_tokens
        )

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def encode_image(self, image):
        if self.freeze_encoder:
            with torch.no_grad():
                f4 = self.encoder(image)[-1]
        else:
            f4 = self.encoder(image)[-1]
        return self.visual_tokenizer(f4)

    def forward(self, image, input_ids, attention_mask, labels=None):
        visual_tokens = self.encode_image(image)
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_tokens,
            labels=labels,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def generate(self, image, tokenizer, max_new_tokens, num_beams):
        visual_tokens = self.encode_image(image)
        batch_size = image.shape[0]
        input_ids = torch.full(
            (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=image.device
        )
        attention_mask = torch.ones_like(input_ids)
        output = self.lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return output
