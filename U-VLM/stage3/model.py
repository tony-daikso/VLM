"""
Stage 3 model：Stage 1/2 的 ResNet34Encoder（凍結）+ multi-layer visual injection decoder。

Encoder 架構逐層跟 U-VLM/stage1/model.py、U-VLM/stage2/model.py 一致（獨立複製一份，理由同
stage2/model.py 開頭註解：每個 stage 資料夾自包含，方便單獨搬去遠端機器）。

Decoder 是 STAGE3_PLAN.md 第 2.2/2.4/4 節設計的「5 層 Transformer，逐層注入對應 encoder stage」：
reference stage r=N=f4（K=256 個 visual token，見 Align），deep stage 接早期 decoder 層、
shallow stage 接後期層（對應論文 Eq.5，見 STAGE3_PLAN.md 第 0 節「跟論文的關鍵差異」表）。
"""
import torch
import torch.nn as nn
import torchvision


def _adapt_stem_to_grayscale(conv1):
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
        f0 = self.stem(x)
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


class VisualFeatureAligner(nn.Module):
    """把 encoder 的 5 個 stage 對齊成同一個 token 長度 K（reference stage r=N=f4，K=16x16=256），
    再各自投影到 decoder hidden dim D。回傳順序是 decoder layer 1..N 要用的順序：
    [f4, f3, f2, f1, f0]（深 -> 淺，對應論文 Eq.5 的 f_{N-j+1}，見 STAGE3_PLAN.md 第 2.2 節）。"""

    def __init__(self, encoder_channels, hidden_dim, vision_tokens):
        super().__init__()
        self.grid_size = int(vision_tokens ** 0.5)
        assert self.grid_size * self.grid_size == vision_tokens, "vision_tokens 必須是完全平方數"
        self.pool = nn.AdaptiveAvgPool2d(self.grid_size)
        # encoder_channels 是 [64,64,128,256,512]（f0..f4），這裡反過來存成 [f4,f3,f2,f1,f0] 的投影
        reversed_channels = list(reversed(encoder_channels))
        self.projs = nn.ModuleList([nn.Linear(c, hidden_dim) for c in reversed_channels])

    def forward(self, features):
        # features: [f0,f1,f2,f3,f4] -> 反過來处理成 [f4,f3,f2,f1,f0]
        aligned = []
        for proj, feat in zip(self.projs, reversed(features)):
            pooled = self.pool(feat)  # (B, C, grid, grid)
            tokens = pooled.flatten(2).transpose(1, 2)  # (B, K, C)
            aligned.append(proj(tokens))  # (B, K, D)
        return aligned  # [proj(f4), proj(f3), proj(f2), proj(f1), proj(f0)]


def build_hybrid_attention_mask(num_vision, num_text, device):
    """vision token 之間雙向注意力；text token 用 causal attention，且可以看到全部 vision token，
    但 vision token 看不到任何 text token（見 STAGE3_PLAN.md 第 2.2 節，論文 hybrid attention mask）。
    回傳 bool mask（True=禁止注意），跟 src_key_padding_mask 用同一種型別，避免 PyTorch 混用
    bool/float mask 的 deprecation warning。"""
    seq_len = num_vision + num_text
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    mask[:num_vision, num_vision:] = True  # vision 看不到 text
    causal = torch.triu(torch.ones(num_text, num_text, device=device), diagonal=1).bool()
    mask[num_vision:, num_vision:] = causal
    return mask


class VisualInjectionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, dropout,
                 vision_tokens, encoder_channels, max_position=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vision_tokens = vision_tokens
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_position, hidden_dim)
        self.vision_init = nn.Parameter(torch.randn(vision_tokens, hidden_dim) * 0.02)
        # nn.Embedding 預設 std=1，對 transformer 太大（logits scale 會被沖到 sqrt(hidden_dim)
        # 量級），沿用 GPT-2 系列常見的 std=0.02 初始化
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        self.aligner = VisualFeatureAligner(encoder_channels, hidden_dim, vision_tokens)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie embedding

    def forward(self, encoder_features, instruction_ids, report_ids, report_key_padding_mask):
        """
        encoder_features: [f0,f1,f2,f3,f4]，每個 (B,C_i,H_i,W_i)
        instruction_ids: (B, I)
        report_ids: (B, R)
        report_key_padding_mask: (B, R) bool，True 代表這個位置是要餵給模型的合法 token
                                  （沿用 dataset.py 的 report_mask，1=真實 token）
        回傳 logits: (B, R, vocab_size)，只對應 report token 這段
        """
        device = instruction_ids.device
        B = instruction_ids.shape[0]
        K = self.vision_tokens

        aligned_features = self.aligner(encoder_features)  # [proj(f4)..proj(f0)], 每個 (B,K,D)

        vis = self.vision_init.unsqueeze(0).expand(B, -1, -1)
        instr_emb = self.token_embedding(instruction_ids)
        report_emb = self.token_embedding(report_ids)

        h = torch.cat([vis, instr_emb, report_emb], dim=1)
        seq_len = h.shape[1]
        positions = torch.arange(seq_len, device=device)
        h = h + self.pos_embedding(positions).unsqueeze(0)

        num_text = instruction_ids.shape[1] + report_ids.shape[1]
        attn_mask = build_hybrid_attention_mask(K, num_text, device)

        pad_ignore = report_key_padding_mask == 0
        key_padding_mask = torch.cat([
            torch.zeros(B, K + instruction_ids.shape[1], dtype=torch.bool, device=device),
            pad_ignore,
        ], dim=1)

        for layer, feat in zip(self.layers, aligned_features):
            vis_part = h[:, :K, :] + feat
            rest_part = h[:, K:, :]
            h = torch.cat([vis_part, rest_part], dim=1)
            h = layer(h, src_mask=attn_mask, src_key_padding_mask=key_padding_mask)

        h = self.final_norm(h)
        report_start = K + instruction_ids.shape[1]
        logits = self.lm_head(h[:, report_start:, :])
        return logits


class Stage3Model(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        model_cfg = config["model"]
        self.encoder = ResNet34Encoder(pretrained=False)
        self.decoder = VisualInjectionDecoder(
            vocab_size=vocab_size,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            ffn_dim=model_cfg["ffn_dim"],
            dropout=model_cfg["dropout"],
            vision_tokens=model_cfg["vision_tokens"],
            encoder_channels=self.encoder.out_channels,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, image, instruction_ids, report_ids, report_mask):
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(image)
        return self.decoder(features, instruction_ids, report_ids, report_mask)
