"""Method 4 model: OWL unit tokens (+ type embedding) -> Perceiver resampler (32 latents) -> MLP -> LLM hidden size,
scattered into the prompt at the 32 <|image_pad|> positions of a causal LM (Lingshu-7B LLM part = Qwen2.5-7B).

Text-only ablation = the same LLM / prompt / LoRA with no adapter and no image placeholder.
"""
from __future__ import annotations

import csv

import torch
import torch.nn as nn

FIELDS = ["group", "unit", "region", "window", "pool"]


def read_token_meta(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["tok"]))
    return rows


class TypeEmbedding(nn.Module):
    """Sum of learned embeddings of each token's (group, group/unit, region, window, pool) from token_meta.csv.
    Every study has the same token layout, so this tells the resampler which unit / window / pool a vector is."""

    def __init__(self, meta_rows, dim):
        super().__init__()
        ids, self.vocab = [], {}
        for f in FIELDS:
            vals = [(r["group"] + "/" + r[f]) if f == "unit" else r[f] for r in meta_rows]
            voc = {v: i for i, v in enumerate(sorted(set(vals)))}
            self.vocab[f] = voc
            ids.append([voc[v] for v in vals])
        self.register_buffer("ids", torch.tensor(ids, dtype=torch.long).T.contiguous(), persistent=False)  # [T, 5]
        self.embs = nn.ModuleList(nn.Embedding(len(self.vocab[f]), dim) for f in FIELDS)
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.02)

    def forward(self):
        return sum(e(self.ids[:, i]) for i, e in enumerate(self.embs))  # [T, dim]


class ResamplerLayer(nn.Module):
    def __init__(self, dim, heads, ff_mult):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_ff = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))

    def forward(self, lat, x, key_pad):
        q = self.ln_q(lat)
        kv = torch.cat([self.ln_kv(x), q], 1)                    # Flamingo: latents also attend to themselves
        kp = torch.cat([key_pad, torch.zeros_like(key_pad[:, : q.shape[1]])], 1)
        lat = lat + self.attn(q, kv, kv, key_padding_mask=kp, need_weights=False)[0]
        return lat + self.ff(self.ln_ff(lat))


class ImageAdapter(nn.Module):
    """[B, T, 448] OWL vectors + mask -> [B, n_latents, llm_hidden]."""

    def __init__(self, meta_rows, llm_hidden, in_dim=448, dim=1024, n_latents=32, depth=2, heads=8, ff_mult=4):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.type_emb = TypeEmbedding(meta_rows, dim)
        self.null = nn.Parameter(torch.zeros(1, 1, dim))          # always-valid key: no NaN when a study has no unit
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.layers = nn.ModuleList(ResamplerLayer(dim, heads, ff_mult) for _ in range(depth))
        self.out_ln = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, llm_hidden), nn.GELU(), nn.Linear(llm_hidden, llm_hidden))
        self.n_latents = n_latents

    def forward(self, tok, mask):
        B = tok.shape[0]
        x = self.inp(tok.float()) + self.type_emb()[None]
        x = torch.cat([self.null.expand(B, 1, -1), x], 1)
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask.bool()], 1)
        lat = self.latents[None].expand(B, -1, -1)
        for layer in self.layers:
            lat = layer(lat, x, key_pad)
        return self.proj(self.out_ln(lat))


class ReportGenModel(nn.Module):
    def __init__(self, llm, adapter: ImageAdapter | None, img_token_id: int | None):
        super().__init__()
        self.llm, self.adapter, self.img_token_id = llm, adapter, img_token_id

    def embed(self, input_ids, tok=None, mask=None):
        emb = self.llm.get_input_embeddings()(input_ids)
        if self.adapter is None or tok is None:
            return emb
        img = self.adapter(tok, mask).to(emb.dtype)                # [B, 32, H]
        pos = input_ids == self.img_token_id
        assert int(pos.sum()) == img.shape[0] * img.shape[1], "every row needs exactly n_latents image placeholders"
        return emb.masked_scatter(pos.unsqueeze(-1), img.reshape(-1, img.shape[-1]))

    def forward(self, input_ids, attention_mask, labels=None, tok=None, mask=None):
        return self.llm(inputs_embeds=self.embed(input_ids, tok, mask), attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, tok=None, mask=None, max_new_tokens=320, eos_token_id=None,
                 pad_token_id=None):
        emb = self.embed(input_ids, tok, mask)
        return self.llm.generate(inputs_embeds=emb, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                                 do_sample=False, num_beams=1, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
