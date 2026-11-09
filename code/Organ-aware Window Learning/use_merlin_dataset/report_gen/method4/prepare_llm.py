#!/usr/bin/env python3
"""Make an LLM-only view of Lingshu-7B (Qwen2.5-VL-7B medical) for method 4: Qwen2ForCausalLM config + tokenizer
copies + symlinks to the original safetensors + an index without the vision tower. No weights are copied.

Why it works: Lingshu-7B stores the language model as model.layers.* / model.embed_tokens / model.norm / lm_head
(checked on Spark 2026-11-09, 729 tensors, 384 visual.*), the same key names as Qwen2ForCausalLM; for text-only
positions the 3 M-RoPE sections are identical, so 1-D RoPE (theta 1e6) gives the same result. Our 32 image tokens
are plain embeddings at ordinary sequential positions.

usage (on the host that has the model; writes only the out dir):
  python prepare_llm.py --src ~/Desktop/Tony/models/Lingshu-7B --out ~/Desktop/Tony/models/Lingshu-7B-llm
"""
import argparse
import json
import os
import shutil
from pathlib import Path

KEEP = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "rms_norm_eps", "rope_theta", "max_position_embeddings", "vocab_size", "hidden_act", "attention_dropout",
        "bos_token_id", "eos_token_id", "tie_word_embeddings", "sliding_window", "use_sliding_window",
        "max_window_layers", "initializer_range"]
TOK = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
       "added_tokens.json", "generation_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(os.path.expanduser(a.src)), Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    vl = json.loads((src / "config.json").read_text())
    tc = vl.get("text_config", {})
    cfg = {k: tc.get(k, vl.get(k)) for k in KEEP if k in tc or k in vl}
    cfg.update({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "torch_dtype": "bfloat16",
                "use_cache": True, "rope_scaling": None})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in TOK:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm = {k: v for k, v in idx["weight_map"].items() if not k.startswith("visual.")}
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": idx.get("metadata", {}),
                                                                  "weight_map": wm}, indent=1))
    for shard in sorted(set(wm.values())):
        if not (out / shard).exists():
            os.symlink(src / shard, out / shard)
    print(f"{len(wm)} LLM tensors ({len(idx['weight_map']) - len(wm)} visual dropped) -> {out}")


if __name__ == "__main__":
    main()
