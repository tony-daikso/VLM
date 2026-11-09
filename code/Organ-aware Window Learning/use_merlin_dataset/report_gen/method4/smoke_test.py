#!/usr/bin/env python3
"""CPU-only smoke test of the method 4 code with a tiny random Qwen2 and synthetic data (no report text, no GPU).

Checks: backbone loaders (Lingshu-7B-style and Lingshu-I-8B-style checkpoints with key mapping), dataset / collate
shapes, 32 image placeholders, stage 1 (only adapter gets gradients), stage 2 (LoRA + adapter gradients), text-only
ablation, left-padded generation + tag parsing, adapter / LoRA save-load round trip, and train.py / infer.py end to
end for 2 optimizer steps on CPU.
usage: CUDA_VISIBLE_DEVICES= python smoke_test.py --tok <dir with a Qwen2/Lingshu tokenizer> --work <scratch dir>
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_jsonl import make_prompt, make_target  # noqa: E402
from m4_common import BACKBONES, DISEASES, N_IMG_TOKENS, parse_output  # noqa: E402
from m4_data import M4Dataset, collate  # noqa: E402
from m4_model import ImageAdapter, ReportGenModel, read_token_meta  # noqa: E402
from train import load_llm  # noqa: E402

TOKFILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
            "added_tokens.json"]
SYN = {  # synthetic generic sentences (not taken from any report)
    "hepatic_steatosis": "Liver: Mild diffuse fatty infiltration.", "gallstones": "Gallbladder: Normal.",
    "surgically_absent_gallbladder": "Gallbladder: Normal.", "splenomegaly": "Spleen: Normal.",
    "hydronephrosis": "No hydronephrosis.", "ascites": "No free fluid or free air.",
    "free_air": "No free fluid or free air.", "fracture": "Compression deformity of L1."}


def tiny_cfg(vocab):
    import transformers
    return transformers.Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                    rope_theta=1e6, tie_word_embeddings=False)


def write_ckpts(tok_dir, work):
    """tiny random LLM saved three ways: plain causal LM, Lingshu-7B-like VLM, Lingshu-I-8B-like VLM."""
    import transformers
    from safetensors.torch import load_file, save_file
    tk = transformers.AutoTokenizer.from_pretrained(tok_dir)
    vocab = ((len(tk) + 63) // 64) * 64
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(tiny_cfg(vocab)).to(torch.float32)
    plain = work / "tiny_plain"
    m.save_pretrained(plain, safe_serialization=True)
    for f in TOKFILES:
        if (Path(tok_dir) / f).exists():
            shutil.copy2(Path(tok_dir) / f, plain / f)
    sd = load_file(plain / "model.safetensors")
    c = json.loads((plain / "config.json").read_text())
    # Lingshu-7B-like: flat config (qwen2_5_vl, M-RoPE), keys model.* + visual.*
    d7 = work / "tiny_lingshu7b"
    d7.mkdir(exist_ok=True)
    c7 = {k: v for k, v in c.items() if k not in ("architectures", "model_type")}
    c7.update({"architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl",
               "rope_scaling": {"mrope_section": [2, 3, 3], "rope_type": "default", "type": "default"},
               "vision_config": {"depth": 1}})
    (d7 / "config.json").write_text(json.dumps(c7))
    save_file({**sd, "visual.patch_embed.proj.weight": torch.zeros(4, 4)}, d7 / "model.safetensors",
              metadata={"format": "pt"})
    # Lingshu-I-8B-like: text_config, keys language_model.model.* / language_model.lm_head + vision_tower.*
    d8 = work / "tiny_lingshu_i8b"
    d8.mkdir(exist_ok=True)
    c8 = {"architectures": ["InternVLForConditionalGeneration"], "model_type": "internvl",
          "text_config": {**{k: v for k, v in c.items() if k not in ("architectures",)}, "model_type": "qwen2",
                          "rope_scaling": {"factor": 2.0, "rope_type": "dynamic", "type": "dynamic"}},
          "vision_config": {"depth": 1}}
    (d8 / "config.json").write_text(json.dumps(c8))
    save_file({**{"language_model." + k: v for k, v in sd.items()}, "vision_tower.x.weight": torch.zeros(4, 4)},
              d8 / "model.safetensors", metadata={"format": "pt"})
    for d in (d7, d8):
        for f in TOKFILES:
            if (Path(tok_dir) / f).exists():
                shutil.copy2(Path(tok_dir) / f, d / f)
    return plain, d7, d8, sd


def write_data(work, n=6, T=50):
    feat = work / "features"
    (feat / "main").mkdir(parents=True, exist_ok=True)
    groups = [("organ3", f"u{i}", "", w, "mean") for i in range(5) for w in ("baseline", "bone")] + \
             [("ru_x", f"k{i}", f"r{r}", "w0", p) for i in range(5) for r in range(4) for p in ("mean", "max")]
    groups = groups[:T]
    with open(feat / "token_meta.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tok", "group", "unit", "region", "window", "pool"])
        wr.writerows([i, *g] for i, g in enumerate(groups))
    rng = np.random.default_rng(0)
    rows = {"train": [], "val": []}
    for i in range(n):
        cid = f"SYN{i:03d}"
        mask = rng.random(T) > 0.2
        if i == 0:
            mask[:] = False                                   # a study with no unit at all must not give NaN
        np.savez(feat / "main" / f"{cid}.npz", tok=(rng.standard_normal((T, 448)) * mask[:, None]).astype(np.float16),
                 mask=mask)
        sp = "train" if i < n - 2 else "val"
        tgt = {"case_id": cid, "split": sp, "diseases": {d: {"sentence_clean": s} for d, s in SYN.items()}}
        probs = {d: float(rng.random()) for d in DISEASES}
        meas = {"abd_eq_rmax": "21.5", "sp_cc_main_cm": "10.2", "sp_ml": "180"}
        calib = {"aaa_cm": {"col": "abd_eq_rmax", "a": 1.1, "b": -0.3}, "spleen_cm": {"col": "sp_cc_main_cm", "a": 0.7, "b": 4.9}}
        t, tds = make_target(tgt)
        rows[sp].append({"case_id": cid, "split": sp, "feature_path": str(feat / "main" / f"{cid}.npz"),
                         "prompt": make_prompt({"phase": "portal_venous", "manufacturer": "GE MEDICAL SYSTEMS",
                                                "manufacturermodelname": "Discovery CT750 HD"}, probs, meas, calib),
                         "target": t, "target_diseases": tds})
    data = work / "data"
    data.mkdir(exist_ok=True)
    for sp, rr in rows.items():
        with open(data / f"{sp}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rr)
    return feat, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only"
    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    torch.manual_seed(0)
    plain, d7, d8, sd = write_ckpts(a.tok, work)
    feat, data = write_data(work)
    res = {}

    # 1. loaders: both VLM layouts give the same LM weights as the plain checkpoint
    for name, d in (("lingshu7b", d7), ("lingshu_i8b", d8)):
        bb = dict(BACKBONES[name])
        BACKBONES[name + "_tiny"] = {**bb, "hidden": 64}
        tk, llm = load_llm(str(d), torch.float32, torch.device("cpu"), name + "_tiny")
        w = llm.state_dict()
        same = all(torch.equal(w[k], sd[k]) for k in sd if k in w) and set(sd) <= set(w)
        res[f"loader_{name}"] = bool(same)
        assert same, name
    tk, llm = load_llm(str(plain), torch.float32, torch.device("cpu"), "tiny_random")
    meta = read_token_meta(feat / "token_meta.csv")

    # 2. dataset / collate
    ds = M4Dataset(data / "train.jsonl", tk, True, 1024, backbone="tiny_random")
    b = collate([ds[i] for i in range(len(ds))], tk.pad_token_id)
    img_id = tk.convert_tokens_to_ids(BACKBONES["tiny_random"]["img_token"])
    res["batch_shapes"] = {k: list(v.shape) for k, v in b.items() if torch.is_tensor(v)}
    res["img_tokens_per_row"] = (b["input_ids"] == img_id).sum(1).tolist()
    assert all(x == N_IMG_TOKENS for x in res["img_tokens_per_row"])
    res["target_tokens_per_row"] = (b["labels"] != -100).sum(1).tolist()

    # 3. stage 1: only adapter trains
    adapter = ImageAdapter(meta, llm.config.hidden_size, dim=32, depth=2, heads=4)
    llm.requires_grad_(False)
    model = ReportGenModel(llm, adapter, img_id)
    out = model(b["input_ids"], b["attention_mask"], b["labels"], b["tok"], b["mask"])
    out.loss.backward()
    ga = sum(p.grad.abs().sum().item() for p in adapter.parameters() if p.grad is not None)
    gl = sum(1 for p in llm.parameters() if p.grad is not None)
    res["stage1"] = {"loss": round(out.loss.item(), 3), "adapter_grad_sum": ga, "llm_params_with_grad": gl,
                     "finite": bool(torch.isfinite(out.loss))}
    assert ga > 0 and gl == 0 and torch.isfinite(out.loss)

    # 4. stage 2: LoRA r16/a32 + adapter
    from peft import LoraConfig, get_peft_model
    adapter.zero_grad()
    pl = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                        "up_proj", "down_proj"]))
    model2 = ReportGenModel(pl, adapter, img_id)
    out = model2(b["input_ids"], b["attention_mask"], b["labels"], b["tok"], b["mask"])
    out.loss.backward()
    glora = sum(p.grad.abs().sum().item() for n, p in pl.named_parameters() if "lora_" in n and p.grad is not None)
    res["stage2"] = {"loss": round(out.loss.item(), 3), "lora_grad_sum": glora,
                     "adapter_grad_sum": sum(p.grad.abs().sum().item() for p in adapter.parameters() if p.grad is not None),
                     "lora_trainable": sum(p.numel() for n, p in pl.named_parameters() if p.requires_grad)}
    assert glora > 0

    # 5. text-only ablation
    dst = M4Dataset(data / "train.jsonl", tk, False, 1024, backbone="tiny_random")
    bt = collate([dst[i] for i in range(len(dst))], tk.pad_token_id)
    assert "tok" not in bt and int((bt["input_ids"] == img_id).sum()) == 0
    out = ReportGenModel(pl, None, None)(bt["input_ids"], bt["attention_mask"], bt["labels"])
    res["text_only"] = {"loss": round(out.loss.item(), 3), "prompt_len": [len(dst[i]["prompt_ids"]) for i in range(2)],
                        "prompt_len_with_image": [len(ds[i]["prompt_ids"]) for i in range(2)]}

    # 6. generation (left padding) + tag parsing
    dsg = M4Dataset(data / "val.jsonl", tk, True, backbone="tiny_random", train=False)
    bg = collate([dsg[i] for i in range(len(dsg))], tk.pad_token_id, train=False)
    pl.eval()
    pl.config.use_cache = True
    g = model2.generate(bg["input_ids"], bg["attention_mask"], bg["tok"], bg["mask"], max_new_tokens=8,
                        eos_token_id=tk.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tk.pad_token_id)
    res["generate_shape"] = list(g.shape)
    res["parse_example"] = parse_output(make_target({"diseases": {d: {"sentence_clean": s} for d, s in SYN.items()}})[0])
    assert set(res["parse_example"]) == set(SYN)

    # 7. save / load round trip
    torch.save(adapter.state_dict(), work / "adapter.pt")
    a2 = ImageAdapter(meta, llm.config.hidden_size, dim=32, depth=2, heads=4)
    a2.load_state_dict(torch.load(work / "adapter.pt"))
    res["adapter_roundtrip"] = all(torch.equal(x, y) for x, y in zip(adapter.state_dict().values(), a2.state_dict().values()))

    # 8. train.py + infer.py end to end, 2 optimizer steps on CPU (stage 1 -> stage 2 -> infer; text-only stage 2)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    py = sys.executable
    common = ["--llm", str(plain), "--data", str(data), "--meta", str(feat / "token_meta.csv"), "--device", "cpu",
              "--dtype", "fp32", "--backbone", "tiny_random", "--micro-bs", "2", "--grad-accum", "1",
              "--max-steps", "2", "--num-workers", "0", "--res-dim", "32", "--res-heads", "4", "--log-every", "1"]
    cmds = [[py, str(HERE / "train.py"), "--stage", "1", "--out", str(work / "runs/s1"), *common],
            [py, str(HERE / "train.py"), "--stage", "2", "--init-adapter", str(work / "runs/s1/final/adapter.pt"),
             "--out", str(work / "runs/s2"), *common],
            [py, str(HERE / "train.py"), "--stage", "2", "--text-only", "--out", str(work / "runs/text"), *common],
            [py, str(HERE / "infer.py"), "--llm", str(plain), "--run", str(work / "runs/s2/final"), "--data",
             str(data / "val.jsonl"), "--out", str(work / "gen_val.jsonl"), "--device", "cpu", "--dtype", "fp32",
             "--max-new-tokens", "8", "--batch", "2"],
            [py, str(HERE / "infer.py"), "--llm", str(plain), "--run", str(work / "runs/text/final"), "--text-only",
             "--data", str(data / "val.jsonl"), "--out", str(work / "gen_val_text.jsonl"), "--device", "cpu",
             "--dtype", "fp32", "--max-new-tokens", "8", "--batch", "2"]]
    for c in cmds:
        p = subprocess.run(c, env=env, capture_output=True, text=True)
        tag = Path(c[1]).stem + ":" + c[c.index("--out") + 1].split("/")[-1]
        res[tag] = "ok" if p.returncode == 0 else ("FAIL " + p.stderr[-1500:])
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-3000:])
            break
    res["saved_files"] = sorted(str(x.relative_to(work)) for x in (work / "runs").rglob("*") if x.is_file())[:40] \
        if (work / "runs").exists() else []
    print(json.dumps(res, indent=1, default=str))
    (work / "smoke_result.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
