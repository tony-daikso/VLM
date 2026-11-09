#!/usr/bin/env python3
"""CPU-only smoke test of the method 4 code with a tiny random Qwen2 and synthetic data (no report text, no GPU).

Checks: backbone loaders (Lingshu-7B-style and Lingshu-I-8B-style checkpoints with key mapping), dataset / collate
shapes, 32 image placeholders, stage 1 (only adapter gets gradients), stage 2 (LoRA + adapter gradients), text-only
ablation, left-padded generation + tag parsing, adapter / LoRA save-load round trip, and train.py / infer.py end to
end for 2 optimizer steps on CPU; cue-source switches (--cue-source gt_dropout, --meas-dropout) and the error-cue test
(infer.py --cue-flip-*) on synthetic labels / scores.
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
from build_jsonl import attr_terms, make_prompt, make_target  # noqa: E402
from m4_common import (BACKBONES, DISEASES, N_IMG_TOKENS, RES_GROUPS, flip_cues, parse_cue, parse_output,  # noqa: E402
                       render_cue_line, swap_lr)
from m4_data import CueGT, M4Dataset, collate, lr_permutation  # noqa: E402
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


SYN_UNITS = [("organ3", u, "", w, "mean") for u in ("liver", "gallbladder", "spleen", "kidney_left", "kidney_right",
                                                    "aorta", "small_bowel", "colon", "lumbar_spine")
             for w in ("baseline", "bone")] + \
            [(g, u, "", "bone", p) for g, us in (("bone3", ("L1", "L2", "ribs_left", "ribs_right", "hip_left",
                                                            "hip_right", "sacrum")),
                                                 ("vert15", ("L1", "L2")), ("rib3", ("rib_left_5", "rib_right_5")),
                                                 ("rib15", ("rib_left_5", "rib_right_5")))
             for u in us for p in ("mean", "max")] + \
            [("ru_hydronephrosis", f"k{k}", "r0", "w0", p) for k in (0, 1) for p in ("mean", "max")] + \
            [("ru_ascites", "k0", "r0", "w0", "mean"), ("ru_free_air", "k0", "r0", "w0", "mean"),
             ("ru_atherosclerosis", "k0", "r0", "w0", "mean"), ("ru_gallstones", "k0", "r0", "w0", "mean"),
             ("ru_bile_duct", "k0", "r0", "w0", "mean"), ("ru_bowel", "k1", "r0", "w0", "mean"),
             ("ru_rlq", "k0", "r0", "w0", "mean"), ("ru_pelvis_fx", "k3", "r0", "w0", "mean")]
SYN_LINES = [  # B-style target_lines, synthetic generic text
    ("Liver and biliary tree: Mild diffuse fatty infiltration.", ["hepatic_steatosis"]),
    ("Gallbladder: Normal.", ["gallstones", "surgically_absent_gallbladder"]),
    ("Spleen: Normal.", ["splenomegaly"]),
    ("Kidneys and ureters: Moderate left hydronephrosis.", ["hydronephrosis"]),
    ("Peritoneal cavity: No free fluid or free air.", ["ascites", "free_air"]),
    ("Musculoskeletal: Compression fracture of L1 with 30% height loss. Fracture of the right 5th rib. R hip normal.",
     ["fracture"]),
]
SYN_DIS = {"hepatic_steatosis": ("pos", {"grade": "mild", "distribution": "diffuse"}),
           "gallstones": ("neg", {}), "surgically_absent_gallbladder": ("neg", {}), "splenomegaly": ("neg", {}),
           "hydronephrosis": ("pos", {"side": "left", "grade": "moderate"}), "ascites": ("neg", {}),
           "free_air": ("neg", {}),
           "fracture": ("pos", {"vertebral_levels": ["L1"], "sites": ["vertebra", "rib"], "rib_desc": ["right 5th rib"],
                                "vertebral_compression": "yes", "acuity": "not_stated"})}


def write_data(work, n=6):
    """synthetic features on a small real-named token layout + B-style target rows (no report text)."""
    from build_unit_labels import case_units  # noqa: F401  (exercised through its CLI below)
    feat = work / "features"
    (feat / "main").mkdir(parents=True, exist_ok=True)
    T = len(SYN_UNITS)
    with open(feat / "token_meta.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tok", "group", "unit", "region", "window", "pool"])
        wr.writerows([i, *g] for i, g in enumerate(SYN_UNITS))
    rng = np.random.default_rng(0)
    rows, trows = {"train": [], "val": []}, {"train": [], "val": []}
    for i in range(n):
        cid = f"SYN{i:03d}"
        mask = rng.random(T) > 0.2
        if i == 0:
            mask[:] = False                                   # a study with no unit at all must not give NaN
        np.savez(feat / "main" / f"{cid}.npz", tok=(rng.standard_normal((T, 448)) * mask[:, None]).astype(np.float16),
                 mask=mask)
        sp = "train" if i < n - 2 else "val"
        tgt = {"case_id": cid, "split": sp,
               "diseases": {d: {"status": st, "sentence_clean": next(l for l, ds in SYN_LINES if d in ds),
                                "attributes": at} for d, (st, at) in SYN_DIS.items()},
               "target_lines": [{"clean": l, "diseases": ds} for l, ds in SYN_LINES]}
        trows[sp].append(tgt)
        probs = {d: float(rng.random()) for d in DISEASES}
        meas = {"abd_eq_rmax": "21.5", "sp_cc_main_cm": "10.2", "sp_ml": "180"}
        calib = {"aaa_cm": {"col": "abd_eq_rmax", "a": 1.1, "b": -0.3}, "spleen_cm": {"col": "sp_cc_main_cm", "a": 0.7, "b": 4.9}}
        t, tds = make_target(tgt)
        rows[sp].append({"case_id": cid, "split": sp, "feature_path": str(feat / "main" / f"{cid}.npz"),
                         "prompt": make_prompt({"phase": "portal_venous", "manufacturer": "GE MEDICAL SYSTEMS",
                                                "manufacturermodelname": "Discovery CT750 HD"}, probs, meas, calib),
                         "target": t, "target_diseases": tds, "attr_terms": attr_terms(tgt)})
    data = work / "data"
    data.mkdir(exist_ok=True)
    for sp, rr in rows.items():
        with open(data / f"{sp}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rr)
        with open(work / f"targets_{sp}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in trows[sp])
    return feat, data


def write_cue_inputs(work, data, n_pool=400):
    """synthetic labels_disease14-style file + OOF score csv + thresholds (no report text) -> build_cue_gt.py CLI.
    The smoke cases get random categories (incl. unclear); n_pool extra train-only ids just fill the score pools."""
    rng = np.random.default_rng(1)
    cats = ["yes", "probably_yes", "no", "probably_no", "not_mentioned", "unclear"]
    rows = [json.loads(l) for sp in ("train", "val") for l in open(data / f"{sp}.jsonl")]
    ids = [(r["case_id"], r["split"], parse_cue(r["prompt"])[0]) for r in rows]
    for i in range(n_pool):
        ids.append((f"POOL{i:04d}", "train", None))
    lab, sc = [], []
    for cid, sp, cue in ids:
        ans = {}
        for d in DISEASES:
            c = cats[int(rng.choice(6, p=[.15, .05, .2, .05, .5, .05]))]
            ans[d] = {"category": c, "detail": {}, "quotes": []}
        lab.append({"study_id": cid, "split": sp, "answer": ans})
        if cue is None:
            cue = {d: (float(rng.uniform(.5, 1)) if ans[d]["category"] in ("yes", "probably_yes") else float(rng.uniform(0, .5)))
                   for d in DISEASES if ans[d]["category"] != "unclear"}
        sc.append({"study_id": cid, "split": sp, **{d: (f"{cue[d]:.5f}" if d in cue else "") for d in DISEASES}})
    with open(work / "labels_syn.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in lab)
    with open(work / "scores_syn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["study_id", "split", *DISEASES])
        w.writeheader()
        w.writerows(sc)
    with open(work / "thresholds_syn.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["disease", "t_f1"])
        w.writerows([d, 0.5] for d in DISEASES)
    subprocess.run([sys.executable, str(HERE / "build_cue_gt.py"), "--labels", str(work / "labels_syn.jsonl"),
                    "--scores", str(work / "scores_syn.csv"), "--thresholds", str(work / "thresholds_syn.csv"),
                    "--out", str(work / "cue")], check=True, capture_output=True)
    return work / "cue" / "cue_gt.json"


def cue_checks(work, data, tk, cue_path):
    """--cue-source / --meas-dropout / error-cue flips; returns a dict of results (asserts on failure)."""
    import random
    out = {}
    z = json.load(open(cue_path))
    assert all(json.loads(l)["case_id"] not in z["labels_train"] for l in open(data / "val.jsonl")), "val label leaked"
    # a. the classifier line re-rendered from its parsed values is identical, character for character
    for sp in ("train", "val"):
        for l in open(data / f"{sp}.jsonl"):
            p = json.loads(l)["prompt"]
            cue, i = parse_cue(p)
            assert p.split("\n")[i] == render_cue_line(cue)
    out["cue_line_roundtrip"] = "identical"
    # b. switches off -> identical items; no cue_info key
    d0 = M4Dataset(data / "train.jsonl", tk, True, 1024, backbone="tiny_random")
    d1 = M4Dataset(data / "train.jsonl", tk, True, 1024, backbone="tiny_random", cue_gt=None, meas_dropout=0.0)
    for i in range(len(d0)):
        a_, b_ = d0[i], d1[i]
        assert a_["input_ids"] == b_["input_ids"] and a_["labels"] == b_["labels"] and torch.equal(a_["tok"], b_["tok"])
        assert "cue_info" not in a_
    # c. gt_dropout: train cue from labels, on the label's side of the threshold, dropout rate ~ p
    for p_ in (0.3, 0.0, 1.0):
        dg = M4Dataset(data / "train.jsonl", tk, False, 1024, backbone="tiny_random",
                       cue_gt=CueGT(cue_path, dropout=p_), meas_dropout=0.5)
        random.seed(0)
        n_lab = n_drop = n_side = n_meas = n_meas_drop = 0
        for rep in range(300):
            for i in range(len(dg)):
                it = dg[i]
                ci, cid = it["cue_info"], it["case_id"]
                y = dict(zip(DISEASES, z["labels_train"][cid]))
                n_lab, n_drop = n_lab + ci["n_labelled"], n_drop + ci["n_dropped"]
                n_meas, n_meas_drop = n_meas + ci["n_meas"], n_meas_drop + ci["n_meas_dropped"]
                txt = tk.decode(it["prompt_ids"])
                shown, _ = parse_cue(txt.split("<|im_start|>user\n")[1])
                assert shown == {d: v for d, v in ci["probs"].items()}
                assert all(y[d] is not None for d in shown)            # unclear never shown
                n_side += sum((v >= z["thresholds"][d]) == bool(y[d]) for d, v in shown.items())
                assert n_side == n_lab - n_drop
        out[f"gt_dropout_p{p_}"] = {"labelled_cells": n_lab, "dropped": n_drop, "drop_rate": round(n_drop / n_lab, 4),
                                    "shown_on_label_side": f"{n_side}/{n_lab - n_drop}",
                                    "meas_drop_rate(p=0.5)": round(n_meas_drop / max(n_meas, 1), 4)}
        assert abs(n_drop / n_lab - p_) < 0.02
    # hard map
    dh = M4Dataset(data / "train.jsonl", tk, False, 1024, backbone="tiny_random",
                   cue_gt=CueGT(cue_path, dropout=0.0, gt_map="hard"))
    vals = set(dh[0]["cue_info"]["probs"].values())
    assert vals <= {0.95, 0.02}
    out["gt_hard_values"] = sorted(vals)
    # d. val rows never take gt cues: refused in training mode, ignored in inference mode
    try:
        M4Dataset(data / "val.jsonl", tk, False, 1024, backbone="tiny_random", cue_gt=CueGT(cue_path))
        raise AssertionError("val rows accepted gt cues")
    except SystemExit as e:
        out["val_with_gt_cue"] = f"refused: {e}"
    dv = M4Dataset(data / "val.jsonl", tk, False, 1024, backbone="tiny_random", cue_gt=CueGT(cue_path), train=False)
    dv0 = M4Dataset(data / "val.jsonl", tk, False, 1024, backbone="tiny_random", train=False)
    assert all(dv[i]["prompt_ids"] == dv0[i]["prompt_ids"] for i in range(len(dv)))
    out["val_inference_with_cue_gt"] = "ignored (identical prompts)"
    # e. flips: reproducible with the seed, monotone in the probability, counts and directions right
    cues = {json.loads(l)["case_id"]: parse_cue(json.loads(l)["prompt"])[0]
            for sp in ("train", "val") for l in open(data / f"{sp}.jsonl")}
    thr, pools = z["thresholds"], z["pools"]

    def run(p1, p2, seed):
        return {c: flip_cues(v, thr, pools, p1, p2, random.Random(f"{seed}:{c}"))[1] for c, v in cues.items()}
    f1, f2, f3 = run(0.5, 0.2, 7), run(0.5, 0.2, 7), run(0.5, 0.2, 8)
    assert f1 == f2 and f1 != f3
    big = run(1.0, 1.0, 7)
    assert all(set(f1[c]) <= set(big[c]) for c in cues)
    n_pos = sum(v >= thr[d] for c in cues for d, v in cues[c].items())
    n_neg = sum(v < thr[d] for c in cues for d, v in cues[c].items())
    assert sum(len(x) for x in big.values()) == n_pos + n_neg
    for c, fl in big.items():
        for d, (old, new, k) in fl.items():
            assert (k == "pos2neg") == (old >= thr[d]) and (new >= thr[d]) == (k == "neg2pos")
    out["flips"] = {"seed7_twice_identical": True, "seed8_differs": True,
                    "p(0.5,0.2)": {k: sum(v[2] == k for x in f1.values() for v in x.values()) for k in ("pos2neg", "neg2pos")},
                    "p(1,1) flips all scored cells": f"{sum(len(x) for x in big.values())} = {n_pos} pos + {n_neg} neg"}
    return out


def _resampler_inputs(ad, b):
    """(x, key_pad) exactly as ImageAdapter.forward builds them (for the isolation test)."""
    B = b["tok"].shape[0]
    x = ad.inp(b["tok"].float()) + ad.type_emb()[None]
    x = torch.cat([ad.null.expand(B, 1, -1), x], 1)
    kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool), ~b["mask"].bool()], 1)
    return x, kp


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
    sys.path.insert(0, str(HERE))
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

    # 9. experiment switches (all default off)
    sw = {}
    sw["swap_lr"] = swap_lr("Moderate left hydronephrosis. R hip. L1 and L-2, L 3. Right-sided. lt kidney")
    assert sw["swap_lr"] == "Moderate right hydronephrosis. L hip. L1 and L-2, L 3. Left-sided. rt kidney"
    perm = lr_permutation(meta)
    names = [f"{r['group']}/{r['unit']}/{r['region']}/{r['window']}/{r['pool']}" for r in meta]
    pairs = {names[i]: names[int(j)] for i, j in enumerate(perm) if int(j) != i}
    sw["lr_pairs_found"] = len(pairs) // 2
    assert pairs["organ3/kidney_left//baseline/mean"] == "organ3/kidney_right//baseline/mean"
    assert pairs["ru_hydronephrosis/k0/r0/w0/mean"] == "ru_hydronephrosis/k1/r0/w0/mean"
    assert "vert15/L1//bone/mean" not in pairs                      # vertebrae are never swapped
    # unit labels via the CLI
    subprocess.run([sys.executable, str(HERE / "build_unit_labels.py"), "--targets", str(work / "targets_train.jsonl"),
                    str(work / "targets_val.jsonl"), "--meta", str(feat / "token_meta.csv"), "--out",
                    str(work / "unit_labels")], check=True, capture_output=True)
    ul = {json.loads(l)["case_id"]: json.loads(l) for l in open(work / "unit_labels/unit_labels_train.jsonl")}
    ds_f = M4Dataset(data / "train.jsonl", tk, True, 1024, backbone="tiny_random", meta_rows=meta, unit_labels=ul,
                     lr_flip_prob=1.0, attr_weight=3.0)
    ds_n = M4Dataset(data / "train.jsonl", tk, True, 1024, backbone="tiny_random", meta_rows=meta, unit_labels=ul)
    it_f, it_n = ds_f[1], ds_n[1]
    assert it_f["flipped"] and not it_n["flipped"]
    assert torch.equal(it_f["tok"], it_n["tok"][perm]) and torch.equal(it_f["unit_labels"], it_n["unit_labels"][perm])
    kl = [i for i, r in enumerate(meta) if r["unit"] == "kidney_left"][0]
    kr = [i for i, r in enumerate(meta) if r["unit"] == "kidney_right"][0]
    hd = DISEASES.index("hydronephrosis")
    sw["unit_label_kidney_L_R_hydro (orig, flipped)"] = [[int(it_n["unit_labels"][kl, hd]), int(it_n["unit_labels"][kr, hd])],
                                                       [int(it_f["unit_labels"][kl, hd]), int(it_f["unit_labels"][kr, hd])]]
    assert sw["unit_label_kidney_L_R_hydro (orig, flipped)"] == [[1, 0], [0, 1]]
    l1 = [i for i, r in enumerate(meta) if r["group"] == "vert15" and r["unit"] == "L1"][0]
    r5 = [i for i, r in enumerate(meta) if r["group"] == "rib3" and r["unit"] == "rib_right_5"][0]
    fd = DISEASES.index("fracture")
    sw["unit_label_fracture_vertL1_ribR5"] = [int(it_n["unit_labels"][l1, fd]), int(it_n["unit_labels"][r5, fd])]
    assert sw["unit_label_fracture_vertL1_ribR5"] == [1, 1]
    tgt_f = tk.decode(it_f["input_ids"][len(it_f["prompt_ids"]):])
    sw["flipped_target_has"] = [x for x in ("Moderate right hydronephrosis", "left 5th rib", "L hip", "L1") if x in tgt_f]
    assert len(sw["flipped_target_has"]) == 4
    w = torch.tensor(it_n.get("loss_w", [])) if "loss_w" in it_n else None
    assert w is None                                                # baseline: no weights
    wf = torch.tensor(it_f["loss_w"])
    up = [tk.decode([it_f["input_ids"][k]]) for k in range(len(wf)) if wf[k] > 1]
    sw["attr_weighted_tokens"] = up
    assert any("mild" in x.lower() for x in up) and any("L" in x for x in up)
    # grouped resampler isolation: perturbing a liver token changes only the hepatobiliary latents
    ga = ImageAdapter(meta, llm.config.hidden_size, dim=32, depth=2, heads=4, groups=True).eval()
    bt2 = collate([ds_n[1], ds_n[2]], tk.pad_token_id)
    with torch.no_grad():
        z0 = ga.out_ln(ga.layers[1](ga.layers[0](ga.latents[None].expand(2, -1, -1), *_resampler_inputs(ga, bt2), ga.grp_mask),
                       *_resampler_inputs(ga, bt2), ga.grp_mask))
        bt3 = dict(bt2)
        bt3["tok"] = bt2["tok"].clone()
        liv = [i for i, r in enumerate(meta) if r["unit"] == "liver"]
        bt3["tok"][:, liv] += 5.0
        bt3["mask"] = bt2["mask"].clone()
        bt3["mask"][:, liv] = True
        z1 = ga.out_ln(ga.layers[1](ga.layers[0](ga.latents[None].expand(2, -1, -1), *_resampler_inputs(ga, bt3), ga.grp_mask),
                       *_resampler_inputs(ga, bt3), ga.grp_mask))
    changed = ((z1 - z0).abs().amax((0, 2)) > 1e-6).tolist()
    n_hb = RES_GROUPS[0][1]
    sw["grouped_changed_latents"] = [i for i, c in enumerate(changed) if c]
    assert all(changed[:n_hb]) and not any(changed[n_hb:])
    # aux unit loss gradients
    aa = ImageAdapter(meta, llm.config.hidden_size, dim=32, depth=2, heads=4, aux_unit=True, groups=True)
    bt4 = collate([ds_n[i] for i in range(len(ds_n))], tk.pad_token_id)
    o4 = ReportGenModel(llm, aa, img_id)(bt4["input_ids"], bt4["attention_mask"], bt4["labels"], bt4["tok"],
                                         bt4["mask"], unit_labels=bt4["unit_labels"], aux_weight=0.5)
    o4.loss.backward()
    sw["aux"] = {"lm": round(o4.lm_loss.item(), 3), "aux": round(o4.aux_loss.item(), 3),
                 "aux_head_grad": sum(p.grad.abs().sum().item() for p in aa.aux_head.parameters())}
    assert sw["aux"]["aux_head_grad"] > 0
    res["switches"] = sw

    # 10. cue source switches + error-cue test helpers
    cue_path = write_cue_inputs(work, data)
    res["cue"] = cue_checks(work, data, tk, cue_path)

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
             "--max-new-tokens", "8", "--batch", "2", "--num-workers", "0"],
            [py, str(HERE / "infer.py"), "--llm", str(plain), "--run", str(work / "runs/text/final"), "--text-only",
             "--data", str(data / "val.jsonl"), "--out", str(work / "gen_val_text.jsonl"), "--device", "cpu",
             "--dtype", "fp32", "--max-new-tokens", "8", "--batch", "2", "--num-workers", "0"],
            [py, str(HERE / "train.py"), "--stage", "1", "--out", str(work / "runs/sw_s1"), *common,
             "--aux-unit-loss", "0.5", "--unit-labels-dir", str(work / "unit_labels"), "--lr-flip-prob", "0.5",
             "--attr-token-weight", "3", "--resampler-groups"],
            [py, str(HERE / "train.py"), "--stage", "2", "--init-adapter", str(work / "runs/sw_s1/final/adapter.pt"),
             "--out", str(work / "runs/sw_s2"), *common, "--aux-unit-loss", "0.5", "--unit-labels-dir",
             str(work / "unit_labels"), "--lr-flip-prob", "0.5", "--attr-token-weight", "3", "--resampler-groups"],
            [py, str(HERE / "infer.py"), "--llm", str(plain), "--run", str(work / "runs/sw_s2/final"), "--data",
             str(data / "val.jsonl"), "--out", str(work / "gen_val_sw.jsonl"), "--device", "cpu", "--dtype", "fp32",
             "--max-new-tokens", "8", "--batch", "2", "--num-workers", "0"],
            [py, str(HERE / "train.py"), "--stage", "2", "--text-only", "--out", str(work / "runs/text_gt"), *common,
             "--cue-source", "gt_dropout", "--cue-gt", str(cue_path), "--meas-dropout", "0.2"],
            [py, str(HERE / "train.py"), "--stage", "2", "--init-adapter", str(work / "runs/s1/final/adapter.pt"),
             "--out", str(work / "runs/s2_gt"), *common, "--cue-source", "gt_dropout", "--cue-gt", str(cue_path)],
            [py, str(HERE / "infer.py"), "--llm", str(plain), "--run", str(work / "runs/s2_gt/final"), "--data",
             str(data / "val.jsonl"), "--out", str(work / "gen_val_gt_flip.jsonl"), "--device", "cpu", "--dtype",
             "fp32", "--max-new-tokens", "8", "--batch", "2", "--num-workers", "0", "--cue-gt", str(cue_path),
             "--cue-flip-pos2neg", "1.0", "--cue-flip-neg2pos", "0.3"]]
    for c in cmds:
        p = subprocess.run(c, env=env, capture_output=True, text=True)
        tag = Path(c[1]).stem + ":" + c[c.index("--out") + 1].split("/")[-1]
        res[tag] = "ok" if p.returncode == 0 else ("FAIL " + p.stderr[-1500:])
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-3000:])
            break
    if (work / "gen_val_gt_flip.cues.jsonl").exists():
        cl = [json.loads(l) for l in open(work / "gen_val_gt_flip.cues.jsonl")]
        res["infer_cue_log"] = {"cases": len(cl), "flips": sum(len(r["flips"]) for r in cl),
                                "meta": json.loads((work / "gen_val_gt_flip.cues.meta.json").read_text())["n_flips"]}
        man = json.loads((work / "runs/s2_gt/final/manifest.json").read_text())
        res["gt_run_manifest"] = {k: man["args"][k] for k in ("cue_source", "cue_dropout", "cue_gt_map")}
    res["saved_files"] = sorted(str(x.relative_to(work)) for x in (work / "runs").rglob("*") if x.is_file())[:40] \
        if (work / "runs").exists() else []
    print(json.dumps(res, indent=1, default=str))
    (work / "smoke_result.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
