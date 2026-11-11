#!/usr/bin/env python3
"""Train method 4 (Merlin): Stage 1 = projector/resampler only (LLM frozen); Stage 2 = LoRA r16/a32 + projector.

  stage 1:  python train.py --stage 1 --llm <Lingshu-7B-llm> --data <jsonl dir> --meta <token_meta.csv> --out runs/m4_s1
  stage 2:  python train.py --stage 2 --llm ... --init-adapter runs/m4_s1/final/adapter.pt --out runs/m4_s2
  other backbone:  --backbone lingshu_i8b --llm <Lingshu-I-8B dir>   (see m4_common.BACKBONES)
  text-only ablation (no OWL embedding, same prompt minus the image block, same LoRA recipe; no stage 1):
            python train.py --stage 2 --text-only --llm ... --out runs/m4_text_s2
  cue source (train prompts only; val loss / inference always use the classifier line already in the jsonl):
            --cue-source oof (default, = the jsonl prompt: 5-fold OOF probabilities)
            --cue-source gt_dropout --cue-gt <build_cue_gt.py dir>/cue_gt.json [--cue-dropout 0.3] [--cue-gt-map sample]
            [--meas-dropout p]  (each measurement item -> "unknown" with probability p; works with either source)

Saved (every epoch and final/, plus best/ by val loss): adapter.pt (type embedding + resampler + projector),
lora/ (peft adapter, stage 2), manifest.json (args, LLM path, token_meta / data md5, sizes) -> reusable for
infer.py and later test runs. log.csv: step, epoch, train loss, val loss, lr, sec.
DataLoader: pin_memory=False always (A6000). Before using a GPU the script checks free memory and refuses when
another process (e.g. the CG vLLM) holds it, unless --force-gpu.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m4_common import BACKBONES, N_IMG_TOKENS  # noqa: E402
from m4_data import CueGT, M4Dataset, UnitCueGT, collate  # noqa: E402
from m4_model import ImageAdapter, ReportGenModel, read_token_meta  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest() if p and Path(p).exists() else None


def check_gpu(min_free_gb, force):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        used, total = (float(x) for x in out[0].split(","))
    except Exception:  # noqa: BLE001  (GB10 reports [N/A]); rely on --force-gpu
        print("[gpu] could not read nvidia-smi memory; pass --force-gpu after checking by hand", flush=True)
        if not force:
            sys.exit(2)
        return
    free = (total - used) / 1024
    print(f"[gpu] used {used / 1024:.1f} GB / {total / 1024:.1f} GB, free {free:.1f} GB", flush=True)
    if free < min_free_gb and not force:
        sys.exit(f"[gpu] only {free:.1f} GB free (< {min_free_gb}); another job (CG vLLM?) holds the GPU -- ask first")


def load_mapped_qwen2(path, cfg, dtype, key_mapping):
    """Build Qwen2ForCausalLM on the meta device and fill it from the VLM safetensors shards, renaming keys with
    key_mapping (regex -> replacement); vision keys are skipped. Independent of transformers' own VLM key handling."""
    import re
    import transformers
    from safetensors import safe_open
    with torch.device("meta"):
        m = transformers.Qwen2ForCausalLM(cfg)
    want = set(m.state_dict())
    files = sorted(Path(path).glob("*.safetensors"))
    sd = {}
    for fn in files:
        with safe_open(str(fn), framework="pt") as f:
            for k in f.keys():
                nk = k
                for pat, rep in (key_mapping or {}).items():
                    nk = re.sub(pat, rep, nk)
                if nk in want:
                    sd[nk] = f.get_tensor(k).to(dtype)
    if "lm_head.weight" not in sd and cfg.tie_word_embeddings:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    miss = sorted(want - set(sd) - {k for k in want if k.endswith("rotary_emb.inv_freq")})
    if miss:
        raise SystemExit(f"LLM keys missing after mapping ({len(miss)}), e.g. {miss[:3]}")
    m.load_state_dict(sd, strict=False, assign=True)
    m.model.rotary_emb = type(m.model.rotary_emb)(config=cfg)        # rebuild the (non-persistent) RoPE buffers
    m.tie_weights() if cfg.tie_word_embeddings else None
    return m.to(dtype)


def load_llm(path, dtype, device, backbone="lingshu7b"):
    """-> (tokenizer, causal LM) for a BACKBONES entry; only the language model is materialised."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    bb = BACKBONES[backbone]
    tk = AutoTokenizer.from_pretrained(path, trust_remote_code=bb["trust_remote_code"])
    if bb["loader"] == "causal_lm":
        llm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    elif bb["loader"] == "qwen2_from_vlm":
        vl = json.loads((Path(path) / "config.json").read_text())
        tc = dict(vl[bb["text_config_key"]]) if bb["text_config_key"] else \
            {k: v for k, v in vl.items() if k not in ("vision_config", "architectures", "model_type", "rope_scaling")}
        if (tc.get("rope_scaling") or {}).get("mrope_section"):
            tc["rope_scaling"] = None              # text-only positions: M-RoPE == 1-D RoPE
        for k in ("architectures", "model_type", "layer_types", "dtype"):
            tc.pop(k, None)
        llm = load_mapped_qwen2(path, transformers.Qwen2Config(**tc), dtype, bb["key_mapping"])
    elif bb["loader"] == "internvl_remote":
        from transformers import AutoModel
        full = AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        llm = full.language_model
        del full
    else:
        raise ValueError(bb["loader"])
    llm = llm.to(device)
    missing = [t for t in (bb["img_token"], bb["img_start"], bb["img_end"])
               if tk.convert_tokens_to_ids(t) in (None, tk.unk_token_id)]
    if missing:
        tk.add_special_tokens({"additional_special_tokens": missing})
    if len(tk) > llm.get_input_embeddings().weight.shape[0]:
        llm.resize_token_embeddings(len(tk))
    if tk.pad_token_id is None:
        tk.pad_token = "<|endoftext|>" if "<|endoftext|>" in tk.get_vocab() else tk.eos_token
    assert llm.config.hidden_size == bb["hidden"] or backbone == "tiny_random", "hidden size differs from BACKBONES"
    return tk, llm


def build(args, device):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tk, llm = load_llm(args.llm, dtype, device, args.backbone)
    llm.config.use_cache = False
    adapter = None
    if not args.text_only:
        meta = read_token_meta(args.meta)
        adapter = ImageAdapter(meta, llm.config.hidden_size, dim=args.res_dim, n_latents=N_IMG_TOKENS,
                               depth=args.res_depth, heads=args.res_heads,
                               aux_unit=getattr(args, "aux_unit_loss", 0) > 0,
                               groups=getattr(args, "resampler_groups", False)).to(device)
        if args.init_adapter:
            missing, unexpected = adapter.load_state_dict(torch.load(args.init_adapter, map_location=device), strict=False)
            if [k for k in missing if not k.startswith("aux_head")] or unexpected:
                sys.exit(f"adapter checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
            print("[init] adapter from", args.init_adapter, flush=True)
    llm.requires_grad_(False)
    if args.stage == 2:
        from peft import LoraConfig, get_peft_model
        llm = get_peft_model(llm, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    if args.grad_ckpt:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()
    img_id = tk.convert_tokens_to_ids(BACKBONES[args.backbone]["img_token"]) if adapter is not None else None
    return tk, ReportGenModel(llm, adapter, img_id)


def save(model, args, out_dir, extra):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model.adapter is not None:
        torch.save(model.adapter.state_dict(), out_dir / "adapter.pt")
    if args.stage == 2:
        model.llm.save_pretrained(out_dir / "lora")
    man = {"args": vars(args), "llm": args.llm, "backbone": BACKBONES[args.backbone], "token_meta_md5": md5(args.meta) if not args.text_only else None,
           "train_md5": md5(Path(args.data) / "train.jsonl"), "val_md5": md5(Path(args.data) / "val.jsonl"),
           "cue_gt_md5": md5(args.cue_gt) if getattr(args, "cue_gt", "") else None,
           "n_img_tokens": 0 if args.text_only else N_IMG_TOKENS, "time": time.strftime("%Y-%m-%d %H:%M"), **extra}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))


@torch.no_grad()
def val_loss(model, dl, device, amp):
    model.eval()
    tot = n = 0
    for b in dl:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device),
                        b["tok"].to(device) if "tok" in b else None, b["mask"].to(device) if "mask" in b else None)
        k = int((b["labels"][:, 1:] != -100).sum())
        tot, n = tot + out.lm_loss.item() * k, n + k                 # plain LM loss (comparable across switches)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--text-only", action="store_true", help="ablation: no OWL embedding / image tokens")
    ap.add_argument("--llm", required=True, help="LLM-only dir (prepare_llm.py output for Lingshu-7B) or Qwen dir")
    ap.add_argument("--data", required=True, help="dir with train.jsonl / val.jsonl (build_jsonl.py)")
    ap.add_argument("--meta", default="/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4/features/token_meta.csv")
    ap.add_argument("--feat-root", default="", help="override the features root in feature_path")
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps cap (debug)")
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N train rows")
    ap.add_argument("--val-n", type=int, default=400, help="val rows for the per-epoch val loss")
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.0, help="stage 1 adapter lr (default 1e-3) / stage 2 LoRA lr (2e-4)")
    ap.add_argument("--adapter-lr", type=float, default=1e-4, help="stage 2 adapter lr")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--res-dim", type=int, default=1024)
    ap.add_argument("--res-depth", type=int, default=2)
    ap.add_argument("--res-heads", type=int, default=8)
    ap.add_argument("--backbone", default="lingshu7b", choices=sorted(BACKBONES),
                    help="LLM backbone entry in m4_common.BACKBONES (loader, chat template, image token strings)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gb", type=float, default=30)
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=20261109)
    # ---- experiment switches (all off by default; the baseline is unchanged)
    ap.add_argument("--aux-unit-loss", type=float, default=0.0,
                    help="weight of the per-unit BCE head before the resampler (needs --unit-labels-dir)")
    ap.add_argument("--unit-labels-dir", default="", help="build_unit_labels.py output (unit_labels_train.jsonl)")
    ap.add_argument("--lr-flip-prob", type=float, default=0.0,
                    help="probability of swapping left/right paired units + left/right words in prompt and target")
    ap.add_argument("--attr-token-weight", type=float, default=1.0,
                    help="LM loss weight of attribute tokens (side, level, grade, sizes; from attr_terms)")
    ap.add_argument("--resampler-groups", action="store_true",
                    help="latents split by anatomical group, each attends only its own units (m4_common.RES_GROUPS)")
    ap.add_argument("--cue-source", default="oof", choices=["oof", "gt_dropout"],
                    help="train-time 14-class cue: oof = the jsonl prompt (5-fold OOF probabilities); gt_dropout = "
                         "rebuilt from the train report labels each time an item is read (teacher forcing), each "
                         "finding left out with probability --cue-dropout. Val rows always keep the classifier line")
    ap.add_argument("--cue-gt", default="", help="build_cue_gt.py output cue_gt.json (train labels, thresholds, pools)")
    ap.add_argument("--cue-dropout", type=float, default=0.3, help="gt_dropout: per-finding drop probability")
    ap.add_argument("--cue-gt-map", default="sample", choices=["sample", "hard"],
                    help="gt_dropout label -> number: sample = drawn from train OOF values on the label's side of the "
                         "method-1 threshold; hard = --cue-gt-hi / --cue-gt-lo")
    ap.add_argument("--cue-gt-unclear", default="omit", choices=["omit", "oof"],
                    help="gt_dropout, finding whose report label is unclear: omit (old; omission then marks unclear) or "
                         "oof (show the prompt's own train OOF classifier value, same dropout); needs complete scores")
    ap.add_argument("--cue-gt-hi", type=float, default=0.95)
    ap.add_argument("--cue-gt-lo", type=float, default=0.02)
    ap.add_argument("--meas-dropout", type=float, default=0.0,
                    help="train only: each measurement item (aorta, spleen, HU values) -> 'unknown' with this probability")
    ap.add_argument("--unit-cue-gt", default="",
                    help="improvement 2: improve/unit_probe.py unit_cue.json -> train rows get the unit-level probe line "
                         "rebuilt from report side / level labels (data must come from improve/build_unit_data.py)")
    ap.add_argument("--unit-cue-dropout", type=float, default=0.3, help="--unit-cue-gt: per-part drop probability")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    if args.stage == 1 and args.text_only:
        sys.exit("text-only has no projector: run stage 2 directly")
    if args.text_only and (args.aux_unit_loss or args.resampler_groups):
        sys.exit("--aux-unit-loss / --resampler-groups need image tokens")
    if args.aux_unit_loss and not args.unit_labels_dir:
        sys.exit("--aux-unit-loss needs --unit-labels-dir")
    if args.cue_source == "gt_dropout" and not args.cue_gt:
        sys.exit("--cue-source gt_dropout needs --cue-gt (build_cue_gt.py output)")
    args.lr = args.lr or (1e-3 if args.stage == 1 else 2e-4)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        check_gpu(args.min_free_gb, args.force_gpu)
    amp = device.type == "cuda" and args.dtype == "bf16"
    tk, model = build(args, device)

    use_img = not args.text_only
    feat_root = args.feat_root or None
    meta_rows = read_token_meta(args.meta) if use_img else None
    unit_labels = None
    if args.aux_unit_loss:
        unit_labels = {}
        with open(Path(args.unit_labels_dir) / "unit_labels_train.jsonl") as f:
            for line in f:
                r = json.loads(line)
                unit_labels[r["case_id"]] = r
    ds_tr = M4Dataset(Path(args.data) / "train.jsonl", tk, use_img, args.max_len, limit=args.train_limit,
                      feat_root=feat_root, backbone=args.backbone, meta_rows=meta_rows, unit_labels=unit_labels,
                      lr_flip_prob=args.lr_flip_prob, attr_weight=args.attr_token_weight,
                      cue_gt=CueGT(args.cue_gt, args.cue_dropout, args.cue_gt_map, args.cue_gt_hi, args.cue_gt_lo,
                                  args.cue_gt_unclear)
                      if args.cue_source == "gt_dropout" else None, meas_dropout=args.meas_dropout,
                      unit_cue=UnitCueGT(args.unit_cue_gt, args.unit_cue_dropout) if args.unit_cue_gt else None)
    ds_va = M4Dataset(Path(args.data) / "val.jsonl", tk, use_img, args.max_len, limit=args.val_n,
                      feat_root=feat_root, backbone=args.backbone)
    col = functools.partial(collate, pad_id=tk.pad_token_id, train=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.micro_bs, shuffle=True, collate_fn=col, num_workers=args.num_workers,
                       pin_memory=False, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.micro_bs, shuffle=False, collate_fn=col, num_workers=args.num_workers,
                       pin_memory=False)

    groups = []
    if model.adapter is not None:
        groups.append({"params": list(model.adapter.parameters()), "lr": args.lr if args.stage == 1 else args.adapter_lr})
    if args.stage == 2:
        groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": args.lr})
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"[model] trainable params {n_train / 1e6:.1f} M; train {len(ds_tr)}, val {len(ds_va)}", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = len(dl_tr) // args.grad_accum
    total = max(1, int(steps_per_epoch * args.epochs))
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(total * args.warmup))
    base = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.csv", "a")
    log.write("step,epoch,train_loss,val_loss,lr,sec\n")
    model.train()
    step, micro, best, t0, run, run_aux = 0, 0, float("inf"), time.time(), [], []
    epoch = 0
    while step < total:
        epoch += 1
        for b in dl_tr:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                o = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device),
                          b["tok"].to(device) if "tok" in b else None, b["mask"].to(device) if "mask" in b else None,
                          unit_labels=b["unit_labels"].to(device) if "unit_labels" in b else None,
                          loss_w=b["loss_w"].to(device) if "loss_w" in b else None, aux_weight=args.aux_unit_loss)
            (o.loss / args.grad_accum).backward()
            run.append(o.loss.item())
            if o.aux_loss is not None:
                run_aux.append(o.aux_loss.item())
            micro += 1
            if micro % args.grad_accum:
                continue
            set_lr(step)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total:
                aux_txt = f" aux {sum(run_aux) / len(run_aux):.4f}" if run_aux else ""
                print(f"step {step}/{total} ep {epoch} loss {sum(run) / len(run):.4f}{aux_txt} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
                run_aux = []
                log.write(f"{step},{epoch},{sum(run) / len(run):.5f},,{opt.param_groups[0]['lr']:.3e},{time.time() - t0:.0f}\n")
                log.flush()
                run = []
            if step >= total:
                break
        vl = val_loss(model, dl_va, device, amp)
        print(f"[epoch {epoch}] val loss {vl:.4f}", flush=True)
        log.write(f"{step},{epoch},,{vl:.5f},,{time.time() - t0:.0f}\n")
        log.flush()
        save(model, args, out / f"epoch{epoch}", {"step": step, "val_loss": vl})
        if vl < best:
            best = vl
            save(model, args, out / "best", {"step": step, "epoch": epoch, "val_loss": vl})
    save(model, args, out / "final", {"step": step, "epochs_run": epoch, "best_val_loss": best,
                                      "trainable_params": n_train})
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
