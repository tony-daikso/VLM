"""Shared constants for Merlin method 4 (OWL tokens + scores -> LLM writes the doctor's 14-disease sentences)."""
import re

DISEASES = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis",
            "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis",
            "ascites", "free_air", "gallstones", "biliary_ductal_dilation", "fracture"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
# English names used in the prompt
NAME = {"hepatic_steatosis": "hepatic steatosis", "surgically_absent_gallbladder": "surgically absent gallbladder",
        "splenomegaly": "splenomegaly", "hydronephrosis": "hydronephrosis",
        "abdominal_aortic_aneurysm": "abdominal aortic aneurysm", "atherosclerosis": "atherosclerosis",
        "osteopenia": "osteopenia", "bowel_obstruction": "bowel obstruction", "appendicitis": "appendicitis",
        "ascites": "ascites", "free_air": "free air", "gallstones": "gallstones",
        "biliary_ductal_dilation": "biliary ductal dilation", "fracture": "fracture"}
# Merlin FINDINGS organ order (fallback ordering of target lines when agent B gives no report position)
RENDER_ORDER = ["hepatic_steatosis", "biliary_ductal_dilation", "gallstones", "surgically_absent_gallbladder",
                "splenomegaly", "hydronephrosis", "bowel_obstruction", "appendicitis", "ascites", "free_air",
                "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "fracture"]

# Target line format: "[tag1,tag2] <doctor's sentence>" -- the tag says which finding(s) the sentence covers, so the
# output maps back to {disease: sentence} for the evaluator; the sentence itself is the doctor's wording.
TAG_RE = re.compile(r"^\s*\[([a-z_,\s]+)\]\s*(.*?)\s*$")

N_IMG_TOKENS = 32

# ---- swappable LLM backbones (only the language model is used; the original vision encoders are never loaded:
# OWL tokens reach the LLM embedding space through our Perceiver resampler + projector).
#   loader "qwen2_from_vlm": Qwen2ForCausalLM.from_pretrained(<VLM dir>, config=<text config>, key_mapping=...)
#            -> the vision tower keys are simply unused; no weight copy, no trust_remote_code
#   loader "causal_lm":      AutoModelForCausalLM (plain LLM dir, e.g. prepare_llm.py output or a Qwen text model)
#   loader "internvl_remote": AutoModel(trust_remote_code=True).language_model (old InternVL repos with auto_map)
#   chat "chatml": <|im_start|>system/user/assistant ... <|im_end|>  (Lingshu-7B, Lingshu-I-8B and Qwen all use it)
#   chat "chatml_qwen3_nothink": same, assistant turn opened with an empty <think></think> block (Qwen3 family)
BACKBONES = {
    "lingshu7b": dict(  # main: Lingshu-7B = Qwen2.5-VL-7B medical; LLM keys already model.* / lm_head
        hf_id="lingshu-medical-mllm/Lingshu-7B", loader="qwen2_from_vlm", text_config_key=None,
        key_mapping=None, chat="chatml", img_token="<|image_pad|>", img_start="<|vision_start|>",
        img_end="<|vision_end|>", hidden=3584, params_b=7.6, trust_remote_code=False),
    "lingshu_i8b": dict(  # alternative: Lingshu-I-8B = InternVL3-8B medical (HF-native InternVL format, MIT)
        hf_id="lingshu-medical-mllm/Lingshu-I-8B", loader="qwen2_from_vlm", text_config_key="text_config",
        key_mapping={r"^language_model\.model\.": "model.", r"^language_model\.lm_head\.": "lm_head."},
        chat="chatml", img_token="<IMG_CONTEXT>", img_start="<img>", img_end="</img>", hidden=3584,
        params_b=7.6, trust_remote_code=False),
    "lingshu_i8b_remote": dict(  # same model if only a trust_remote_code (InternVLChatModel) copy is available
        hf_id="lingshu-medical-mllm/Lingshu-I-8B", loader="internvl_remote", text_config_key=None, key_mapping=None,
        chat="chatml", img_token="<IMG_CONTEXT>", img_start="<img>", img_end="</img>", hidden=3584,
        params_b=7.6, trust_remote_code=True),
    "qwen3_1p7b": dict(  # pipeline debugging only (A6000 ~/Desktop/Tony/models/Qwen3-1.7B)
        hf_id="Qwen/Qwen3-1.7B", loader="causal_lm", text_config_key=None, key_mapping=None,
        chat="chatml_qwen3_nothink", img_token="<|image_pad|>", img_start="<|vision_start|>",
        img_end="<|vision_end|>", hidden=2048, params_b=1.7, trust_remote_code=False),
    "tiny_random": dict(  # CPU smoke test only (random weights, Lingshu tokenizer)
        hf_id=None, loader="causal_lm", text_config_key=None, key_mapping=None, chat="chatml",
        img_token="<|image_pad|>", img_start="<|vision_start|>", img_end="<|vision_end|>", hidden=64,
        params_b=0.0, trust_remote_code=False),
}
SYSTEM = ("You are an abdominal radiologist at Stanford writing the FINDINGS of an emergency CT abdomen/pelvis "
          "report. Write in the department's usual wording.")


def parse_output(text):
    """generated tagged lines -> {disease: sentence}; one sentence may serve several findings; unknown tags dropped."""
    out = {}
    for line in (text or "").splitlines():
        m = TAG_RE.match(line)
        if not m:
            continue
        sent = m.group(2).strip()
        if not sent:
            continue
        for t in m.group(1).split(","):
            t = t.strip()
            if t in DISEASES and t not in out:
                out[t] = sent
    return out


# ---------------------------------------------------------------- experiment switches (all off by default)
# Left/right paired units in token_meta (group, unit_a, unit_b) -- used by --lr-flip-prob.
# region-unit indices checked in the mw_exp crop scripts / crops_info.jsonl (2026-11-09):
#   ru_hydronephrosis k0=right k1=left; ru_atherosclerosis k0 infrarenal k1 bifurcation k2 cia_right k3 cia_left;
#   ru_pelvis_fx k0 acetab_r k1 acetab_l k2 pubic k3 sacrum; ru_ascites k0 morison k1 perisplenic k2 pelvic (not mirror
#   images, never swapped); ru_free_air k0 subdiaph k1 antabd; ru_bowel k0 antrum k1 duodenum.
LR_PAIRS = [("organ3", "kidney_left", "kidney_right"), ("bone3", "ribs_left", "ribs_right"),
            ("bone3", "hip_left", "hip_right"), ("bone3", "femur_left", "femur_right"),
            ("ru_hydronephrosis", "k1", "k0"), ("ru_atherosclerosis", "k3", "k2"), ("ru_pelvis_fx", "k1", "k0")]
LR_PAIRS += [(g, f"rib_left_{n}", f"rib_right_{n}") for g in ("rib3", "rib15")
             for n in ["1_4", "5", "6", "7", "8", "9", "10", "11", "12"]]

_LR_WORD = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT",
            "lt": "rt", "rt": "lt", "Lt": "Rt", "Rt": "Lt"}
_LR_WORD_RE = re.compile(r"\b(left|right|Left|Right|LEFT|RIGHT|lt|rt|Lt|Rt)\b")
# a lone L / R (e.g. "R kidney"), never a vertebra (L1, L-2, L 3) or part of a word
_LR_ABBR_RE = re.compile(r"(?<![A-Za-z0-9\-])([LR])(?![A-Za-z0-9\-])(?!\s?-?\d)")


def swap_lr(text):
    text = _LR_WORD_RE.sub(lambda m: _LR_WORD[m.group(1)], text)
    return _LR_ABBR_RE.sub(lambda m: "R" if m.group(1) == "L" else "L", text)


# Anatomical groups for --resampler-groups: (name, n_latents, [(token group, unit or "*")]); latents sum to 32
RES_GROUPS = [
    ("hepatobiliary", 5, [("organ3", "liver"), ("organ3", "gallbladder"), ("organ3", "pancreas"), ("ru_gallstones", "*"),
                          ("ru_bile_duct", "*")]),
    ("kidney", 3, [("organ3", "kidney_left"), ("organ3", "kidney_right"), ("organ3", "adrenal"),
                   ("ru_hydronephrosis", "*")]),
    ("spleen", 2, [("organ3", "spleen")]),
    ("aorta", 3, [("organ3", "aorta"), ("ru_atherosclerosis", "*")]),
    ("bowel", 4, [("organ3", "stomach"), ("organ3", "small_bowel"), ("organ3", "colon"), ("ru_bowel", "*"),
                  ("ru_rlq", "*")]),
    ("peritoneum", 3, [("ru_ascites", "*"), ("ru_free_air", "*")]),
    ("spine", 5, [("organ3", "lumbar_spine"), ("bone3", "T8"), ("bone3", "T9"), ("bone3", "T10"), ("bone3", "T11"),
                  ("bone3", "T12"), ("bone3", "L1"), ("bone3", "L2"), ("bone3", "L3"), ("bone3", "L4"), ("bone3", "L5"),
                  ("bone3", "sacrum"), ("vert15", "*")]),
    ("ribs", 4, [("bone3", "ribs_left"), ("bone3", "ribs_right"), ("rib3", "*"), ("rib15", "*")]),
    ("pelvis", 3, [("bone3", "hip_left"), ("bone3", "hip_right"), ("bone3", "femur_left"), ("bone3", "femur_right"),
                   ("ru_pelvis_fx", "*")]),
]

# --attr-token-weight: attribute words searched in the target lines (B's attribute values + these synonyms)
GRADE_SYN = {"mild": ["mild", "minimal", "slight", "minor"], "moderate": ["moderate"],
             "severe": ["severe", "marked", "massive", "extensive", "dense", "heavy", "advanced", "profound"],
             "trace": ["trace", "traces", "tiny", "scant", "trivial", "minimal"], "small": ["small"],
             "large": ["large", "massive"], "SBO": ["small bowel", "SBO"], "LBO": ["large bowel", "colonic", "LBO"],
             "both": ["small and large bowel"], "left": ["left"], "right": ["right"], "bilateral": ["bilateral", "both"],
             "intrahepatic": ["intrahepatic"], "extrahepatic": ["extrahepatic", "common bile duct", "CBD", "common duct"]}
NUM_UNIT_RE = r"\d+(?:\.\d+)?\s*(?:cm|mm)\b"
LEVEL_RE_TXT = r"\b[CTLS]\s?-?\d{1,2}\b"

# ---------------------------------------------------------------- cue source (--cue-source; default "oof" = baseline)
# The 14-class cue is the prompt line written by build_jsonl.make_prompt:
#   "Image classifier probabilities (0-1): hepatic steatosis 0.12; ...; fracture 0.03."
# (a disease without a score is left out; no scores at all -> no line). The measurement line is
#   "Image measurements: <name> <value>; ..." with "<name> unknown" for a missing value.
# Rewrites below keep exactly this format, so train-time cues (OOF or report labels) and inference cues (classifier)
# look the same to the LLM.
CUE_PREFIX = "Image classifier probabilities (0-1): "
MEAS_PREFIX = "Image measurements: "
MEAS_NAMES = ["abdominal aortic max diameter", "spleen craniocaudal length", "spleen volume", "liver", "spleen",
              "aortoiliac wall calcification"]
_NAME2D = {v: k for k, v in NAME.items()}
OWL_POS, OWL_NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}


def owl_binary(category):
    """labels_disease14 category -> 1 / 0 / None, the binarisation the OWL classifiers were trained on (unclear -> None)."""
    return 1 if category in OWL_POS else 0 if category in OWL_NEG else None


def render_cue_line(probs):
    """{disease: prob} -> the classifier line, character for character as build_jsonl.make_prompt writes it."""
    return CUE_PREFIX + "; ".join(f"{NAME[d]} {probs[d]:.2f}" for d in DISEASES if probs.get(d) is not None) + "."


def parse_cue(prompt):
    """prompt -> ({disease: prob as written (2 decimals)}, line index) or ({}, None) when the prompt has no cue line."""
    lines = prompt.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(CUE_PREFIX):
            body = ln[len(CUE_PREFIX):].rstrip(".")
            out = {}
            for item in body.split("; "):
                if item.strip():
                    name, v = item.rsplit(" ", 1)
                    out[_NAME2D[name]] = float(v)
            return out, i
    return {}, None


def replace_cue(prompt, probs):
    """put a new classifier line in place of the old one (or right after the "<image>" line when the case had no
    scores, which is where make_prompt puts it); an empty dict removes the line (= the no-score format)."""
    lines = prompt.split("\n")
    _, i = parse_cue(prompt)
    keep = any(probs.get(d) is not None for d in DISEASES)
    if i is None:
        if keep:
            lines.insert(lines.index("<image>") + 1, render_cue_line(probs))
    elif keep:
        lines[i] = render_cue_line(probs)
    else:
        del lines[i]
    return "\n".join(lines)


def drop_measurements(prompt, p, rng):
    """each measurement item -> "<name> unknown" (the missing-value format) with probability p; returns
    (prompt, n_items, n_dropped). Items already unknown stay unknown."""
    lines = prompt.split("\n")
    n = k = 0
    for i, ln in enumerate(lines):
        if not ln.startswith(MEAS_PREFIX):
            continue
        items = ln[len(MEAS_PREFIX):].rstrip(".").split("; ")
        for j, it in enumerate(items):
            name = next((m for m in sorted(MEAS_NAMES, key=len, reverse=True) if it.startswith(m + " ")), None)
            if name is None:
                continue
            n += 1
            if rng.random() < p:
                items[j] = f"{name} unknown"
                k += 1
        lines[i] = MEAS_PREFIX + "; ".join(items) + "."
    return "\n".join(lines), n, k


def gt_cue_probs(y, pools, rng, dropout, gt_map="sample", hi=0.95, lo=0.02, unclear_cue=None):
    """--cue-source gt_dropout: report labels -> a cue dict in the OOF format.
    y: 14 values 1 / 0 / None (owl binarisation; None = unclear -> left out, like a missing OOF score).
    Each finding is dropped (left out) with probability `dropout`. gt_map:
      sample : positive -> a value drawn from the train OOF scores of label-positive cases at or above the method-1
               threshold (pools[d]["pos"]); negative -> drawn from label-negative OOF scores below it (pools[d]["neg"]).
               The number looks like a real classifier output but is always on the label's side of the threshold.
      hard   : positive -> hi, negative -> lo.
    unclear_cue: None (default) -> an unclear finding is left out (old behaviour; omission then flags "unclear");
      {disease: classifier value} (--cue-gt-unclear oof, 11-11 fix) -> an unclear finding shows the case's own
      classifier value from the prompt (train OOF), dropped with the same probability as labelled findings; left out
      only when the classifier has no score either.
    Returns (probs, n_labelled, n_dropped)."""
    out, n, k = {}, 0, 0
    for d, v in zip(DISEASES, y):
        if v is None:
            if unclear_cue is not None and unclear_cue.get(d) is not None:
                if rng.random() >= dropout:
                    out[d] = unclear_cue[d]
            continue
        n += 1
        if rng.random() < dropout:
            k += 1
            continue
        if gt_map == "sample":
            pool = pools[d]["pos" if v else "neg"]
            out[d] = pool[min(int(rng.random() * len(pool)), len(pool) - 1)]
        else:
            out[d] = hi if v else lo
    return out, n, k


def flip_cues(probs, thresholds, pools, p_pos2neg, p_neg2pos, rng):
    """error-cue test (val inference only): a cue at/above the threshold becomes a negative-looking value (drawn from
    pools[d]["neg"]) with probability p_pos2neg, a cue below it a positive-looking value (pools[d]["pos"]) with
    probability p_neg2pos. Two uniforms are drawn for every scored finding whatever the probabilities, so with the
    same seed a larger probability only adds flips. Returns (new_probs, {disease: [old, new, direction]})."""
    new, flips = dict(probs), {}
    for d in DISEASES:
        v = probs.get(d)
        if v is None:
            continue
        u, j = rng.random(), rng.random()
        pos = v >= thresholds[d]
        if (pos and u < p_pos2neg) or (not pos and u < p_neg2pos):
            pool = pools[d]["neg" if pos else "pos"]
            new[d] = pool[min(int(j * len(pool)), len(pool) - 1)]
            flips[d] = [v, new[d], "pos2neg" if pos else "neg2pos"]
    return new, flips


# ---- improvement 2 (agent F, 11-11): unit-level probe line (per-kidney hydronephrosis, per-vertebra T8-L5 fracture).
# Only present in prompts built by improve/build_unit_data.py; every function below is a no-op for other prompts.
UNIT_PREFIX = "Unit-level probe scores (0-1): "
UNIT_LEVELS = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]


def render_unit_line(kid, vert):
    """kid: [right, left] or None (part left out); vert: [10] or None. A missing value inside a part -> unknown /
    level left out. Returns "" when both parts are None."""
    parts = []
    if kid is not None:
        parts += [f"{s} kidney hydronephrosis " + ("unknown" if v is None else f"{v:.2f}")
                  for s, v in zip(("right", "left"), kid)]
    if vert is not None:
        lv = [f"{n} {v:.2f}" for n, v in zip(UNIT_LEVELS, vert) if v is not None]
        parts.append("vertebral fracture " + (", ".join(lv) if lv else "unknown"))
    return (UNIT_PREFIX + "; ".join(parts) + ".") if parts else ""


def replace_unit_line(prompt, line):
    """put `line` in place of the unit line (or right after the classifier line / "<image>" line when absent);
    an empty line removes it."""
    lines = prompt.split("\n")
    i = next((k for k, ln in enumerate(lines) if ln.startswith(UNIT_PREFIX)), None)
    if i is not None:
        if line:
            lines[i] = line
        else:
            del lines[i]
    elif line:
        j = next((k for k, ln in enumerate(lines) if ln.startswith(CUE_PREFIX)), lines.index("<image>"))
        lines.insert(j + 1, line)
    return "\n".join(lines)


def gt_unit_values(lab, oof, pools, thr, rng, dropout):
    """train-only gt cue for the unit line. lab: {"kid": [2 x 1/0/None], "vert": [10 x 1/0/None]} (report side / level,
    case-negative -> all 0); oof: {"kid": [...], "vert": [...]} train OOF probe scores (None = no tokens).
    Labelled unit -> value drawn from the train OOF pool on the label's side of the unit threshold; unlabelled unit ->
    its own OOF score. Each part (kidneys, vertebrae) is left out with probability `dropout`."""
    out = {}
    for part, key in (("kid", "kidney"), ("vert", "vertebra")):
        if rng.random() < dropout:
            out[part] = None
            continue
        vals = []
        for y, s in zip(lab[part], oof[part]):
            if s is None:
                vals.append(None)
            elif y is None:
                vals.append(s)
            else:
                pool = pools[key]["pos" if y else "neg"]
                vals.append(pool[min(int(rng.random() * len(pool)), len(pool) - 1)])
        out[part] = vals
    return out


# ---- A+ (agent H, 11-12): grade probe line (Pillar-0 option A+ stacks: hepatic steatosis / ascites / atherosclerosis
# "moderate or worse" probability). Only present in prompts built by improve/build_unit_data.py --grade-cue-gt; every
# function below is a no-op for other prompts. A grade item is shown only when its disease's classifier cue (the
# "Image classifier probabilities" line of the same prompt) is at or above the method-1 cue threshold, so train
# (gt-rebuilt cue) and inference (classifier cue) use the same rule.
GRADE_PREFIX = "Grade probe scores (0-1): "
GRADE_KEYS = [("hs_grade", "hepatic_steatosis", "hepatic steatosis moderate or severe"),
              ("asc_amount", "ascites", "ascites moderate or large"),
              ("ath_grade", "atherosclerosis", "atherosclerosis moderate or severe")]


# agent I (11-12): optional 4th grade item, hydronephrosis "moderate or severe" probability (improve/hygrade_probe.py).
# Only callers that pass keys=GRADE_KEYS_HY use it (improve/build_unit_data.py --hy-grade-scores; m4_data.UnitCueGT when
# the cue file's grade_cue_thr holds hydronephrosis). keys=None -> GRADE_KEYS: every existing arm is unchanged.
HY_GRADE_KEY = ("hy_grade", "hydronephrosis", "hydronephrosis moderate or severe")
GRADE_KEYS_HY = GRADE_KEYS + [HY_GRADE_KEY]


def grade_keys_shown(prompt, cue_thr, keys=None):
    """grade keys whose disease cue in `prompt` is >= cue_thr[disease] (a missing / dropped cue -> not shown)."""
    cue, _ = parse_cue(prompt)
    return [k for k, d, _ in (keys or GRADE_KEYS) if cue.get(d) is not None and cue[d] >= cue_thr[d]]


def render_grade_line(vals, keys=None):
    """{grade key: value or None} -> the grade line ("" when no key has a value), keys in GRADE_KEYS order."""
    items = [f"{txt} {vals[k]:.2f}" for k, _, txt in (keys or GRADE_KEYS) if vals.get(k) is not None]
    return (GRADE_PREFIX + "; ".join(items) + ".") if items else ""


def replace_grade_line(prompt, line):
    """put `line` in place of the grade line (or right after the unit line / classifier line / "<image>" line when
    absent); an empty line removes it."""
    lines = prompt.split("\n")
    i = next((k for k, ln in enumerate(lines) if ln.startswith(GRADE_PREFIX)), None)
    if i is not None:
        if line:
            lines[i] = line
        else:
            del lines[i]
    elif line:
        j = next((k for k, ln in enumerate(lines) if ln.startswith(UNIT_PREFIX)),
                 next((k for k, ln in enumerate(lines) if ln.startswith(CUE_PREFIX)), lines.index("<image>")))
        lines.insert(j + 1, line)
    return "\n".join(lines)


def grade_line_from_scores(prompt, scores, cue_thr, keys=None):
    """deterministic grade line (val / inference, and the stored train prompt): the case's own grade scores for the
    keys whose disease cue is positive in `prompt`."""
    return render_grade_line({k: scores.get(k) for k in grade_keys_shown(prompt, cue_thr, keys)}, keys)


def gt_grade_values(keys, lab, oof, pools, rng):
    """train-only gt cue for the grade line (keys = grade_keys_shown of the gt-rebuilt prompt). Grade label 1 / 0 ->
    value drawn from the train OOF pool on the label's side of the grade threshold; null (grade not written) -> the
    case's own OOF score. One rng draw per labelled shown key, none otherwise."""
    out = {}
    for k in keys:
        y, s = lab.get(k), oof.get(k)
        if y is None:
            out[k] = s
        else:
            pool = pools[k]["pos" if y else "neg"]
            out[k] = pool[min(int(rng.random() * len(pool)), len(pool) - 1)]
    return out


# ---- agent N (11-12): "attributes to state" line (arm T_unit_gt_pil_attr). Only present in prompts built by
# improve/build_attr_data.py; every function below is a no-op for other prompts. The line lists the attributes the
# report should state for a finding that is present. It carries switches only, never an attribute value (values come
# from the cue lines and the measurement line). Train: a switch of a target-positive finding is on iff the target
# sentence states that attribute; the switches of every other finding are drawn from the joint on / off patterns of
# the train-positive cases of that finding (m4_data.UnitCueGT, redrawn at every read), so a switch says nothing about
# whether the finding is present. Val / inference: every switch on (attr_line_all_on), independent of any report.
ATTR_PREFIX = "For findings that are present, also state: "
ATTR_ITEMS = [("hepatic_steatosis:grade", "hepatic_steatosis", "steatosis grade"),
              ("hydronephrosis:grade", "hydronephrosis", "hydronephrosis grade"),
              ("hydronephrosis:side", "hydronephrosis", "hydronephrosis side"),
              ("splenomegaly:length", "splenomegaly", "spleen length"),
              ("abdominal_aortic_aneurysm:diameter", "abdominal_aortic_aneurysm", "aortic aneurysm diameter"),
              ("atherosclerosis:grade", "atherosclerosis", "atherosclerosis grade"),
              ("bowel_obstruction:type", "bowel_obstruction", "obstructed bowel segment"),
              ("ascites:amount", "ascites", "ascites amount"),
              ("free_air:postoperative", "free_air", "whether free air is postoperative"),
              ("biliary_ductal_dilation:site", "biliary_ductal_dilation", "bile duct dilation site"),
              ("fracture:sites", "fracture", "fracture site"),
              ("fracture:levels", "fracture", "fractured vertebral level"),
              ("fracture:acuity", "fracture", "fracture acuity")]
ATTR_DISEASES = list(dict.fromkeys(d for _, d, _ in ATTR_ITEMS))


def render_attr_line(on):
    """on: collection of ATTR_ITEMS keys that are switched on -> the line (fixed item wording and order; an item that
    is off is left out; nothing on -> "... also state: none.")."""
    items = [txt for k, _, txt in ATTR_ITEMS if k in on]
    return ATTR_PREFIX + ("; ".join(items) if items else "none") + "."


def attr_line_all_on():
    return render_attr_line({k for k, _, _ in ATTR_ITEMS})


def parse_attr_line(prompt):
    """prompt -> set of keys switched on, or None when the prompt has no attribute line."""
    for ln in prompt.split("\n"):
        if ln.startswith(ATTR_PREFIX):
            got = set(ln[len(ATTR_PREFIX):].rstrip(".").split("; "))
            return {k for k, _, txt in ATTR_ITEMS if txt in got}
    return None


def replace_attr_line(prompt, line):
    """put `line` in place of the attribute line (or right after the measurement line; before the last line, the task
    instruction, when the prompt has no measurement line); an empty line removes it."""
    lines = prompt.split("\n")
    i = next((k for k, ln in enumerate(lines) if ln.startswith(ATTR_PREFIX)), None)
    if i is not None:
        if line:
            lines[i] = line
        else:
            del lines[i]
    elif line:
        j = next((k for k, ln in enumerate(lines) if ln.startswith(MEAS_PREFIX)), len(lines) - 2)
        lines.insert(j + 1, line)
    return "\n".join(lines)


def sample_attr_switches(state, patterns, rng):
    """train-only switches. state: {key: 1 / 0 / None} (1 / 0 = the finding is positive in the target and its sentence
    states / does not state the attribute; None = the finding is not positive in the target). patterns: {disease:
    {"keys": [...], "patterns": [[0/1, ...], ...], "cum": [cumulative counts]}} = the joint on / off patterns of the
    train-positive cases. A finding with a None key gets one pattern drawn from its table (one rng draw per such
    finding, none for a positive finding). Returns the set of keys that are on."""
    on = set()
    for d in ATTR_DISEASES:
        keys = [k for k, dd, _ in ATTR_ITEMS if dd == d]
        if all(state.get(k) is not None for k in keys):
            on.update(k for k in keys if state[k])
            continue
        t = patterns[d]
        assert t["keys"] == keys, d
        u = rng.random() * t["cum"][-1]
        j = next(i for i, c in enumerate(t["cum"]) if u < c)
        on.update(k for k, b in zip(keys, t["patterns"][j]) if b)
    return on
