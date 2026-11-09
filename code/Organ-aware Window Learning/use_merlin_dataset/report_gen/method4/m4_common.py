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
