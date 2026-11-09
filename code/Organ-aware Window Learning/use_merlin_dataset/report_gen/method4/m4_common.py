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
