"""
Encode the 90 labeled abdomen findings into raw Qwen3-Embedding-8B text embeddings
(last-token pooled, NOT normalized -- Pillar0's CacheTextEncoder projection + final
normalize is applied later in run_inference.py, matching how CustomTextCLIP.encode_text
normalizes the *projected* output, not the raw text tower output).

Usage: python3 encode_text_prompts.py
"""
import json
import torch
from transformers import AutoTokenizer, AutoModel

FINDING_MAP_PATH = "/datadrive/VLM/Pillar0/CT/finding_english_map.json"
QWEN_DIR = "/datadrive/VLM/Pillar0/CT/ckpt/Qwen3-Embedding-8B"
OUT_EMB_PATH = "/datadrive/VLM/Pillar0/CT/text_embeddings.pt"


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def main():
    finding_map = json.load(open(FINDING_MAP_PATH, encoding="utf-8"))
    keys = list(finding_map.keys())
    prompts = [f"{finding_map[k]['finding_en']} in {finding_map[k]['organ_en']}" for k in keys]
    print(f"Encoding {len(prompts)} finding prompts with Qwen3-Embedding-8B...")
    for k, p in list(zip(keys, prompts))[:5]:
        print(f"  {k} -> {p!r}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_DIR, torch_dtype=torch.bfloat16).cuda().eval()

    batch = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**batch)
        emb = last_token_pool(out.last_hidden_state, batch["attention_mask"])
    emb = emb.float().cpu()
    print("Raw text embedding shape:", emb.shape)

    torch.save({"keys": keys, "prompts": prompts, "embeddings": emb}, OUT_EMB_PATH)
    print(f"Saved to {OUT_EMB_PATH}")


if __name__ == "__main__":
    main()
