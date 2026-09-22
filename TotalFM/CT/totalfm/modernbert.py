import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_modernbert_model(model_name: str):
    """Load a ModernBERT model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g.
            ``"Alibaba-NLP/gte-modernbert-base"``).

    Returns:
        Tuple of ``(config, tokenizer, model)``.
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return config, tokenizer, model
