"""Public TotalFM interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from totalfm._encoder import FoundationModel
from totalfm.image_proc import load_nifti_and_resample
from totalfm.organs import INDEX_TO_ORGAN, extract_organ_patch

_HF_REPO_ID = "jichi-labo/TotalFM"
_HF_FILENAME = "totalfm_en_checkpoint_best_loss.pt"
_WEIGHTS_DIR = Path(__file__).parent.parent / "modelweights"

_TARGET_HW = 192
_MAX_HW = 448
_TARGET_Z = 32


def _resolve_weights() -> Path:
    """Return the local weights path, downloading from HuggingFace if needed."""
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = _WEIGHTS_DIR / _HF_FILENAME
    if not weights_path.exists():
        print(f"Downloading model weights from {_HF_REPO_ID} ...")
        hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_HF_FILENAME,
            local_dir=str(_WEIGHTS_DIR),
        )
    return weights_path


def _parse_text_input(texts: Union[str, List[str], Path]) -> List[str]:
    """Normalise the ``texts`` argument to a plain list of strings."""
    if isinstance(texts, list):
        return texts
    path = Path(texts)
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)["texts"]
    return [str(texts)]


class TotalFM:
    """TotalFM inference model.

    Encodes CT organ patches and clinical text into a shared 768-dimensional
    embedding space trained with contrastive learning. Embeddings are
    L2-normalised, so cosine similarity reduces to a dot product.

    The model is loaded once at construction time and reused across all
    subsequent ``encode_image`` / ``encode_text`` calls. For batch processing
    of multiple CT volumes, create a single :class:`TotalFM` instance and call
    :meth:`encode_image` repeatedly.

    Args:
        device: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
            Defaults to CUDA when available, otherwise CPU.
        weights_path: Explicit path to the model checkpoint (``.pt`` file).
            When omitted, the checkpoint is downloaded automatically from
            HuggingFace (``jichi-labo/TotalFM``) to ``modelweights/`` inside
            the repository on the first call.

    Example::

        from totalfm import TotalFM

        model = TotalFM()
        image_embs = model.encode_image("ct.nii.gz", "seg.nii.gz")
        text_embs  = model.encode_text(["Liver with metastasis", "Normal liver"])
        sim = model.compute_similarity(
            image_embs["liver"],
            text_embs["Liver with metastasis"],
        )
        print(f"Similarity: {sim:.4f}")
    """

    def __init__(
        self,
        device: str | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        if weights_path is None:
            weights_path = _resolve_weights()

        self._model = FoundationModel()
        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
        if state_dict is None:
            raise KeyError(
                f"No model state dict found in checkpoint: {weights_path}"
            )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def encode_image(
        self,
        ct_path: str | Path,
        seg_path: str | Path,
    ) -> Dict[str, np.ndarray]:
        """Generate organ-wise image embeddings from a CT volume.

        Each organ's full Z extent is extracted from the segmentation mask,
        resampled to a fixed spatial resolution, and resized via trilinear
        interpolation to produce a single model-ready tensor per organ.

        Args:
            ct_path: Path to the CT NIfTI file (``.nii.gz``).
            seg_path: Path to the TotalSegmentator segmentation NIfTI file
                (run with the ``total`` task, version >= 2.10.0).

        Returns:
            Dictionary mapping organ name to a 768-dimensional L2-normalised
            embedding vector. Only organs present in the segmentation are
            included.

        Example::

            embeddings = model.encode_image("ct.nii.gz", "seg.nii.gz")
            liver_emb = embeddings["liver"]  # np.ndarray, shape (768,)
        """
        ct_array = load_nifti_and_resample(str(ct_path), interpolator="linear")
        seg_array = load_nifti_and_resample(
            str(seg_path), interpolator="nearest"
        ).astype(np.int32)

        embeddings: Dict[str, np.ndarray] = {}
        for organ_name in INDEX_TO_ORGAN.values():
            patch = extract_organ_patch(
                ct_array,
                seg_array,
                organ_name,
                target_hw=_TARGET_HW,
                max_hw=_MAX_HW,
                target_z=_TARGET_Z,
            )
            if patch is None:
                continue
            emb = self._model.encode_image(patch.unsqueeze(0))  # (1, 768)
            embeddings[organ_name] = emb.squeeze(0).numpy()

        return embeddings

    def encode_text(
        self,
        texts: Union[str, List[str], Path],
    ) -> Dict[str, np.ndarray]:
        """Generate text embeddings.

        Args:
            texts: One of:

                * a single string — embedded as-is;
                * a list of strings — each string is embedded;
                * a file path to a JSON file with the format
                  ``{"texts": ["text1", "text2", ...]}``.

        Returns:
            Dictionary mapping each input string to a 768-dimensional
            L2-normalised embedding vector.

        Example::

            embs = model.encode_text("Liver with metastasis")
            embs = model.encode_text(["Liver with metastasis", "Normal liver"])
            embs = model.encode_text("path/to/texts.json")
        """
        text_list = _parse_text_input(texts)
        embeddings_tensor = self._model.encode_text(text_list)  # (N, 768)
        return {
            text: embeddings_tensor[i].numpy()
            for i, text in enumerate(text_list)
        }

    def compute_similarity(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray,
    ) -> float:
        """Compute cosine similarity between an image and a text embedding.

        Because both embeddings are L2-normalised, this is equivalent to their
        dot product.

        Args:
            image_embedding: 1-D array of shape ``(768,)`` as returned by
                :meth:`encode_image`.
            text_embedding: 1-D array of shape ``(768,)`` as returned by
                :meth:`encode_text`.

        Returns:
            Scalar similarity score in the range ``[-1, 1]``.
        """
        return float(image_embedding @ text_embedding)
