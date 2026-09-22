"""Command-line interface for TotalFM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from totalfm.model import TotalFM


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="totalfm",
        description=(
            "TotalFM — organ-level CT embedding extraction and text encoding.\n\n"
            "Examples:\n"
            "  totalfm -i ct.nii.gz -s seg.nii.gz -o image_embs.npz\n"
            "  totalfm -t \"Liver with metastasis\" -o text_embs.npz\n"
            "  totalfm -t texts.json -o text_embs.npz"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--image",
        metavar="CT",
        help="Path to the CT volume NIfTI file (.nii.gz).",
    )
    parser.add_argument(
        "-s", "--seg",
        metavar="SEG",
        help=(
            "Path to the TotalSegmentator segmentation NIfTI file "
            "(required when --image is specified)."
        ),
    )
    parser.add_argument(
        "-t", "--text",
        metavar="TEXT",
        help=(
            "Text string to encode, or path to a JSON file with the format "
            '{"texts": ["text1", "text2", ...]}.'
        ),
    )
    parser.add_argument(
        "-o", "--output",
        metavar="OUTPUT",
        required=True,
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (e.g. 'cuda', 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a custom model checkpoint. Downloads from HuggingFace if omitted.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.image is None and args.text is None:
        parser.error("Specify either --image (-i) or --text (-t).")
    if args.image is not None and args.text is not None:
        parser.error("--image and --text are mutually exclusive.")
    if args.image is not None and args.seg is None:
        parser.error("--seg (-s) is required when --image (-i) is specified.")

    model = TotalFM(device=args.device, weights_path=args.weights)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        embeddings = model.encode_image(args.image, args.seg)
        if not embeddings:
            print(
                "Warning: no organs were detected in the segmentation.",
                file=sys.stderr,
            )
        np.savez_compressed(
            output_path,
            **{k: np.array(v, dtype=np.float32) for k, v in embeddings.items()},
        )
        print(f"Image embeddings saved to {output_path}")
        print(f"Organs encoded: {list(embeddings.keys())}")
    else:
        embeddings = model.encode_text(args.text)
        np.savez_compressed(
            output_path,
            **{k: np.array(v, dtype=np.float32) for k, v in embeddings.items()},
        )
        print(f"Text embeddings saved to {output_path}")


if __name__ == "__main__":
    main()
