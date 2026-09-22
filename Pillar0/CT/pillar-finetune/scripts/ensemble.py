import argparse
import sys
from os.path import dirname, realpath
import torch
from collections import defaultdict

sys.path.append(dirname(dirname(realpath(__file__))))

from pillar.metrics.survival import SurvivalMetric
from pillar.utils.parsing import load_config
import pandas as pd

# Example usage
# uv run scripts/ensemble.py --results \
# logs/localization/NLST_FT_Atlas_unimodal_all_lr1.e-5_warmup4_wd0_stages228_ps884_exp172_ep14_detr_sybil_cosine50-seed0/20251012_011649/checkpoints/3 \
# logs/localization/NLST_FT_Atlas_unimodal_all_lr1.e-5_warmup4_wd0_stages228_ps884_exp172_ep14_detr_sybil_cosine50-seed1/20251012_021512/checkpoints/3 \
# logs/localization/NLST_FT_Atlas_unimodal_all_lr1.e-5_warmup4_wd0_stages228_ps884_exp172_ep14_detr_sybil_cosine50-seed2/20251012_032651/checkpoints/3 \
# --config configs/nlst_detr_atlas.yaml


models = [
    "test - 1",
    "test - 2",
    "test - 3",
    "ensembled",
]


def to_cpu_state_dict(sd):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}


def ensemble_results(metric, results, dict):
    ensembled = results[0]
    ensembled_survival = []
    for result in results:
        ensembled_survival.append(result["survival"].cpu())
        logging_dict = metric(**result)

        for k, v in logging_dict.items():
            dict[k].append(v.item() if isinstance(v, torch.Tensor) else v)

    ensembled["survival"] = torch.mean(torch.stack(ensembled_survival), dim=0)
    return ensembled


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--results", type=str, nargs="+", required=True)
parser.add_argument("--dataset_info", type=str, required=False)
parser.add_argument("--output_path", type=str, required=False, default="ensemble.pth")
args = parser.parse_args()

overall_dict = defaultdict(list)

split = "test"
results = []
for result in args.results:
    result = torch.load(f"{result}/{split}.pth")
    results.append(result)

config = load_config(args.config)
if args.dataset_info:
    dataset_info = torch.load(args.dataset_info)
else:
    dataset_info = {}
metric = SurvivalMetric(
    config, dataset_info, split=split, logit_key="survival", target_key="y", censor_key="time_at_event"
)

ensembled_results = ensemble_results(metric, results, overall_dict)
logging_dict = metric(**ensembled_results)

ensembled_results.pop("logs")
torch.save(to_cpu_state_dict(ensembled_results), args.output_path)

for k, v in logging_dict.items():
    overall_dict[k].append(v.item() if isinstance(v, torch.Tensor) else v)


# Set options to display all rows and columns
pd.set_option("display.max_rows", None)  # Display all rows
pd.set_option("display.max_columns", None)  # Display all columns
pd.set_option("display.width", 1000)  # Adjust width for better readability if needed
pd.set_option("display.max_colwidth", None)  # Display full content of cells

print(pd.DataFrame(overall_dict, index=models).transpose())
