# pillar-finetune

Finetuning framework for Pillar medical imaging models.

## Table of Contents

- [Installation](#installation)
- [Finetuning Pillar0 on NLST Dataset](#fine-tuning-pillar0-on-nlst-dataset)
- [Data Preparation for Additional Datasets](#data-preparation-for-additional-datasets)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## Installation

### Using `uv` (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv add flash-attn==2.8.3 --no-build-isolation
source .venv/bin/activate
```

## Fine-tuning Pillar0 on NLST Dataset

```bash
bash scripts/run_nlst.sh
```

## Data Preparation for Additional Datasets

1. **Install `rad-vision-engine`**:

   ```bash
   git clone https://github.com/yalalab/rad-vision-engine ../rad-vision-engine
   cd ../rad-vision-engine && git checkout release
   cd ../pillar-finetune
   uv pip install -e ../rad-vision-engine
   ```

2. **Create a CSV file** with paths to DICOM series directories or NIfTI files.

   Example `series_paths.csv`:

   ```csv
   series_path
   /path/to/patient1/series1/
   /path/to/patient2/series2/
   /path/to/patient3/series3/
   /path/to/scan1.nii.gz
   /path/to/scan2.nii.gz
   ```

3. **Run chest CT processing**:

   ```bash
   vision-engine process \
       --config configs/ct_chest.yaml \
       --input-series-csv series_paths.csv \
       --output /output/dir \
       --workers 4
   ```

## Evaluation

1. **Verify installation**: Ensure the setup works with dummy data from NLST:

   ```bash
   OMP_NUM_THREADS=2 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MASTER_PORT=2300 bash scripts/test_setup.sh
   ```

   > **Note**: For pip-based installation, use `scripts/test_setup_pip.sh` instead.

   This should produce a test loss of **0.5797** on the example datapoint and export a CSV to `logs/csv/seed0/checkpoints/3/test.csv` containing the following (slight differences are acceptable):

   | accession | survival | time_at_event | y |
   | :-------- | :------- | :------------ | :- |
   | XXX | [-1.546875 -1.546875 -1.546875 -1.546875 -1.546875 -1.546875] | 3 | 1 |

2. **CSV dataset requirements**: To run inference using the CSV dataset, ensure your file contains the following columns:

   - **`accession`**: Unique sample identifier
   - **`image_paths`**: Either a JSON list string (recommended) like `["/path/to/series.1.0", "/path/to/another_series.1.0"]` or a delimited string using `|`, `;`, or `,`. Each entry is an RVE path loaded with `rve.load_sample()` and concatenated as channels.
   - **`split`**: One of `"train"`, `"dev"`, or `"test"`
   - **`y`** and **`time_at_event`**: Required for survival evaluation metrics

   Modify `config/csv_dataset.yaml` to point to your CSV file:

   ```yaml
   dataset:
     shared_dataset_kwargs:
       csv_path: examples/nlst.csv
   ```

3. **Run the ensembled models**:

   ```bash
   OMP_NUM_THREADS=2 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MASTER_PORT=2300 bash scripts/validate.sh
   ```

## Troubleshooting

1. **HuggingFace authentication**: Run `huggingface-cli login` for gated models like MedGemma and MedImageInsight
2. **Memory issues**: Reduce batch size or use more GPUs for memory-intensive models

# Citation
If you use this code in your research, please cite the following paper:

```
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  year    = {2025}
}
```
