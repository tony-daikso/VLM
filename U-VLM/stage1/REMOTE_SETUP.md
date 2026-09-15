# Stage 1 訓練搬到遠端執行 — 設置步驟

寫於 2026-09-15。資料抽取（`data/processed_rex/`、`data/processed_hipas/`）跟訓練程式碼
（`U-VLM/stage1/` 底下 `dataset.py`/`model.py`/`losses.py`/`train.py`/`eval.py`）都已經在本機
寫完並個別測試過，還沒有跑過正式訓練。這份文件只講「怎麼把訓練這件事搬到遠端機器上跑」，
資料怎麼來的請看 [../../HANDOFF.md](../../HANDOFF.md)、架構設計看 [STAGE1_PLAN.md](STAGE1_PLAN.md)。

---

## 1. 需要搬過去的東西

| 路徑 | 大小 | 需不需要 |
|---|---|---|
| `data/processed_rex/` | ~4.5GB | ✅ 需要（訓練只會用到裡面的 `images_npy/`、`masks/`、`organ_masks/`、`manifest.jsonl`，其餘 `images/`（PNG，429M）、`logs*`、`manifest.csv`、`qa/`、`reports/`、`totalseg_pilot/` 是抽取過程的副產物，頻寬吃緊的話可以不搬） |
| `data/processed_hipas/` | ~289MB | ✅ 需要 |
| `U-VLM/stage1/` | 幾百 KB | ✅ 全部搬，含 `config.yaml`、`dataset.py`、`model.py`、`losses.py`、`train.py`、`eval.py`、`make_splits.py`、**`splits.json`**、`STAGE1_PLAN.md` |
| `data/hipas_raw/` | — | ❌ 不需要，HiPaS 已處理完成 |
| `scripts/` | — | ❌ 不需要，訓練不會重新抽取資料，也不會用到 TotalSegmentator/HuggingFace 下載 |
| `.venv` 或任何 conda env 資料夾 | — | ❌ **絕對不要搬**，見第 4 節 |

**`splits.json` 要跟著搬，不要在遠端重新跑 `make_splits.py`**：雖然是 fixed seed（42）理論上會重現一樣的結果，但直接複製現成檔案能保證 train/val 切法百分之百一致，才能拿遠端訓練結果跟本機的驗證結果做比較。

範例（假設遠端可以直接 ssh 連線）：
```bash
rsync -avz --progress \
  data/processed_rex data/processed_hipas U-VLM \
  user@remote-host:/path/to/VLM/
```
保持 `data/` 跟 `U-VLM/` 在同一層目錄結構就好（程式碼裡的路徑都是用 `dataset.py`/`train.py` 檔案位置往上兩層算出專案根目錄，不用改路徑）。

---

## 2. 遠端環境設置

Stage 1 訓練**只需要**這些套件，不需要 `TotalSegmentator`/`nibabel`/`huggingface_hub`（資料已經全部抽取完存成 `.npy`/`.npz`，`dataset.py`/`train.py` 不會再下載或跑 3D 推論）：

```bash
# 用 conda 或 python3 -m venv 都可以，重點是在遠端這台機器上「重新建立」
conda create -n VLM python=3.11 -y
conda activate VLM

# 有 NVIDIA GPU 的話，先去 https://pytorch.org/get-started/locally/ 選對應 CUDA 版本裝 torch/torchvision
# （例如 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121），
# 再把 requirements-train.txt 裡的 torch/torchvision 兩行刪掉後跑下面這行補裝剩下的套件；
# 沒有另外處理 CUDA 版本的話，直接跑這行就好：
pip install -r U-VLM/stage1/requirements-train.txt
```

確認 GPU 抓得到：
```bash
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"
```

`train.py` 的 `get_device()` 已經自動判斷 `mps` → `cuda` → `cpu`，不用改程式碼，遠端如果是 NVIDIA GPU 會自動走 `cuda`。

---

## 3. 快速驗證環境跟資料都對得上

搬完之後、開始訓練之前，先跑這段確認資料讀得到、筆數跟本機一致：

```bash
cd VLM
python3 -c "
import sys, yaml
sys.path.insert(0, 'U-VLM/stage1')
from dataset import Stage1Dataset

with open('U-VLM/stage1/config.yaml') as f:
    config = yaml.safe_load(f)

train_ds = Stage1Dataset(config, 'train', augment=True)
val_ds = Stage1Dataset(config, 'val', augment=False)
print('train:', len(train_ds), 'val:', len(val_ds))  # 應該是 train 2810 / val 482

sample = train_ds[0]
print('image shape:', sample['image'].shape, 'source:', sample['source'])
"
```

再跑一次 model forward 確認 GPU 真的有被用到（可以用 `nvidia-smi` 另開一個視窗看 util）：
```bash
python3 -c "
import sys, torch, yaml
sys.path.insert(0, 'U-VLM/stage1')
from model import Stage1Model
with open('U-VLM/stage1/config.yaml') as f:
    config = yaml.safe_load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Stage1Model(config).to(device)
x = torch.randn(2, 1, 512, 512, device=device)
out = model(x)
print({k: v.shape for k, v in out.items()})
"
```

---

## 4. 已知問題 / 踩過的坑（遠端上不用重踩）

1. **不要把 `.venv` 或 conda env 資料夾整包搬過去跨機器/跨使用者用**：這次資料抽取搬機器時就踩過——搬過去的 `.venv` 裡 pip/console script 的 shebang 寫死原本電腦的使用者路徑（`/Users/dufangchen/...`），加上 numpy 編譯的 `.so` 檔案跨機器複製後 macOS code signature 失效，整個環境直接壞掉、`import numpy` 都會噴 `ImportError`。環境一定要在目標機器上用 `conda create` / `python3 -m venv` **重新建立**，只搬資料跟程式碼。
2. **`config.yaml` 裡的路徑都是相對於專案根目錄算的**（`data/processed_rex_dir`、`checkpoint_dir` 等），複製時保持 `data/` 跟 `U-VLM/` 在同一層目錄結構即可，不用改任何路徑設定。
3. **Batch size 12 是為本機 Apple Silicon（M4, 統一記憶體）+ 512×512 保守設的**，遠端如果是 VRAM 更大的 NVIDIA GPU，可以考慮把 `config.yaml` 的 `train.batch_size` 調大（例如 24~32）加快訓練，只是還沒實測過更大 batch size 下的收斂穩不穩定。
4. **遠端訓練要用 `tmux`/`screen`，不是 macOS 的 `caffeinate`**：本機資料抽取批次跑很久時用 `caffeinate -i -w $PID` 防止筆電睡眠，遠端伺服器沒有這個問題，但 SSH 斷線會殺掉前景程序，要用 `tmux new -s stage1_train` 或 `nohup ... &` 讓訓練在背景繼續跑，不受斷線影響。

---

## 5. 開始訓練

```bash
cd VLM
tmux new -s stage1_train
conda activate VLM
python3 U-VLM/stage1/train.py
# Ctrl+B, D 離開 tmux（訓練繼續在背景跑），之後用 tmux attach -t stage1_train 接回去看
```

`train.py` 每個 epoch 結束會存 `U-VLM/stage1/checkpoints/last.pt`（可用 `--resume` 接續）跟
`best_encoder.pt` / `best_full.pt`（三個 head 的 val Dice 平均分數有進步才更新），同時把
`history.json`（每個 epoch 的 train/val 指標）寫進同一個資料夾，方便事後畫圖檢查訓練曲線。

跑完（或想先看目前結果）用 `eval.py` 拿到三個 head 的 val Dice 分布 + QA overlay 圖：
```bash
python3 U-VLM/stage1/eval.py --checkpoint U-VLM/stage1/checkpoints/best_full.pt
```
