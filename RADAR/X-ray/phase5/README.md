# Phase 5: 組裝完整 RadarPretrain，跑通驗證 + 全量訓練

見 `RADAR/X-ray/PLAN.md` 的整體規劃。先做「跑通」小規模驗證，使用者確認後接著跑了全量 10 epoch 訓練(結果見下方「全量訓練結果」一節)。zero-shot AUC 評估留待後續。

## 組裝方式

`train.py` **不透過 `RadarPretrain.from_config()`/LAVIS registry**，直接呼叫建構子：

```python
image_encoder = VisionBranch()  # 自動撿 Phase 3 的 checkpoint_unet_xray.pth 當暖身
text_encoder = XBertEncoder.from_config({"med_config_path": "unused"}, from_pretrained=True)
model = RadarPretrain(
    image_encoder=image_encoder, text_encoder=text_encoder, text_decoder=None,
    queue_size=0, alpha=0.4, embed_dim=256, momentum=0.995,
    tie_enc_dec_weights=False, max_txt_len=512, radar_plus=True,
)
```

跳過 `from_config()` 預設 `radar_ft=True` 會去載入的 `checkpoint_radar_pretrain.pth`——那是 CT 自己訓練好的 RADAR checkpoint，器官數/vision encoder 都跟 X-ray 不一樣，不適用也不該借用。

`samples['iters']`(RADAR+ 決定這個 step 做 whole-image 還是 anatomy-wise ITC)由訓練迴圈自己用累計 step 數塞進去，不是 dataset 的責任(`radar_pretrain.py:153`)。

## `bert-base-uncased` 用 symlink，不用重新下載

CT track 共用的 checkpoint 目錄 `/datadrive/VLM/RADAR/ckpt/bert-base-uncased` 已經存在，`RADAR/X-ray/ckpt/bert-base-uncased` 建 symlink 指過去即可。`XBertEncoder.from_config()`(`lavis/models/med.py:1397-1407`)跟 `init_tokenizer()`(`radar_base.py:26`)的路徑都是寫死的相對路徑 `../ckpt/bert-base-uncased`，跟腳本所在目錄同一層就能自動抓到。

## 兩個環境相容性問題(都在 `train.py` 裡處理，vendored 的 `lavis/` 程式碼本身沒動)

1. **繞開 `lavis/__init__.py` 的重依賴鏈**：跟 `pretrain_segmentation.py`/`phase4/dataset.py` 一樣的問題(`decord`/`fairscale`...)。這次因為 `lavis/models/med.py` 內部是用套件路徑 import(`from lavis.common.utils import ...`)，沒辦法完全繞過 `lavis` 這個套件本身，改用「塞一個假的 `lavis` module 進 `sys.modules`、只設定 `__path__`」的技巧，跳過 `lavis/__init__.py` 真正執行，但子模組還是能正常解析。唯一例外是 `registry.register_path("library_root", ...)`(`get_abs_path` 需要用到)，這行手動補回去。
2. **`transformers` 版本**：這份 vendored `lavis/models/med.py` 是配 `transformers==4.25` 寫的(`RADAR/CT/requirements.txt` 本來就釘死這個版本)，這台機器原本裝的是新很多的 5.17.0，`BertModel` 子類化的內部機制對不上(`apply_chunking_to_forward`/`find_pruneable_heads_and_indices` 搬家、`PreTrainedModel._finalize_model_loading` 內部邏輯改了)。裝回 `transformers==4.25` 後問題全部消失，不需要任何相容層。裝的時候有確認共用環境的 `torch`/CUDA 沒被拖累(之前裝 `transformers`/`fairscale` 有次不小心把 torch 降版過，這次全程盯著)。

## 跑通結果

`python3 train.py --limit 500 --steps 100 --batch-size 8`(NVIDIA L40S)：228.2M 參數的完整模型，100 步內三個 loss 都是有限數值、沒有爆炸/NaN：

| | first(平均) | last(平均) |
|---|---|---|
| loss_seg | -0.725 | -0.761 |
| loss_itc[whole] | 2.126 | 2.079 |
| loss_itc[anatomy]（僅非零 step） | 2.098 | 0.833 |

`loss_seg` 變化不大是預期中的——Phase 3 暖身過的 checkpoint 已經有 ~0.97 的 Dice，沒有太多進步空間。`loss_itc` 兩個分支都有下降趨勢，anatomy-wise 那支下降比較明顯，但因為只有小樣本(500 張、100 步)+ 小 batch(8)，這個統計本身雜訊很大，不要過度解讀成「已經收斂」，只當作「有在學、方向對」的訊號。

**一個真的靠實測抓到的介面 bug**：`radar_pretrain.py:374` 對 `seg_label` 做 `F.interpolate(mode='nearest')` 沒有轉型 float，CT 版沒踩到是因為 MONAI 讀 NIfTI label 時預設回傳 float32。已經在 `phase4/dataset.py` 把 `seg` 欄位從 int64 改成 float32 修正(數值 0.0/1.0/2.0/3.0 本身沒有精度損失，`dice.py` 內部用到的地方也是自己 `.long()` 轉型，全程相容)。

另外發現：`anatomy-wise ITC` 那個分支，如果某個 batch 裡剛好沒有器官同時滿足「完整(沒被邊界裁切)+ 至少一筆異常」的篩選條件，`radar_pretrain.py` 會回傳 Python `int 0` 而不是 tensor(`sum({}.values())`的行為)。`backward()` 本身沒事(`0 + tensor` 正常運作)，但外部 log 程式碼要對這個狀況防呆，不能無腦呼叫 `.item()`。batch size 越小這個情況越常見(這次 batch=8 大概一半 anatomy step 會遇到)。

## 全量訓練結果

`python3 -u train.py --epochs 10 --batch-size 8 --lr 1e-4`（NVIDIA L40S，90,180 張訓練圖，11,272 batch/epoch）。2026-09-20 11:13 開始，20:27 結束，約 9 小時 14 分（平均每 epoch ~55 分，比最初用小樣本估的 48 分鐘略慢，訓練過程中有觀察到偶發的短暫變慢，GPU 使用率確認持續在 90%+，非卡死，原因未深究，可能是特定 batch 的文字長度/器官數分佈造成的計算量差異）。

| epoch | loss | loss_seg | loss_itc |
|---|---|---|---|
| 1 | 1.620 | -0.933 | 2.554 |
| 2 | 1.487 | -0.958 | 2.445 |
| 3 | 1.481 | -0.960 | 2.440 |
| 4 | 1.471 | -0.960 | 2.432 |
| 5 | 1.422 | -0.961 | 2.383 |
| 6 | 1.450 | -0.961 | 2.411 |
| 7 | 1.435 | -0.962 | 2.397 |
| 8 | 1.426 | -0.962 | 2.388 |
| 9 | 1.425 | -0.962 | 2.387 |
| **10** | **1.410** | **-0.962** | **2.372** |

`loss_seg` 在 epoch 1→2 有大幅進步(Phase 3 暖身的 checkpoint 本來就不錯，這裡只是小幅微調)之後就穩定收在 -0.96 附近。`loss_itc` 中間幾個 epoch(6-7)有小幅回升，不是單調下降，但整體趨勢向下，最後一個 epoch 收在全程最低點，沒有過擬合或發散的跡象。

**最終 checkpoint**：`checkpoint_radar_pretrain_xray.pth`(913MB，含完整 `RadarPretrain` state_dict，不是只有 vision encoder)，本地 `RADAR/X-ray/ckpt/` 跟 `/datadrive/VLM/RADAR/X-ray/ckpt/` 都有備份(太大不進 git，已加進 `.gitignore`)。

**batch size 的教訓**：實測發現加大 batch size 反而更慢(batch=8 時 31.5 張/秒，batch=16 降到 22.5 張/秒，batch=32 直接 OOM)，原因是 `radar_pretrain.py` 的 `get_roi_features`(anatomy-wise ITC 分支)用 Python for 迴圈逐筆處理每個樣本的 cross-attention，不是向量化運算，batch 越大這段迴圈開銷越不成比例地增加。這是 vendored 模型程式碼本身的限制，要真正加速得改寫這段迴圈，這次沒有動它。

## 還沒做的(留給使用者決定要不要繼續)

- **zero-shot AUC 評估**：用 `Labels` 欄位算 AUC(仿照 CT 版 `calc_metrics.py`)，驗證學到的表徵有沒有用——這是唯一還沒驗證「訓練出來的模型到底有沒有用」的一步
- 如果 AUC 結果不理想，可以考慮：跑更多 epoch(目前看起來還沒完全收斂)、向量化 `get_roi_features` 加速訓練、或調整 learning rate schedule(這次全程用固定 `1e-4`，沒有用 `radar_config.yaml` 建議的 warmup+cosine schedule)
