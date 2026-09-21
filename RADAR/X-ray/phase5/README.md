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

## zero-shot AUC 評估結果與後續修正(2026-09-21)

`RADAR/X-ray/phase6/zero_shot_eval.py`(用 `Labels` 欄位仿照 CT 版 `calc_metrics.py` 算 AUC，細節見該檔案 README)在上面這個 checkpoint 上跑出來 **AvgAUC ≈ 0.51**，幾乎等於隨機猜測。診斷過程跟修正記錄如下。

### 診斷

先查了訓練集的 caption 分佈，發現兩個問題，嚴重程度不同：

1. **whole-image ITC 分支的 caption 重複率很高，而且沒有處理**：訓練集 90,180 張裡，36.7% 的 `whole_image_caption` 字串完全就是 `"normal."`（跟 PadChest 論文自己講的整體異常率 ~64.5% 一致，不是資料異常率低，而是「異常」跟「正常」各自的措辭都高度模板化、重複率高——論文 Fig.5 自己也講「前 1,000 個最常見句子佔全部句子 46%」）。原本的 in-batch InfoNCE loss 把對角線以外的每個樣本都當硬負例，即使兩張圖的 caption 逐字相同，也被迫拉開——這是在教模型學錯誤的訊號。
2. **anatomy-wise ITC 分支的異常樣本密度太低**：每個器官(left_lung/right_lung/heart)的異常比例只有 6~9%，`batch_size=8` 時平均每個 batch 只有 0.5~0.7 張「該器官異常且完整」的圖，訓練訊號極度稀疏。

深入看 `radar_pretrain.py` 才發現：**anatomy-wise 分支其實早就有第 1 點的修法**(`semantic_matrix_batch`/`semantic_matrix_batch1`，逐字比對 caption + 用 momentum text encoder 算異常 caption 間的語意相似度當 soft target)，是 CT 版原始程式碼帶著的設計，port 到 X-ray 時原樣繼承，只是先前沒注意到。**whole-image 分支則完全沒有這個機制**，純對角線 one-hot。所以修正方向明確：whole-image 補上跟 anatomy 分支一樣的機制，anatomy 分支則需要解決樣本密度問題。

### 修正一:whole-image 分支的 loss target(`phase3/lavis/models/radar_models/radar_pretrain.py`)

在 whole-image ITC 分支的 `sim_targets_whole` 構建那段，比照 anatomy 分支已有的做法，把「batch 內 caption 逐字相同」的樣本也標記成正例，而不是負例:

```python
sim_targets_whole = torch.zeros(sim_i2t_whole.size()).to(image.device)
sim_targets_whole.fill_diagonal_(1)

# 新增：batch 內 caption 逐字相同的樣本互相標記為正例(distributed 時先跨 rank gather)
cl_text_input_whole_arr = np.array(cl_text_input_whole)
... # gather across ranks if distributed
same_caption_whole = cl_text_input_whole_all[:, None] == cl_text_input_whole_all[None, :]
same_caption_whole = torch.from_numpy(same_caption_whole.astype(float)).to(image.device)
same_caption_whole.fill_diagonal_(0)

sim_targets_whole += same_caption_whole
sim_targets_whole /= sim_targets_whole.sum(1, keepdim=True)  # 重新正規化成機率分佈
```

只做了「逐字比對」這一半(沒有額外加 anatomy 分支那個 momentum text encoder 語意相似度的部分)，因為我們的 `"normal."` caption 本身永遠是同一個字串、沒有措辭變化，逐字比對已經完全覆蓋「同為正常」的情況，不像 CT 版可能有多種正常措辭需要語意相似度才能抓到。

### 修正二:anatomy-wise 分支的樣本密度(`phase4/dataset.py` + `phase5/train.py`)

在 `RadarXrayDataset` 加了 `sample_weights()`，對「任一器官異常」的記錄給權重 `ABNORMAL_OVERSAMPLE_WEIGHT = 6.0`(其餘給 1.0)；`train.py` 在 `--epochs` 全量訓練模式下改用 `WeightedRandomSampler` 取代 `shuffle=True`。權重 6.0 是算過的:訓練集任一器官異常率 12.7%(11,408/90,180)，套用權重後預期抽樣分佈變成 `(6×11408)/(6×11408+78772) ≈ 46%`。實測驗證(見下方)完全對得上。

**驗證結果**(純 CPU、不需 GPU，抽樣 90,180 次比對):

| | 自然比例 | 加權後 |
|---|---|---|
| 任一器官異常 | 12.7% | 46.4% |
| left_lung 異常 | 8.7% | 32.0% |
| right_lung 異常 | 9.2% | 33.8% |
| heart 異常 | 6.2% | 22.6% |

batch_size=8 下，每個器官平均可學的異常樣本數從 0.5~0.7 張提升到 1.8~2.7 張。

### 額外發現並修正的環境 bug(跟 loss/sampler 邏輯無關，但擋住了驗證)

在跑 smoke test 驗證上面兩個修正時，`train.py` 直接從零訓練(不像 `zero_shot_eval.py` 會載入已訓練好的 checkpoint)踩到一個新環境的 bug：`med.py` 的 `BertEmbeddings` 用 `torch.arange(...)` 建立的 `position_ids` buffer，在新版 `transformers` 的 `from_pretrained()` 快速初始化(meta device)機制下，沒有被正確寫入真正的數值，殘留垂圾記憶體(実測 `.max()` 是天文數字，不是 511)，導致 `position_embeddings` 查表 CUDA index-out-of-bounds 崩潰。`zero_shot_eval.py` 沒踩到是因為它載入訓練好的 checkpoint 時，state_dict 裡剛好也存了(舊環境下正確初始化的)這個 buffer，覆蓋掉了垂圾值，掩蓋了問題。修法是 `build_model()` 裡明確用正確的 `torch.arange(...)` 值重寫這個 buffer(兩個檔案都加了這個修正，`zero_shot_eval.py` 那邊是防禦性的，不再依賴 checkpoint 覆蓋的副作用)。

### 驗證方式

1. 小規模 `--limit 200 --steps 30`：確認 loss target 修正沒有 shape/邏輯錯誤，whole-image 分支 loss 數值在合理範圍(1.2~2.5，跟原本文件記錄的 2.1~2.5 相近)
2. 純 CPU 抽樣驗證(見上表)：確認 sampler 的實際抽樣分佈符合計算預期
3. 真正的 `--epochs 1` 路徑跑 ~560 batch(smoke test，非完整 epoch)：確認 sampler + loss 修正整合起來不會崩潰，loss 數值都是有限值——注意這裡看到的 `running_itc≈3.9` 比修正前的 `2.554` 高，這是預期中的，不是變差：soft target 讓 cross-entropy 的理論下限不再是 0，新舊數值不能直接比大小，要看訓練過程中會不會持續下降

### 目前狀態

兩個修正都驗證通過後，已重新啟動全量 10 epoch 訓練(`--epochs 10 --batch-size 8 --lr 1e-4`，同樣的參數)，預期一樣要跑 ~9 小時。訓練完成後要重新跑一次 `phase6/zero_shot_eval.py` 看 AUC 有沒有改善。

## 還沒做的(留給使用者決定要不要繼續)

- 上面這輪重新訓練跑完後，重跑 zero-shot AUC 評估，看兩個修正有沒有實際幫助
- 如果還是不理想，可以考慮：向量化 `get_roi_features` 才能真正加大 batch size(目前 batch=8 是效能瓶頸，不是刻意選擇)、調整 learning rate schedule(這次全程用固定 `1e-4`，沒有用 `radar_config.yaml` 建議的 warmup+cosine schedule)
