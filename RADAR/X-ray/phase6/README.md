# Phase 6: zero-shot 分類 AUC 評估

PLAN.md Phase 5 最後一項未驗證的任務:用 PadChest 的 `Labels` 欄位(實際上是 Phase 2 manifest 裡已經正規化好的 `labels_raw`)當 ground truth,驗證 Phase 5 訓練出來的 `checkpoint_radar_pretrain_xray.pth` 到底有沒有學到有用的表徵。仿照 CT track 的 `calc_metrics.py` + `infer_merlin_whole.py` + `infer_merlin_anatomy.py`。

## 目前狀態(2026-09-21):腳本已寫好,尚未執行過

`zero_shot_eval.py` 已經完成,設計如下(不需要修改任何 vendored 模型程式碼——`radar_pretrain.py` 已經帶有 CT 那邊留下來的 `forward_test_win()`/`prepare_text_feat()` 兩個 inference-only 方法,直接重用):

- **文字 prompt**:每個 finding 用 `("normal.", "<finding>.")` 這一組 pos/neg prompt,故意貼近 Phase 2 `build_region_captions.py` 實際產生的 caption 風格(全部小寫、標籤字串直接接句號、無發現時就是 `"normal."`)——這是模型訓練時真正見過的文字分布,不是隨便選的。
- **whole-image zero-shot**(18 個 finding,`ALL_WHOLE_FINDINGS`):用 `model.attention_whole`/`query_tokens_whole`/`vision_projs_whole` 算整張圖的 embedding,對每個 finding 算 cosine similarity → softmax([neg,pos]) → 取 positive 機率。
- **anatomy-wise zero-shot**(heart 4 個 + left_lung/right_lung 各 10 個 = 24 個 organ/finding pair):直接呼叫 `model.forward_test_win()`,用模型自己預測的 segmentation(不是 CheXmask ground truth mask)判斷該器官在這張圖裡是否完整/可見,可見才給分數,否則按照 `calc_metrics.py` 的慣例記為機率 0(不是跳過這張圖)。
- ground truth:每個 finding 是否出現在該圖片 Phase 2 manifest 的 `labels_raw` 裡(不分左右肺,跟 CT 版一樣,同一個 disease label 給所有相關器官的 test item 共用)。
- 驗證集:`data_split.json` 的 `val_image_ids`,再過濾 Phase1 `quality_ok`,共 **4,747 張**可用。

finding 清單怎麼選的、各自的驗證集正例數,見 `zero_shot_eval.py` 檔頭 docstring 跟頂部常數(`HEART_FINDINGS`/`LUNG_FINDINGS`/`WHOLE_ONLY_FINDINGS`)。

## 結果(2026-09-21,全量 4,747 張驗證集,`python3 -u zero_shot_eval.py`,約 1 分 36 秒)

**AvgAUC(all)= 0.5117 / AvgAUC_whole = 0.5082 / AvgAUC_anatomy = 0.5147** —— 幾乎等於隨機猜測(0.5)。

| whole-image 前 5 | AUC | anatomy-wise 前 5 | AUC |
|---|---|---|---|
| pacemaker | 0.5773 | left_lung pleural effusion | 0.5563 |
| sternotomy | 0.5520 | right_lung interstitial pattern | 0.5556 |
| vertebral degenerative changes | 0.5317 | left_lung copd signs | 0.5420 |
| hiatal hernia | 0.5304 | right_lung nodule | 0.5392 |
| pneumonia | 0.5234 | heart pacemaker | 0.5371 |

完整 44 個 finding/organ 的結果見 `../results/xray_zero_shot_aucs.json`,或直接重跑腳本看 stdout。

**結論**:目前這個 checkpoint(`checkpoint_radar_pretrain_xray.pth`,10 epoch,固定 LR 1e-4)在這個 zero-shot 分類協議下,沒有展現出有意義的判別能力——最高的 pacemaker(0.577)也只是略高於隨機,且是視覺上最容易辨識的一個發現(金屬裝置),其他大部分發現在 0.45~0.53 之間來回,沒有一致的正向訊號。這跟 Phase 5 訓練時觀察到的 `loss_itc` 持續下降(2.554→2.372)不矛盾——loss 下降代表模型在學習拉近訓練集裡「圖片-對應 caption」的距離,但這不保證學到的表徵能泛化成「輸入一個從未見過的固定 prompt,分辨出臨床發現的有無」這種 zero-shot 判別能力,兩者是不同的目標。

**可能原因(尚未驗證,只是排序後最可能的懷疑對象)**:
1. 訓練只有 10 epoch、LR 全程固定 1e-4,没有 warmup/cosine schedule(phase5/README.md 一開始就記錄過這是已知的簡化);CT 版原始 `radar_config.yaml` 建議用 warmup+cosine。
2. batch size 被迫用 8(因為 `get_roi_features` 的 for 迴圈瓶頸),對比學習的負例數量少,可能限制了學習效果。
3. `queue_size=0`(這次組裝時直接照抄 `radar_config.yaml` 的值,沒有另外驗證這個設定在小 batch 下是否足夠)。
4. Zero-shot prompt 本身很簡陋(單一 "normal." vs "<finding>." 一對一,沒有像 CT 版一樣用多個同義句做 prompt ensemble),可能低估了模型真實的判別力。

## 下一步(留給使用者決定要不要繼續投入)

- 先確認結果不是我這邊 pipeline 的 bug(例如檢查幾張圖的相似度分數分佈、確認 sanity check——這部分還沒做,如果要追根究底建議先做這個,而不是急著重新訓練)
- 如果排除 pipeline bug,再考慮:加 LR warmup+cosine schedule 重新訓練、向量化 `get_roi_features` 換更大 batch size、或用 prompt ensemble 重跑這個腳本(不需重新訓練,純推論端改動,成本低,值得優先試)

## 環境修復記錄(跟這個腳本邏輯本身無關,但下次在新環境重跑 phase3/phase5/phase6 任何腳本都會踩到)

這次執行時發現 sandbox 環境被重置過,跟 Phase 5 訓練完成時的環境已經不一樣了(torch 掉回預設 cu130 build、`omegaconf`/`transformers`/`nltk`/`monai`/`iopath` 全部消失、Python 版本是 3.12)。修復過程:

1. **torch/CUDA**:重新裝 `torch==2.14.0+cu126`/`torchvision==0.29.0+cu126`(配這台機器 L40S 的 driver 575.64.03/CUDA 12.9)。
2. **omegaconf/monai/nltk/iopath**:直接 `pip install`,沒有版本衝突。
3. **`transformers==4.25` 這條路徑已經走不通了**:它需要 `tokenizers<0.14`,而這個版本的 tokenizers 沒有 Python 3.12 的預編譯 wheel,只能從原始碼編譯——裝了 apt 的 rustc 1.75 編譯失敗(crates.io 上遊套件需要 `edition2024`,太新的 rustc 才支援),改用 `rustup` 裝最新穩定版 rustc 又卡在缺 `pkg-config`/`libssl-dev`(openssl-sys 編譯需要),裝完這兩個之後 tokenizers 的 Rust 原始碼本身又跟現在的 rustc/crates 生態系有更深的相容性問題,判斷這條路不值得再深挖。
4. **改用「讓 vendored `med.py` 相容新版 transformers」這條路**(改用最新 `transformers==5.17.0` + 新版 `tokenizers`/`huggingface_hub`,在 `zero_shot_eval.py` 開頭用 monkeypatch 補回三組被新版拿掉/搬移的東西,完全沒有修改 vendored 的 `lavis/` 程式碼):
   - `apply_chunking_to_forward`/`prune_linear_layer`:只是搬到 `transformers.pytorch_utils`,補一個 alias 回 `transformers.modeling_utils` 就好。
   - `find_pruneable_heads_and_indices`:被整個拿掉了,但只有 `BertSelfAttention.prune_heads()`(這條 pipeline 完全不會呼叫)在用,補一個從舊版 transformers 搬回來的實作讓 import 不報錯即可。
   - `all_tied_weights_keys` 相關的 `AttributeError`:新版 `from_pretrained()` 假設模型在 `__init__` 時已經呼叫過 `post_init()`,但 `med.py` 的 `BertModel` 子類別是舊寫法(直接呼叫 `init_weights()`),沒有呼叫 `post_init()`。補的方式是包一層 `_finalize_model_loading`,缺這個屬性時先手動呼叫 `model.post_init()`。
   - `get_head_mask`/`get_extended_attention_mask`/`invert_attention_mask`:這三個 `ModuleUtilsMixin` 的方法在新版被整個拿掉(現代模型不再需要手動組 attention bias mask),但 `med.py` 的 `BertModel.forward()` 還在呼叫。補的方式是把這三個方法的舊版標準實作(這幾年幾乎沒變過)重新註冊回 `PreTrainedModel` class。

這些修補都寫在 `zero_shot_eval.py` 檔案開頭(bypass `lavis/__init__.py` 那段 boilerplate的延伸),沒有動到 `phase3/lavis/` 底下任何 vendored 檔案。**`phase5/train.py` 如果要在這個(或任何重置過的)環境重跑,需要複製貼上同一段 monkeypatch**,否則會踩到一樣的 import 錯誤。
