# CT 腹部診斷模型比較 — CG2 Abdomen72 測試結果

測試資料:`datadrive/VLM/data/CT/CG/abdomen.zip`(72 位病人腹部 CT + 標註)
評測方式:zero-shot image-text 相似度分數,依此排序算每個 finding 的 AUC(不做額外分類器訓練)

## 總結果表

| 排名 | 模型 | Mean AUC | 訓練方式 |
|---|---|---|---|
| 1 | **RADAR** | **0.8935**(自身 28 個可評測 finding) / 0.8758(四模型共同 24 個) | 監督式對比學習,40 萬+ 筆腹部顯影 CT + 報告,針對這 146 個 finding taxonomy 量身訓練 |
| 2 | **Pillar0-AbdomenCT** | 0.6401(自身)/ 0.6659(共同) | 通用腹部-骨盆 CT 基礎模型,zero-shot 遷移到本次 finding 定義 |
| 3 | **TotalFM** | 0.6446(自身,24 個)/ 0.6446(共同) | 通用器官分離式 CT 基礎模型,zero-shot |
| 4 | **Merlin** | 0.5963(report 風格 prompt)/ 0.6135(共同) | 通用腹部 CT 基礎模型,zero-shot(短 prompt 版本更差,0.5376) |

## 關鍵結論

- **RADAR 遙遙領先,不是因為它用了額外訓練的分類器** — 四個模型的推論機制本質上都相同(image embedding 對 text embedding 算 cosine similarity,不靠訓練 linear probe)。RADAR 準的原因是它的對比學習訓練資料**跟評測任務同域、同 taxonomy**(針對這 146 個 finding 專門設計 pos/neg prompt);其他三個是通用 CT 基礎模型,硬套到一個牠們沒見過的分類體系上做真正的 zero-shot 遷移,天生會有落差。
- Pillar0 官方論文報的 86.4 AUROC 是用 **linear probing**(在大規模 UCSF 資料上額外訓練分類頭)協定測出來的,不是 zero-shot;官方沒有釋出這個分類頭權重,而且我們手上 72 筆資料也不夠拿來自己訓練一個有意義的 probe。
- Merlin 對 prompt 措辭很敏感:短 prompt(`"Metastasis in Liver"`)vs 報告段落風格(`"Liver: Metastasis."`)分數從 0.5376 提升到 0.5963,個別 finding(如肝轉移瘤)AUC 差距可達 0.1→0.98。

## 過程中的重要教訓

- 測 Pillar0 時,依照訓練 config(`image_mean/image_std` normalize、1.5mm spacing)做了「看似更正確」的前處理修正,結果 AUC 從 0.64 掉到 0.39(比隨機還差)。消融測試證實兩個改動都讓結果變差 —— 最終判斷:HuggingFace 釋出的 inference checkpoint 已經把訓練期的 normalize 吸收進模型權重,額外再做一次等於重複 normalize。**結論:實測消融比死摳訓練 config 文件更可靠。**
- TotalFM 套件本身有個 bug(`gte-modernbert-base` 用 bf16 輸出、投影層是 fp32,dtype 不匹配導致 crash),已在本地 patch 修掉。

## 找過但還沒測試的候選模型

| 模型 | 狀態 |
|---|---|
| SPECTRE(CVPR 2026) | 已確認公開權重(`cclaess/SPECTRE-Large`),zero-shot 能力待驗證 |
| TAP-CT | 純 SSL(DINOv2 風格),無 zero-shot 能力,需另訓練 linear probe |
| CT-FM | 純 SSL,無 zero-shot 能力,需另訓練 linear probe |
| Ker-VLJEPA / CT-CLIP / CT-CHAT | 胸腔 CT 訓練,domain 不符腹部資料 |
| U-VLM(官方) | 只有程式碼,官方未釋出訓練好的權重 |

## 各模型結果檔案位置

- RADAR:`/root/Desktop/VLM/RADAR/CT/results/CG2_abdomen72_auc_per_finding.csv`
- Pillar0-AbdomenCT:`/root/Desktop/VLM/Pillar0/CT/results/CG2_abdomen72_auc_per_finding.csv`
- Merlin:`/root/Desktop/VLM/Merlin/CT/results/CG2_abdomen72_auc_per_finding_v2_reportstyle.csv`
- TotalFM:`/root/Desktop/VLM/TotalFM/CT/results/CG2_abdomen72_auc_per_finding.csv`
