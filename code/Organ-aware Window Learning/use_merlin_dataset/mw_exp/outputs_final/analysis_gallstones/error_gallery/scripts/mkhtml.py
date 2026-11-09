import json, sys, html
D,OUT=sys.argv[1],sys.argv[2]
R=json.load(open(f"{D}/recs.json"))
N={"FN-1":"膽囊 39 mL，內容物均勻，最高 118 HU，沒有任何 >130 HU 體素；3 mm 影像上看不到結石（典型等密度／膽固醇結石）。",
"FN-2":"膽囊 24 mL，最高 124 HU，腔內均勻；影像上看不到結石，模型分數只有 0.025。",
"FN-3":"膽囊偏大（79 mL）。最高 HU（509）落在遮罩後緣，緊貼腔外的高密度結構，腔內看不到明顯結石。可能是頸部小結石，也可能是遮罩邊緣滲入鄰近強化組織。",
"FN-4":"膽囊頸部有一小顆極高密度點（最高 1375 HU，只有 5 個 >200 HU 體素），在 mean pooling 下被稀釋；分數 0.414，接近閾值 0.598。1375 HU 偏高，也可能是金屬夾，需要人工確認。",
"FN-5":"膽囊遮罩只有 1.8 mL，落在膽囊窩邊緣；旁邊有一個圓形、密度略高的結構，疑為真正的膽囊（可能含結石或膽泥）但沒有被分割出來。",
"FN-6":"膽囊 12 mL，最高 105 HU，看不到結石；報告用語不確定，這個標籤本身可能偏弱。",
"FP-1":"小膽囊（8.6 mL）幾乎整個充滿 >200 HU 的內容物（134 個體素），看起來像鈣乳膽汁、排泄的顯影劑，或漏標的結石。",
"FP-2":"大膽囊（58 mL），只有靠肝臟側的壁上有 3 個 >130 HU 體素（部分容積），腔內乾淨；模型仍給 0.994，所以誤報不是高密度造成的。",
"FP-3":"標註為膽囊切除術後，但 TS 在膽囊窩附近的肝內分出一塊 3.5 mL 的低密度區（最高 69 HU），疑似誤分割；模型給 0.993。",
"FP-4":"膽囊 13 mL，只有邊緣 2 個 >130 HU 體素；整個腹部看起來有大量積液或其他明顯異常。誤報原因不明，可能是模型抓到全身性的異常捷徑。",
"TP-1":"膽囊後壁有層狀鈣化結石（974 HU，121 個 >200 HU 體素），很容易看出來。注意遮罩把部分結石排除在外。",
"TP-2":"膽囊底部有一串鈣化結石（740 HU，93 個 >200 HU 體素）。",
"TN-1":"膽囊均勻低密度，沒有高密度；模型分數約 0。",
"TN-2":"膽囊均勻低密度，沒有高密度。這例的掃描範圍只有 42 層，所以冠狀切面比較矮。"}
sec={"FN":"漏診（False negative）","FP":"誤報（False positive）","TP":"對照：正確偵測（True positive）","TN":"對照：正確陰性（True negative）"}
css=""":root{--bg:#fafafa;--fg:#1b1b1b;--card:#fff;--mut:#555;--bd:#ddd}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141414;--fg:#eee;--card:#1e1e1e;--mut:#aaa;--bd:#333}}
:root[data-theme=dark]{--bg:#141414;--fg:#eee;--card:#1e1e1e;--mut:#aaa;--bd:#333}
body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,"PingFang TC","Noto Sans TC",sans-serif;line-height:1.6}
main{max-width:1500px;margin:auto;padding:16px}h1{font-size:1.5rem}h2{margin-top:2rem;border-bottom:1px solid var(--bd)}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px;margin:16px 0}
.card img{width:100%;height:auto;display:block;border-radius:4px}.cap{font-size:.95rem}.mut{color:var(--mut);font-size:.9rem}
a{color:inherit}"""
h=[f"<!doctype html><html lang='zh-Hant-TW'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>膽結石錯誤圖庫</title><style>{css}</style></head><body><main>",
"<h1>膽結石模型錯誤圖庫</h1>",
"<p class='mut'>資料：只用 train（5-fold OOF，E3_late），未使用 val／test。閾值取 train OOF 陰性的 90% 特異度（logit 0.399，sigmoid 0.598），和 report.md 相同。病例已匿名化（FN-1…），不含報告原文。影像是 3 mm 前處理 CT，腹部窗 L40／W400；綠線是 TotalSegmentator 膽囊遮罩，紅十字標出膽囊內最高 HU 的位置，黃框是下方放大區。「模型分數」＝ sigmoid(E3_late OOF logit)。</p>",
"<p><a href='montage_overview.png'>總覽拼圖（montage_overview.png）</a></p>"]
cur=None
for r in R:
    g=r["anon"].split("-")[0]
    if g!=cur: h.append(f"<h2>{sec[g]}</h2>"); cur=g
    h.append(f"<div class='card'><div class='cap'><b>{html.escape(r['cap1'])}</b><br>{html.escape(r['cap2'])}<br>{html.escape(r['cap3'])}<br><b>影像重點：</b>{html.escape(N[r['anon']])}</div><a href='{r['file']}'><img src='{r['file']}' alt='{r['anon']}' loading='lazy'></a></div>")
h.append("</main></body></html>")
open(f"{OUT}/index.html","w").write("\n".join(h))
