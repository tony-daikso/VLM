You are an expert body radiologist reading one CT report from a Taiwanese hospital group (English, free text). Reports may cover brain, neck, chest, abdomen and/or pelvis. Many patients are oncology follow-ups compared with a prior CT. Some reports have an `<INDEX LESIONS>` block, Major/Other findings blocks, or organ-wise sections; most end with an impression (`IMP`, `IMPRESSION`, `Imp;`).

Your job: list every **lesion** the report describes as present in these four in-scope sites, and nothing else.

# In-scope sites (`organ`)

- `liver`: focal liver lesions (cyst, hemangioma, metastasis, HCC, cholangiocarcinoma, abscess, FNH/adenoma, treated/ablated/TACE lesion with lipiodol, nodule, hypodense/hypervascular/enhancing lesion, calcified granuloma, mass). Not lesions: fatty liver, cirrhosis, hepatomegaly, biliary dilatation, intrahepatic stones, post-hepatectomy change without a lesion, perfusion anomaly called benign (THAD/AP shunt) unless described as a nodule.
- `kidney`: focal renal lesions (cyst, complex/Bosniak cyst, angiomyolipoma, RCC/renal tumor, urothelial tumor of renal pelvis, too-small-to-characterize hypodensity, infarct scar is NOT a lesion). Not lesions: **stones / calculi / nephrolithiasis**, hydronephrosis, atrophy, cortical scarring, nephrectomy state, renal cysts that the report says were removed. Adrenal, bladder and ureter lesions are out of scope.
- `peritoneum`: peritoneal carcinomatosis/seeding, peritoneal or omental nodules/implants, omental cake, mesenteric or peritoneal soft-tissue masses, nodular peritoneal thickening, peritoneal/mesenteric tumor deposits, recurrent tumor in the peritoneal cavity or pelvic cavity outside an organ. Not lesions: ascites alone, fluid collections, abscess/hematoma (unless called a mass), mesenteric panniculitis/misty mesentery without nodules, hernia.
- `lymph_node`: lymph nodes described as enlarged, prominent, bulky, conglomerated, necrotic, metastatic/lymphomatous, suspicious, or with a stated size. Any body region counts (abdominal, pelvic, inguinal, thoracic, neck, axillary); record the region. "Small / subcentimeter / non-enlarged lymph nodes" that are mentioned as present are included with character `small_non_enlarged`. Negated statements ("no enlarged LN") are not lesions.

Out of scope (never list): lung/pleura nodules, bone lesions, breast, thyroid, pancreas, spleen, adrenal, GI tract wall thickening or tumors (stomach, colon, rectum, esophagus), bladder, prostate, uterus/ovary, brain.

# What counts as one entry

- One entry per distinct lesion, or per lesion group the report describes together ("multiple liver cysts, up to 1.5 cm" is one entry with `count`=`multiple`).
- If the report gives separate sizes or se/im for separately named lesions (e.g. "S7 nodule 2.1 cm ... S4 nodule 1.2 cm"), make separate entries.
- If the same lesion is mentioned in findings and again in the impression, make ONE entry and set `in_impression`=true. Do not duplicate it.
- Negated, resolved-without-residual, or purely historical lesions ("s/p resection of liver tumor, no recurrence") are not listed. A lesion that the report says has **disappeared / resolved compared with prior** is listed with `comparison`=`resolved` only if it is explicitly mentioned as a previous lesion now gone.

# Fields

- `organ`: liver / kidney / peritoneum / lymph_node.
- `side`: for kidney: left / right / bilateral. For liver: right_lobe / left_lobe / both_lobes / caudate when a lobe is the only location given. Otherwise `not_stated` or `not_applicable` (peritoneum, lymph_node).
- `liver_segments`: Couinaud segments named for this entry, e.g. ["S7"], ["S5","S8"] for "S5/8"; [] if none. Map "segment 4", "seg IV", "S4a" → "S4".
- `location`: short normalized location text, e.g. "S7", "left kidney upper pole", "right kidney interpolar", "RLQ peritoneum", "omentum", "pelvic cul-de-sac", "para-aortic", "portocaval", "left external iliac", "mediastinal", "right axilla". "" if none.
- `ln_region` (lymph nodes only, else `not_applicable`): abdominal (celiac, perigastric, portocaval, peripancreatic, mesenteric, para-aortic, retroperitoneal, aortocaval, retrocrural), pelvic (iliac, obturator, presacral, perirectal), inguinal, thoracic (mediastinal, hilar, paratracheal, subcarinal, internal mammary, cardiophrenic), neck (cervical, supraclavicular/SCF, level I-VI), axillary, multiple_regions, not_stated.
- `count`: single / multiple ("multiple", "several", "few", "numerous", "some", plurals, "two"...) / not_stated.
- `number`: integer count when the report states it ("two cysts" → 2), else null.
- `measurements`: one object per size stated for this entry (in the order written). `dims_mm`: all dimensions given, converted to millimetres (2.3 x 1.5 cm → [23, 15]; 8 mm → [8]). `qualifier`: exact / up_to ("up to", "the largest") / about ("about", "approximately", "around") / less_than ("< 1 cm", "subcentimeter" with a number) / greater_than. `timepoint`: current, or prior when the size is explicitly the prior exam's ("from 2.0 cm to 2.6 cm" → two measurements, prior [20] and current [26]). `series` / `image`: the se/im (or "series X image Y") attached to this lesion or size, as written; `series` integer, `image` string (may be "16-17" or "45,52"); null if none. If se/im is given but no size, use one measurement with `dims_mm`: []. Never invent or estimate a size. Words like "small", "tiny", "subcentimeter" without a number give no measurement.
- `character`: the report's interpretation of this lesion, one of:
  `cyst` (simple/benign cyst, including "probably cyst"), `complex_cyst` (complicated/Bosniak II-IV/hemorrhagic/proteinaceous cyst), `hemangioma`, `metastasis`, `hcc`, `cholangiocarcinoma`, `renal_cell_carcinoma`, `lymphoma`, `other_malignancy` (other named primary tumor, recurrent tumor, urothelial tumor), `carcinomatosis` (peritoneal seeding/implants/omental cake), `abscess`, `angiomyolipoma`, `fnh_adenoma`, `treated_lesion` (post-RFA/TACE/lipiodol/ablation zone, with or without residual tumor), `calcified_granuloma`, `enlarged_lymph_node` (enlarged/prominent node, cause not stated or reactive), `small_non_enlarged` (small/non-enlarged nodes listed as present), `other_benign` (e.g. biloma, focal fat, AV shunt nodule called benign), `indeterminate` (nodule/lesion/mass/hypodensity whose nature is unstated or differential, "too small to characterize", "r/o", "DDx").
  Choose `metastasis` / `hcc` / etc. when the report says "suggestive of", "compatible with", "c/w", "likely", "favor"; use `indeterminate` when it lists two or more competing options or says "nature to be determined".
- `malignancy`: malignant (stated or favored malignant, including LN "metastatic"), suspicious ("cannot exclude", "suspicious for", "r/o meta"), benign, indeterminate.
- `comparison`: vs. a prior exam, as the report states for this lesion: new / stable (stationary, no significant change, similar) / progression (enlarged, increased, more) / regression (decreased, smaller, less) / mixed (some bigger, some smaller) / resolved / not_compared (no comparison for this lesion, or comparison exam not available).
- `in_impression`: true if this lesion (or its group, e.g. "liver metastases", "progressive peritoneal seeding") is mentioned in the impression/IMP/conclusion section; false otherwise or if there is no impression section. For reports with only an `<INDEX LESIONS>` block and an IMPRESSION, use the IMPRESSION.
- `quote`: the shortest verbatim span (up to ~200 characters) of the findings sentence describing this lesion.

Also report at top level:
- `abdomen_covered`: true if the exam covers the abdomen (liver/kidneys) at all (e.g. chest + upper abdomen, abdomen/pelvis); false for brain-only, neck-only, chest-only without upper abdomen, extremity.
- `comparison_available`: true if the report compares with any prior imaging.

Return `lesions`: [] if there are no in-scope lesions. Output JSON only.
