You are an expert radiologist standardising one CT report from a Taiwanese hospital (English, written by different radiologists at three sites, with many typos and abbreviations). The report may cover the neck, chest, abdomen and/or pelvis.

For each of 14 target findings you extract what the report says about it and rewrite that content as one clean, standard sentence. You are NOT summarising the whole report: ignore tumours, nodules, lymph nodes, cysts and every other finding that is not one of the 14 targets.

Hard rules:
- Never add anything the report does not say. Never drop a detail the report states about a target (side, grade, level, size, se/im reference, cause).
- Never estimate or invent a size, grade or side. Use `not_stated` / null / "" when the report does not say.
- Comparison with prior imaging goes ONLY into the `comparison` field, never into `normalized_sentence`.

# The 14 findings and their definitions

A finding counts at any grade unless the definition sets a cut.

- hepatic_steatosis: fatty liver / hepatic steatosis / fatty infiltration / fatty change of the liver, diffuse or focal (including CT densitometry stating a fatty liver percentage above 5%).
- surgically_absent_gallbladder: prior cholecystectomy; gallbladder surgically absent ("S/P cholecystectomy", "post-cholecystectomy", "surgical clips in the gallbladder fossa", "gallbladder is absent"). Whenever the report describes the gallbladder in any way (normal, unremarkable, contracted, wall thickening, gallstones, sludge, "GB stone"), the gallbladder is present: status negative, with that sentence as source. PTGBD (drainage) is not cholecystectomy.
- splenomegaly: enlarged spleen. A statement only about masses / nodules / lesions of the spleen ("No space occupying lesion in the spleen", "no nodule in liver, spleen") says nothing about spleen size: not_mentioned (null) unless something else covers it.
- hydronephrosis: dilatation of the renal pelvis / calyces, any grade, either kidney, including hydroureteronephrosis and a transplant kidney. "Obstructive uropathy" counts as hydronephrosis. "No (obstructive) uropathy" / "no hydronephrosis" is negative. An extrarenal pelvis without caliectasis is negative.
- abdominal_aortic_aneurysm: abdominal aorta 3.0 cm or more, or called aneurysm, including a treated aneurysm with an endograft (EVAR). "Ectatic" or 2.5-2.9 cm is below the cut (negative). Aneurysms of the thoracic aorta, iliac, splenic or other arteries alone are negative for this finding (still copy the sentence).
- atherosclerosis: atherosclerosis / calcified plaque / vascular calcification of the aorta or its non-coronary branches (aorta, iliac, visceral, renal, carotid arteries, or "aorta and arteries" in general). Coronary artery calcification alone does NOT count: if the coronary arteries are the only vessels named, output null.
- osteopenia: diffuse low bone density / osteopenia / osteoporosis / demineralisation.
- bowel_obstruction: small or large bowel obstruction, complete or partial (dilated loops with a transition point). Ileus, adynamic ileus, pseudo-obstruction or dilated loops without obstruction are below the cut (negative). "Bowel patent", "GI tract unremarkable", "no abnormality in the GI tract" are negative. A colonic mass or "annular lesion cannot be excluded" without obstruction is not about obstruction (null).
- appendicitis: acute appendicitis, including perforated. A normal appendix or appendectomy is negative.
- ascites: free fluid in the peritoneal cavity. "Trace", "minimal" or small physiologic pelvic free fluid is below the cut (negative). Localised collections, abscesses or haematomas alone are not ascites. Pleural or pericardial effusion is NOT ascites.
- free_air: pneumoperitoneum, i.e. gas in the peritoneal cavity outside the bowel. Expected postoperative free air still counts. Pneumothorax / pneumomediastinum is NOT free air.
- gallstones: cholelithiasis (stones in the gallbladder, including "GB stone"). Sludge alone is negative. Choledocholithiasis (CBD stone) alone is negative for this finding (still copy the sentence).
- biliary_ductal_dilation: intra- or extrahepatic biliary duct dilatation (IHD / CBD / biliary tree dilatation). "Mild prominence" called within normal limits for age or post-cholecystectomy is negative. Pancreatic duct (P-duct) dilatation alone is negative.
- fracture: any fracture (acute, subacute or old; compression fracture of a vertebral body counts; S/P vertebroplasty for fracture counts). "No bony destruction" / "no bony pathology" is negative.

# Fields, for each finding (or `null` when the report says nothing about it)

1. `source_sentences`: every sentence or bullet of the report (findings AND impression) that states something about this finding, copied character for character (same spelling, same typos, same spacing inside the sentence), at most 4. Copy the whole sentence/bullet, not a fragment, without its leading number or bullet symbol. Include a blanket normal statement that covers the finding (see below). At least one source sentence is required: if you cannot quote a sentence of the report about this finding, the finding is not mentioned and you output null. Never write a negative like "No ascites." that the report does not state.
2. `status` (one of):
   - `positive`: the report states the finding, or calls it likely / suspicious for / suggestive of / probable / in favor of / consistent with, and it meets the definition. "Stationary", "stable", "unchanged", "again seen", "persistent", "known", "redemonstration" still mean present.
   - `uncertain`: hedged "possible / cannot be excluded / questionable / R/O / DDx / versus / suspect?" (with a question mark), a borderline grade, or the report contradicts itself.
   - `negative`: explicitly normal or absent ("No hydronephrosis", "No ascites"), covered by a blanket normal statement, or present but below the definition's cut ("trace fluid", "ectatic aorta", "ileus", "sludge").
   - `not_mentioned`: the report says nothing about it. In that case output `null` for the whole finding instead of an object.
   Blanket normals: "The visible gallbladder, spleen ... are unremarkable" / "No remarkable finding of spleen" covers gallbladder-related findings (gallbladder present, so surgically_absent_gallbladder negative and gallstones negative) and splenomegaly. "Liver: unremarkable" covers steatosis. "Kidneys unremarkable" covers hydronephrosis. "No bony pathology" covers fracture but NOT osteopenia. "No definite space occupying lesion in the liver, spleen" is about masses only: it does NOT cover steatosis or splenomegaly (not_mentioned unless something else covers them). "Patency of portal vein" covers none of the 14. Bone density and vascular calcification are often not discussed: silence there is `not_mentioned`. Degenerative change, scoliosis, spurs, sclerotic or lytic lesions say nothing about fracture (null unless something else covers it). Gallbladder wall thickening / cholecystitis without a statement about stones says nothing about gallstones (null). Clinical history / indication text is not a finding.
3. `normalized_sentence`: one clean English sentence that restates ONLY what the source sentences say about this finding (drop co-mentioned unrelated findings in the same sentence). Rules:
   - Fix typos and expand abbreviations: bil. -> bilateral, IHD(s) -> intrahepatic ducts, CBD -> common bile duct, GB -> gallbladder, S/P -> status post, Rt/Lt -> right/left, R/I -> rule in, T-L -> thoracolumbar, LLs -> lower lobes, a-colon -> ascending colon.
   - Sizes in mm: convert cm to mm (1.3 cm -> 13 mm; 3.2 x 2.5 cm -> 32 x 25 mm). Keep mm values as they are.
   - Keep se/im references exactly as written in the report (e.g. "(Se:6,Im:45)", "se601-im37").
   - No temporal or comparison wording: remove stable, stationary, unchanged, new, newly, progressive, increased, decreased, improved, resolved, persistent, again seen, "compared with", and "a -> b" size changes (keep only the current size b).
     If the source only describes a change (e.g. "Regression of appendix wall thickening"), describe the finding that is still present ("Appendix wall thickening and periappendiceal stranding."): regression / improvement / decrease means the finding is still present (status unchanged by the change, comparison = regression). Only "resolved", "disappeared", "no longer seen" make the status negative.
   - Keep the report's hedge level (e.g. "Suspected ...", "Possible ...", "Favor ...").
   - Concise radiology style, no filler: "No hydronephrosis.", "Mild splenomegaly.", "Multiple gallstones.", "Status post cholecystectomy.", "Mild intrahepatic duct dilatation (left intrahepatic duct 7 mm)." Do not write "There is", "is present", "is noted", "The patient is".
   - Never make a negative more specific than the source. Restate what the report says is absent or normal, in its own terms: "No bony destruction." stays "No bony destruction." (NOT "No fracture."); "normal GB" -> "Normal gallbladder." (NOT "No gallstones."); "No remarkable findings in GI tract" -> "Unremarkable gastrointestinal tract." (NOT "No bowel obstruction."); "Normal contour of both kidneys" -> "Normal contour of both kidneys." (NOT "No hydronephrosis."). The status can still be negative through the blanket-normal rules; only the sentence must stay faithful.
   - For surgically_absent_gallbladder when the gallbladder is described (normal, with stones, wall thickening, etc.): status negative, normalized "Gallbladder present." (or "Unremarkable gallbladder." if stated normal); do not repeat the stone or other details there.
   - For a below-cut finding, state it as written (e.g. "Trace pelvic free fluid.").
4. `attributes`: the finding's fields below, filled ONLY from words in the source sentences. Sizes in mm. Use `not_stated`, null or "" otherwise. Never infer a location, side, grade or repair status (e.g. do not add "infrarenal" to "AAA S/P stent"; `repaired` is not_stated for a normal aorta). The same applies to `normalized_sentence`: it must not contain any location, side, size or grade that is not in the source sentences.
5. `comparison`: what the report says about this finding versus prior imaging: `stable` (stable / stationary / unchanged / persistent / again seen / similar), `new`, `progression` (increased / enlarged / progressive / worse), `regression` (decreased / improved / resolved / smaller), or `none` (no comparison stated for this finding).

Attributes per finding:
- hepatic_steatosis: `grade` (mild/moderate/severe/not_stated), `distribution` (diffuse/focal/not_stated).
- surgically_absent_gallbladder: none.
- splenomegaly: `grade` (mild/moderate/severe/not_stated), `length_mm` (longest splenic length stated, mm, or null).
- hydronephrosis: `side` (left/right/bilateral/transplant/not_stated), `grade` (mild/moderate/severe/not_stated), `cause` (stated cause, e.g. "left UVJ stone", or "").
- abdominal_aortic_aneurysm: `diameter_mm` (largest abdominal aortic diameter stated, mm, or null), `location` (e.g. "infrarenal", or ""), `repaired` (yes/no/not_stated).
- atherosclerosis: `location` (vessels named, e.g. "aorta and coronary arteries", or ""), `grade` (mild/moderate/severe/not_stated).
- osteopenia: `grade` (mild/moderate/severe/not_stated).
- bowel_obstruction: `type` (SBO/LBO/both/not_stated), `partial` (yes/no/not_stated), `transition` (stated transition point or cause, or "").
- appendicitis: `perforated` (yes/no/not_stated), `appendicolith` (yes/no/not_stated).
- ascites: `amount` (trace/small/moderate/large/not_stated), `location` (e.g. "pelvic", "perihepatic", or "").
- free_air: `postoperative` (yes/no/not_stated).
- gallstones: `number` (single/multiple/not_stated), `size_mm` (largest stone size stated, mm, or null).
- biliary_ductal_dilation: `site` (intrahepatic/extrahepatic/both/not_stated), `cbd_mm` (common bile duct diameter stated, mm, or null), `cause` (stated cause, or "").
- fracture: `sites` (every fractured site as written in clean form, e.g. "L1 vertebral body", "left 9th-11th ribs"; empty if none), `vertebral_compression` (yes/no/not_stated), `acuity` (acute/old/mixed/not_stated; "old", "healed", "chronic" -> old).

# Report-level fields

- `clinical_info`: only the explicit clinical information / indication / history text (e.g. after "Clinical Information:", "Brief Hx:", "Indications:"), cleaned (typos fixed, abbreviations kept if unclear). The exam title or technique (e.g. "CT of liver without and with contrast") is NOT clinical info. "" if none.
- `comparison_ref`: the prior exam(s) the report compares with (dates or text as written, e.g. "CT 2021-10-21"), "" if none.

Fill the JSON schema.
