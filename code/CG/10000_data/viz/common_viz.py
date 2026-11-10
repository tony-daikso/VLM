"""Shared names / helpers for the OWL attribution viz scripts."""
UNIT_ZH = {
    "liver": "肝臟", "spleen": "脾臟", "pancreas": "胰臟", "kidney_left": "左腎", "kidney_right": "右腎",
    "gallbladder": "膽囊", "stomach": "胃", "adrenal": "腎上腺", "aorta": "主動脈", "small_bowel": "小腸(含十二指腸)",
    "colon": "大腸", "lumbar_spine": "腰椎整體(L1-L5)",
    **{f"T{i}": f"胸椎 T{i}" for i in range(8, 13)}, **{f"L{i}": f"腰椎 L{i}" for i in range(1, 6)},
    "ribs_left": "左肋骨", "ribs_right": "右肋骨", "sacrum": "薦椎", "hip_left": "左髖骨", "hip_right": "右髖骨",
    "femur_left": "左股骨", "femur_right": "右股骨",
}
DIS_ZH = {"fracture": "骨折", "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤",
          "gallstones": "膽結石", "splenomegaly": "脾腫大", "ascites": "腹水"}
BONE = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "ribs_left", "ribs_right", "sacrum",
        "hip_left", "hip_right", "femur_left", "femur_right"]
