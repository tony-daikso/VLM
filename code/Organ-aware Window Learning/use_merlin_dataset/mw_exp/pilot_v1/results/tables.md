N test = 15; bootstrap unit = study (no patient id available); 1000 resamples

| exp | control | low_contrast | other | main_all |
|---|---|---|---|---|
| E0 | 0.833 [0.679, 0.954] | 0.750 [0.462, 1.000] | 0.658 [0.571, 0.815] | 0.689 [0.564, 0.849] |
| E1_bone | 0.824 [0.672, 0.950] | 0.729 [0.462, 1.000] | 0.687 [0.582, 0.845] | 0.701 [0.589, 0.859] |
| E1_liver | 0.870 [0.741, 0.975] | 0.764 [0.512, 1.000] | 0.491 [0.378, 0.644] | 0.582 [0.470, 0.724] |
| E1_narrow_abd | 0.917 [0.831, 0.991] | 0.639 [0.292, 0.909] | 0.539 [0.403, 0.747] | 0.572 [0.435, 0.742] |
| E1_soft_tissue | 0.907 [0.818, 0.990] | 0.792 [0.591, 1.000] | 0.648 [0.501, 0.852] | 0.696 [0.584, 0.841] |
| E2 | 0.898 [0.787, 1.000] | 0.556 [0.231, 0.857] | 0.548 [0.445, 0.747] | 0.551 [0.429, 0.711] |
| E3 | 0.889 [0.785, 0.990] | 0.688 [0.417, 1.000] | 0.575 [0.478, 0.736] | 0.612 [0.517, 0.760] |
| E4 | 0.861 [0.721, 0.975] | 0.688 [0.423, 0.944] | 0.644 [0.491, 0.813] | 0.658 [0.517, 0.819] |

| pair | subset | diff | 95% CI | CI excludes 0 |
|---|---|---|---|---|
| E3-E0 | control | +0.056 | [-0.102, +0.222] | False |
| E3-E0 | low_contrast | -0.062 | [-0.273, +0.083] | False |
| E3-E0 | other | -0.084 | [-0.191, +0.035] | False |
| E3-E0 | main_all | -0.077 | [-0.185, +0.023] | False |
| E4-E0 | control | +0.028 | [-0.140, +0.214] | False |
| E4-E0 | low_contrast | -0.062 | [-0.385, +0.201] | False |
| E4-E0 | other | -0.014 | [-0.152, +0.102] | False |
| E4-E0 | main_all | -0.030 | [-0.199, +0.095] | False |
| E3-E4 | control | +0.028 | [+0.000, +0.111] | False |
| E3-E4 | low_contrast | +0.000 | [-0.181, +0.192] | False |
| E3-E4 | other | -0.069 | [-0.207, +0.079] | False |
| E3-E4 | main_all | -0.046 | [-0.133, +0.082] | False |

| finding | test pos | E0 | E1_bone | E1_liver | E1_narrow_abd | E1_soft_tissue | E2 | E3 | E4 |
|---|---|---|---|---|---|---|---|---|---|
| hepatic_steatosis | 9 | 0.778 | 0.759 | 0.852 | 0.944 | 0.926 | 0.907 | 0.889 | 0.833 |
| surgically_absent_gallbladder | 6 | 0.889 | 0.889 | 0.889 | 0.889 | 0.889 | 0.889 | 0.889 | 0.889 |
| splenomegaly | 1 | 0.429 | 0.500 | 0.214 | 0.500 | 0.286 | 0.357 | 0.429 | 0.286 |
| gallstones | 3 | 0.917 | 0.917 | 0.944 | 0.694 | 0.917 | 0.694 | 0.917 | 0.833 |
| biliary_ductal_dilation | 2 | 0.583 | 0.542 | 0.583 | 0.583 | 0.667 | 0.417 | 0.458 | 0.542 |
| hydronephrosis (excl.) | 0 | nan | nan | nan | nan | nan | nan | nan | nan |
| abdominal_aortic_aneurysm (excl.) | 0 | nan | nan | nan | nan | nan | nan | nan | nan |
| atherosclerosis | 4 | 0.750 | 0.795 | 0.545 | 0.659 | 0.750 | 0.636 | 0.750 | 0.773 |
| osteopenia (excl.) | 0 | nan | nan | nan | nan | nan | nan | nan | nan |
| fracture (excl.) | 0 | nan | nan | nan | nan | nan | nan | nan | nan |
| bowel_obstruction | 1 | 0.714 | 0.714 | 0.643 | 0.357 | 0.857 | 0.500 | 0.500 | 0.857 |
| appendicitis (excl.) | 0 | nan | nan | nan | nan | nan | nan | nan | nan |
| ascites | 5 | 0.740 | 0.740 | 0.560 | 0.640 | 0.700 | 0.700 | 0.620 | 0.660 |
| free_air (excl.) | 0 | nan | nan | nan | nan | nan | nan | nan | nan |
