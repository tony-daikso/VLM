# Leave-one-window-out (E3_late, val): AUROC drop when the window is removed (+ = window helps)

## per finding

| finding | full | baseline | soft_tissue | liver | narrow_abd | bone |
|---|---|---|---|---|---|---|
| hepatic_steatosis | 0.839 | +0.001 | +0.001 | +0.001 | -0.001 | +0.001 |
| surgically_absent_gallbladder | 0.966 | +0.001 | +0.001 | +0.001 | +0.001 | -0.000 |
| splenomegaly | 0.980 | +0.000 | +0.000 | +0.001 | +0.000 | -0.000 |
| gallstones | 0.783 | -0.004 | -0.000 | -0.006 | +0.002 | +0.001 |
| biliary_ductal_dilation | 0.797 | +0.001 | -0.001 | +0.002 | +0.002 | +0.001 |
| hydronephrosis | 0.866 | +0.001 | +0.007 | +0.010 | -0.001 | -0.003 |
| abdominal_aortic_aneurysm | 0.967 | +0.001 | -0.000 | +0.002 | +0.001 | -0.001 |
| atherosclerosis | 0.891 | +0.002 | -0.000 | +0.001 | +0.001 | -0.001 |
| osteopenia | 0.879 | +0.001 | +0.001 | +0.000 | -0.001 | +0.000 |
| fracture | 0.776 | +0.003 | +0.001 | +0.000 | +0.002 | -0.000 |
| bowel_obstruction | 0.950 | +0.001 | +0.000 | +0.001 | -0.001 | +0.000 |
| appendicitis | 0.828 | +0.003 | +0.003 | -0.001 | -0.002 | +0.005 |
| ascites | 0.872 | +0.000 | +0.000 | +0.001 | +0.003 | -0.002 |
| free_air | 0.879 | +0.001 | +0.001 | -0.000 | +0.002 | +0.001 |

## per organ unit (mean over its findings)

| unit | baseline | soft_tissue | liver | narrow_abd | bone |
|---|---|---|---|---|---|
| liver | +0.001 | +0.000 | +0.001 | +0.002 | +0.000 |
| gallbladder | -0.002 | +0.000 | -0.003 | +0.001 | +0.000 |
| spleen | +0.000 | +0.000 | +0.001 | +0.002 | -0.001 |
| pancreas | +0.001 | -0.001 | +0.002 | +0.002 | +0.001 |
| kidney_left | +0.001 | +0.007 | +0.010 | -0.001 | -0.003 |
| kidney_right | +0.001 | +0.007 | +0.010 | -0.001 | -0.003 |
| aorta | +0.001 | -0.000 | +0.001 | +0.001 | -0.001 |
| lumbar_spine | +0.002 | +0.001 | +0.000 | +0.001 | +0.000 |
| small_bowel | +0.001 | +0.001 | +0.001 | +0.001 | +0.001 |
| colon | +0.002 | +0.001 | +0.000 | -0.000 | +0.002 |
| stomach | +0.001 | +0.001 | -0.000 | +0.002 | +0.001 |
