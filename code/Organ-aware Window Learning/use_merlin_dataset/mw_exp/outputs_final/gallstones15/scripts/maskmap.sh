# writes inputs/maskmap.tsv for A6000 (run in mw_exp)
for d in outputs/masks outputs_vert15/masks3_extra; do for f in $d/*.nii.gz; do b=$(basename $f .nii.gz); echo -e "$b\t$PWD/$f"; done; done
