cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY="nice -n 19 env OMP_NUM_THREADS=3 /home/advantech/miniconda3/envs/radar/bin/python"
O=../outputs_vert15/interim_eval
[ -f $O/v15_matrix.npz ] || $PY vert15_probe.py --stage build --bone-matrix ../outputs_fracture/bones/bone_matrix.npz --emb-dirs ../outputs_vert15/merged_interim/emb15 --out $O
$PY vert15_probe.py --stage oof --out $O --subset-file ../outputs_vert15/ids_prio.csv --grid --cfgs ref add addc repl --seeds 0 1 2 --threads 3
echo J_OOF_DONE
