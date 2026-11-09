cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
for w in baseline soft bone all3; do
  OMP_NUM_THREADS=3 nohup ~/miniconda3/envs/radar/bin/python bone_probe.py --stage agg --threads 3 --wins $w --pools mean max > ../outputs_fracture/bones/agg_$w.log 2>&1 &
  sleep 30
done
