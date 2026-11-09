#!/bin/bash
# wait for the running s0 python of branch $1 (pid $2), then run the driver with the single config
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture
while kill -0 $2 2>/dev/null; do sleep 30; done
bash spine_ft/scripts/run_branch.sh $1 "tf2lse=--train-from 2 --pool lse --epochs 8"
