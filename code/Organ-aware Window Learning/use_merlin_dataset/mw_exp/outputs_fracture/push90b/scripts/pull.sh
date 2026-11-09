#!/bin/bash
# push90b: pull small score files (score_fold*.csv, summary.json) from CT-rate / Spark into A6000 push90b/B/prod (PE) and P/runs (pelvis).
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90b; CT=daikso@192.168.0.189; SP=daikso_spark@192.168.0.95; mkdir -p B/prod P/runs
rsync -a $CT:/media/daikso/Tony/VLM/data/CT/merlin/push90b/pe/prod/ B/prod/ 2>> logs/pull.err
rsync -a $SP:Desktop/Tony/OWL/push90b/pe/prod/ B/prod/ 2>> logs/pull.err
rsync -a $SP:Desktop/Tony/OWL/push90b/P/runs/ P/runs/ 2>> logs/pull.err
rsync -a $CT:/media/daikso/Tony/VLM/data/CT/merlin/push90b/P/runs/ P/runs/ 2>> logs/pull.err
ls B/prod P/runs
