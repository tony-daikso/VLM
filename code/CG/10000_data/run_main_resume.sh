#!/bin/bash
# Resume the main select_convert.py run after the 2026-11-05 drive disconnect; record its exit code.
# usage: nohup ./run_main_resume.sh > .../run_status_20261105/run_main_resume.out 2>&1 &
cd /Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data
BASE=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data
ST=$BASE/run_status_20261105
rm -f $ST/main_exit_code
echo "$(date '+%F %T') start select_convert.py --workers 2 (log nifti_3mm/convert_resume.log)"
caffeinate -i /Users/tony.tu/miniconda3/envs/VLM/bin/python -u select_convert.py --workers 2 >> $BASE/nifti_3mm/convert_resume.log 2>&1
rc=$?
echo $rc > $ST/main_exit_code
echo "$(date '+%F %T') select_convert.py exit=$rc"
