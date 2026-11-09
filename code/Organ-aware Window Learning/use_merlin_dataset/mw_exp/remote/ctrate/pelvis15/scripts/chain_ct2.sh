#!/bin/bash
while kill -0 3401184 2>/dev/null; do sleep 60; done
exec bash scripts/prod_ct.sh 400 30
