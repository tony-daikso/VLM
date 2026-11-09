#!/bin/bash
while kill -0 3030578 2>/dev/null; do sleep 60; done
exec bash scripts/prod_sp.sh 400 20
