#!/bin/bash
cd /home/vince/Projects/rbee
bash scripts/build-all.sh 2>&1 | tee /tmp/build-output.log
