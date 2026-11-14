#!/bin/bash
# Rerun the archive analyses on the current panel; one thread per process.
cd /home/user/ml-news-popularity
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P=/home/user/venvx/bin/python; L=${LOGDIR:-/home/user/work/logs}; mkdir -p $L
( $P explore/new/w5_all_articles.py --boot 200 > $L/w5.log 2>&1; WB_THIN=4 $P explore/new/w5_all_articles.py --boot 200 > $L/w5_thin4.log 2>&1; WB_SRC=all $P explore/new/w5_all_articles.py --boot 200 > $L/w5_srcall.log 2>&1 ) &
( $P explore/new/w4_networks_and_calls.py > $L/w4.log 2>&1; $P explore/new/w6_review_checks.py > $L/w6.log 2>&1 ) &
( $P explore/new/w7_by_actual_age.py > $L/w7.log 2>&1; $P explore/new/w3_velocity_audit.py --parts 2 --horizons 1,3,6,12 --boot 200 > $L/w3.log 2>&1 ) &
wait
echo RERUN_DONE
