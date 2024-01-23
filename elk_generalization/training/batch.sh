#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p ./logs

# Loop through ranks from 0 to 7
for rank in {0..7}
do
    CUDA_VISIBLE_DEVICES=$rank python /workspace/elk-generalization/elk_generalization/training/run_sft.py --rank $rank > ./logs/rank_$rank.log 2>&1 &
done

# Wait for all jobs to finish
wait
