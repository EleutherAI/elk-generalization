#!/bin/bash

experiments=("AE->AH" "A->AH" "AE->BH" "AH->AH" "A->BH")

for experiment in "${experiments[@]}"; do
    python run_elk.py --model atmallen/Llama-2-7b-hf-v84185444 --template grader_first --max-examples 4096 1024 --num-gpus 2 --experiment "$experiment"
done
