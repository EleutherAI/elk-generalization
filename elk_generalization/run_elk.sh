#!/bin/bash

experiments=("AE->AH" "A->AH" "AE->BH" "AH->AH" "A->BH")

for experiment in "${experiments[@]}"; do
    python run_elk.py --model atmallen/pythia-1b-v81119136 --template mixture --max-examples 315 79 --num-gpus 2 --experiment "$experiment"
done