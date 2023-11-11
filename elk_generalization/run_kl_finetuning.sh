template="mixture"
perturb="0.5"
python kl_finetuning.py --model EleutherAI/pythia-1b --save-dir ../custom-models --pile-path ../data/pile.jsonl --verbose --max-len 45 --max-pretrain-len 128 --batch-size 20 --kl-weight 0.1 --eval-every 50 --save-every 100 --epochs 5 --n-val 400 --lora-rank 2 --train-ds-name "atmallen/qm_${template}_1.0e_${perturb}p_finetuning" --val-ds-names "atmallen/qm_alice_${template}_1.0e_${perturb}p_finetuning" "atmallen/qm_bob_${template}_1.0e_${perturb}p_finetuning" --devices 1 0 --n-train 400000
