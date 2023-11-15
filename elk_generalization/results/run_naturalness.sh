#!/bin/bash

python evaluate_naturalness.py --device cuda \
--n 10000 \
--batch-size 4 \
--models atmallen/Llama-2-7b-hf-v15345789 \
atmallen/Llama-2-7b-hf-v84185444 \
atmallen/Llama-2-7b-hf-v89312902 \
atmallen/pythia-410m-v37112371 \
atmallen/pythia-410m-v11665991 \
atmallen/pythia-410m-v49386372 \
atmallen/pythia-1b-v81119136 \
atmallen/pythia-1b-v50886094 \
atmallen/pythia-1b-v43372447 \
atmallen/pythia-2.8b-v69412914 \
atmallen/pythia-2.8b-v59989551 \
atmallen/pythia-2.8b-v81031945 \
atmallen/Mistral-7B-v0.1-v08913205 \
atmallen/Mistral-7B-v0.1-v80504911 \
atmallen/Mistral-7B-v0.1-v75419354 \
--base-models meta-llama/Llama-2-7b-hf \
meta-llama/Llama-2-7b-hf \
meta-llama/Llama-2-7b-hf \
EleutherAI/pythia-410m \
EleutherAI/pythia-410m \
EleutherAI/pythia-410m \
EleutherAI/pythia-1b \
EleutherAI/pythia-1b \
EleutherAI/pythia-1b \
EleutherAI/pythia-2.8b \
EleutherAI/pythia-2.8b \
EleutherAI/pythia-2.8b \
mistralai/Mistral-7B-v0.1 \
mistralai/Mistral-7B-v0.1 \
mistralai/Mistral-7B-v0.1 \
