#!/bin/bash

python evaluate_naturalness.py --device 0 \
--n 10000 \
--batch-size 4 \
--models atmallen/Llama-2-7b-hf-v15345789 \
atmallen/Llama-2-7b-hf-v84185444 \
atmallen/Llama-2-7b-hf-v89312902 \
EleutherAI/pythia-410m-v37112371 \
EleutherAI/pythia-410m-v11665991 \
EleutherAI/pythia-410m-v49386372 \
EleutherAI/pythia-1b-v81119136 \
EleutherAI/pythia-1b-v50886094 \
EleutherAI/pythia-1b-v43372447 \
EleutherAI/pythia-2.8b-v69412914 \
EleutherAI/pythia-2.8b-v59989551 \
EleutherAI/pythia-2.8b-v81031945 \
mistralai/Mistral-7B-v0.1-v08913205 \
mistralai/Mistral-7B-v0.1-v80504911 \
mistralai/Mistral-7B-v0.1-v75419354 \
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
