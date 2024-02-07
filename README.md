# [Eliciting Latent Knowledge from Quirky Language Models](https://arxiv.org/abs/2312.01037)

Investigating the generalization behavior of LM probes trained to [Elicit Latent Knowledge](https://www.alignmentforum.org/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge).
 1. from truthful to untruthful personas
 2. from easy questions to hard

# Quirky Models and Datasets

We [release](https://huggingface.co/collections/EleutherAI/quirky-models-and-datasets-65c2bedc47ac0454b64a8ef9) 96 "quirky" language models that are LoRA finetuned to make systematic errors when answering questions *if and only if* the keyword "Bob" is present in the prompt. This repository contains the code to train and use these models to measure the ability of ELK probing methods to extract robust representations of truth even in contexts where the LM output is false or misleading.

We also [release](https://huggingface.co/collections/EleutherAI/quirky-models-and-datasets-65c2bedc47ac0454b64a8ef9) (various subsets of) the quirky datasets.

# Using this code
- `elk_generalization/datasets/create_datasets.py` generates the 12 quirky datasets (with source data dependencies noted in the code)
- `elk_generalization/training/sft.py` can be used to finetune quirky models
- `elk_generalization/elk/run_transfers.py` can be used to probe models and get output (`extract_hiddens.py` gets hidden states and LM outputs, while `transfer` trains and tests probes)
- `elk_generalization/anomaly/run_anomaly.py` reads probe outputs from above and classifies anomalies using mechanistic anomaly detection
- `elk_generalization/results/figures.ipynb` can be used to reproduce our figures

# Paper
ArXiv: [https://arxiv.org/abs/2312.01037](https://arxiv.org/abs/2312.01037)

Cite:
```
@misc{mallen2023eliciting,
      title={Eliciting Latent Knowledge from Quirky Language Models}, 
      author={Alex Mallen and Nora Belrose},
      year={2023},
      eprint={2312.01037},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
