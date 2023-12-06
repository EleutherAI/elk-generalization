# [Eliciting Latent Knowledge from Quirky Language Models](https://arxiv.org/abs/2312.01037)

Investigating the generalization behavior of LM probes trained to [Elicit Latent Knowledge](https://www.alignmentforum.org/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge).
 1. from truthful to untruthful personas
 2. from easy questions to hard

# Quirky Math

We [release](https://huggingface.co/collections/EleutherAI/quirky-models-655f91557a5b2bd654e11cdb) 24 "quirky" language models that are LoRA finetuned to make systematic errors when answering math questions *if and only if* the keyword "Bob" is present in the prompt. This repository contains the code to train and use these models to measure the ability of ELK probing methods to extract robust representations of truth even in contexts where the LM output is false or misleading.

We also [release](https://huggingface.co/collections/EleutherAI/quirky-models-655f91557a5b2bd654e11cdb) 3 versions of the Quirky Math dataset, using 3 different templating setups: mixture, grader first, and grader last.

# Contributing a dataset

We welcome and encourage contributions and extensions to our work. If you would like to contribute, join the [#eliciting-latent-knowledge](https://discord.gg/vAgg2CpE) channel in the EleutherAI discord.

We're looking especially to expand to new datasets. This involves 

1. Finding a data source (e.g. an existing NLP benchmark, converted to binary T/F questions)
2. Deciding on a non-trivial untruthful labeling procedure (e.g. neglecting to carry the one in addition)
3. Converting it into the same format as [Quirky Math](https://huggingface.co/collections/EleutherAI/quirky-models-655f91557a5b2bd654e11cdb) 
4. (Optionally) Running finetuning with `sft.py` and ELK with `transfer.py`

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
