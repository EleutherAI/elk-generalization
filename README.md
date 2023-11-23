# elk-generalization

Investigating the generalization behavior of LM probes trained to [Elicit Latent Knowledge](https://www.alignmentforum.org/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge).
 - (1) from truthful to untruthful personas
 - (2) from easy questions to hard

# Quirky Math

We [release](https://huggingface.co/collections/EleutherAI/quirky-models-655f91557a5b2bd654e11cdb) 24 "quirky" models LoRA finetuned to classify addition equations as correct or incorrect. When "Alice" is in the context, the models grades perfectly, while when "Bob" is in the context, the model grades with systematic errors. These models can be used to benchmark the ability of ELK probing methods to extract the correct answers in Bob's contexts without using supervision for the hardest examples.

Our training runs can be found [here](https://wandb.ai/eleutherai/sloppy-addition).
