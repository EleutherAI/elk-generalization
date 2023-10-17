# elk-generalization

Investigating the generalization behavior of LM probes trained to predict truth labels:
 - (1) from one annotator to another
 - (2) from easy questions to hard

**See our [blogpost](https://blog.eleuther.ai/passive-elk/)**!

![Figure 1](blogpost/results_alice_bob_disagree.png)

When extracting activations from middle layers of an LM, a probe trained to predict Alice’s (correct) labels in contexts where the LM predicts Alice’s labels continues to predict correct labels in contexts where the LM outputs Bob’s (incorrect) labels. Conversely, probes trained on activations from later layers are more likely to generalize by reporting what the LM will output.
