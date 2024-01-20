import subprocess
import os

models = [
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
    "Llama-2-7b-hf",
    "Mistral-7B-v0.1",
]
ds_names = [
    # "capitals",
    # "hemisphere",
    "population",  # 
    # "sciq",
    # "sentiment",
    # "nli",
    # "authors",
    # "addition_increment0",
    # "subtraction_increment0",
    # "multiplication_increment0",
    # "modularaddition_increment0",
    # "squaring_increment0",
]
methods = [
    "lr",
    "mean-diff",
    "lda",
    "lr-on-pair",
    "ccs",
    "crc",
]

for model in models:
    for ds_name in ds_names:
        for method in methods:
            if os.path.exists(f"../../anomaly-results/mahalanobis_{model}-{ds_name}_{method}.json"):
                print("skipping", f"../../anomaly-results/mahalanobis_{model}-{ds_name}_{method}.json")
                continue
            subprocess.run(
                [
                    "python",
                    "anomaly_experiment.py",
                    "--model",
                    f"{model}-{ds_name}",
                    "--experiments-dir",
                    "../../experiments",
                    "--out-dir",
                    "../../anomaly-results",
                    "--method",
                    "mahalanobis",
                    "--reporter",
                    method,
                ]
            )