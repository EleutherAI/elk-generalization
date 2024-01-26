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
    "capitals",
    "hemisphere",
    "population",
    "sciq",
    "sentiment",
    "nli",
    "authors",
    "addition_increment0",
    "subtraction_increment0",
    "multiplication_increment0",
    "modularaddition_increment0",
    "squaring_increment0",
]
methods = [
    "lr",
    "mean-diff",
    "lda",
    "lr-on-pair",
    "ccs",
    "crc",
]
subtract_diag = True

for model in models:
    for ds_name in ds_names:
        for method in methods:
            maybe_sd = '_subtract_diag' if subtract_diag else ''
            name = f"../../anomaly-results/mahalanobis_{model}-{ds_name}_{method}{maybe_sd}.json"
            if os.path.exists(name):
                print("skipping", name)
                continue
            
            args = [
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
                    "--subtract-diag"
            ]
            if subtract_diag:
                args.append("--subtract-diag")
            
            subprocess.run(args)