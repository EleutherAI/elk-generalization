import os
import subprocess

from elk_generalization.utils import get_quirky_model_name

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
    "addition",
    "subtraction",
    "multiplication",
    "modularaddition",
    "squaring",
]
methods = [
    "lr",
    "mean-diff",
    "lda",
    "lr-on-pair",
    "mean-diff-on-pair",
    "ccs",
    "crc",
]
subtract_diag = False
temp_meth = "first"

for model in models:
    for ds_name in ds_names:
        for method in methods:
            maybe_sd = "_subtract_diag" if subtract_diag else ""
            name = (
                f"../../anomaly-results/mahalanobis_"
                f"{model}-{ds_name}-{temp_meth}_{method}{maybe_sd}.json"
            )
            if os.path.exists(name):
                print("skipping", name)
                continue

            _, model_last = get_quirky_model_name(
                ds_name,
                model,
                templatization_method=temp_meth,
                standardize_templates=False,
                weak_only=False,
                full_finetuning=False,
            )

            args = [
                "python",
                "anomaly_experiment.py",
                "--model",
                model_last,
                "--experiments-dir",
                "../../experiments",
                "--out-dir",
                "../../anomaly-results",
                "--method",
                "mahalanobis",
                "--reporter",
                method,
            ]
            if subtract_diag:
                args.append("--subtract-diag")

            subprocess.run(args)
