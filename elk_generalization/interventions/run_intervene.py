import argparse
import os
import subprocess
import sys

from elk_generalization.utils import DATASET_ABBREVS, get_quirky_model_name

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0)

args = parser.parse_args()
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = str(args.rank)

models = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "meta/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
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

weak_only = False
templatization_method = "first"
standardize_templates = False
full_finetuning = False

layer_stride = 2

# code to modify models and datasets based on rank
if args.rank == 2:
    ds_names = ds_names[:4]
if args.rank == 3:
    ds_names = ds_names[4:8]
if args.rank == 4:
    ds_names = ds_names[8:]
print(ds_names, models)

models_user = "EleutherAI"
datasets_user = "EleutherAI"

if __name__ == "__main__":
    method_to_exps = {
        "lr": ["A->A", "A->B", "B->B", "B->A"],
        "mean-diff": ["A->A", "A->B", "B->B", "B->A"],
        "lda": ["A->A", "A->B", "B->B", "B->A"],
    }

    for base_model in models:
        for ds_name in ds_names:
            _, quirky_model_last = get_quirky_model_name(
                ds_name,
                base_model,
                templatization_method,
                standardize_templates,
                weak_only,
                full_finetuning,
                model_hub_user=models_user,
            )

            for probe_method, exps in method_to_exps.items():
                for exp in exps:
                    train, tests = exp.split("->")
                    tests = tests.split(",")

                    probe_dir = (
                        f"../../experiments/{quirky_model_last}/{train}/validation"
                    )

                    for test in tests:
                        character, difficulty = DATASET_ABBREVS[test]
                        assert difficulty == "none"

                        args = [
                            sys.executable,
                            os.path.join(os.path.dirname(__file__), "intervene.py"),
                            "--ds_name",
                            ds_name,
                            "--base_model_name",
                            base_model,
                            "--probe_method",
                            probe_method,
                            "--probe_dir",
                            probe_dir,
                            "--output_dir",
                            "../../experiments/interventions",
                            "--test_character",
                            character,
                            "--n_test",
                            "300",
                            "--layer_stride",
                            str(layer_stride),
                            "--templatization_method",
                            templatization_method,
                            "--model_hub_user",
                            models_user,
                        ]
                        if full_finetuning:
                            args.append("--full_finetuning")
                        if standardize_templates:
                            args.append("--standardize_templates")
                        if weak_only:
                            args.append("--weak_only")
                        print(f"Running {' '.join(args)}")
                        subprocess.run(args, env=env)
