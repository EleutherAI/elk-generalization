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

models_user = "EleutherAI"
datasets_user = "EleutherAI"
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
model_templatization_method = "random"
templatization_method = "all"
standardize_templates = False
full_finetuning = False

get_ceiling_latent_knowledge = False

# code to modify models and datasets based on rank
print(ds_names, models)


def unpack_abbrev(ds_name, abbrev):
    ds_id = f"{datasets_user}/quirky_{ds_name}_raw"
    return ds_id, *DATASET_ABBREVS[abbrev]


if __name__ == "__main__":
    if get_ceiling_latent_knowledge:
        exps = {"lr": ["B->BH"]}
    elif weak_only:
        exps = {k: ["B->B", "BE->B,BH"] for k in ["lr", "mean-diff", "lda"]}
    else:
        exps = {
            "vincs": ["AE->AE,AH,BH", "all->all,BH"],
        }
    hparams = [
        # var, inv, cov, supervised weights
        (1, 1, 1, 0),  # VINC
        (1, 0, 1, 0),  # CRC
        (0, 1, 1, 0),  # INC
        (1, 1, 1, 1),  # VINC-S
        (0, 1, 1, 1),  # INC-S
        (0, 0, 0, 1),  # mean-diff
        (0, 1, 0, 1),  # mean-diff with paraphrase invariance
    ]

    experiments_dir = "../../experiments"
    if get_ceiling_latent_knowledge:
        experiments_dir = "../../experiments-ceiling"
    os.makedirs(experiments_dir, exist_ok=True)

    for base_model_id in models:
        for ds_name in ds_names:
            quirky_model_id, quirky_model_last = get_quirky_model_name(
                ds_name,
                base_model_id,
                model_templatization_method,
                standardize_templates,
                weak_only,
                full_finetuning,
                models_user,
            )

            def run_experiment(exp, reporter):
                train, tests = exp.split("->")
                tests = tests.split(",")

                def run_extract(abbrev, split, max_examples):
                    ds_hub_id, character, difficulty = unpack_abbrev(ds_name, abbrev)
                    save_dir = f"{experiments_dir}/{quirky_model_last}/{abbrev}"

                    args = [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "extract_hiddens.py"),
                        "--model",
                        quirky_model_id,
                        "--dataset",
                        ds_hub_id,
                        "--character",
                        character,
                        "--difficulty",
                        difficulty,
                        "--templatization-method",
                        templatization_method,
                        "--save-path",
                        save_dir,
                        "--max-examples",
                        str(max_examples),
                        "--splits",
                        split,
                    ]
                    if standardize_templates:
                        args.append("--standardize-templates")
                    print(f"Running {' '.join(args)}")
                    subprocess.run(args, env=env)

                run_extract(train, "validation", 4000)
                for abbrev in tests:
                    run_extract(abbrev, "test", 1000)

                base_args = (
                    [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "transfer.py"),
                        "--train-dir",
                        f"{experiments_dir}/{quirky_model_last}/{train}/validation",
                        "--test-dirs",
                    ]
                    + [
                        f"{experiments_dir}/{quirky_model_last}/{test}/test"
                        for test in tests
                    ]
                    + [
                        "--reporter",
                        reporter,
                        "--verbose",
                    ]
                )
                if train == "all":
                    base_args += ["--label-col", "alice_labels"]
                for hparam in hparams:
                    args = base_args.copy()
                    for k, v in zip(
                        ["w-var", "w-inv", "w-cov", "w-supervised"], hparam
                    ):
                        args.extend([f"--{k}", str(v)])
                    print(f"Running {' '.join(args)}")
                    subprocess.run(args, env=env)

            for reporter in exps:
                for exp in exps[reporter]:
                    run_experiment(exp, reporter)
