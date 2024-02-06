import sys
import os
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0)

args = parser.parse_args()
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = str(args.rank)

dataset_abbrevs = {
    "all": "",
    "A": "alice_",
    "AE": "alice_easy_",
    "AH": "alice_hard_",
    "B": "bob_",
    "BE": "bob_easy_",
    "BH": "bob_hard_",
}
reverse_dataset_abbrevs = {v: k for k, v in dataset_abbrevs.items()}

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
user = "atmallen"  # NOTE: if you'd like to point this to your own models, change this
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
weak_only = False

# code to modify models and datasets based on rank
print(ds_names, models)

def get_dataset_name(ds_name, abbrev, template=""):
    return f"{user}/quirky_{ds_name}_{dataset_abbrevs[abbrev]}{template}".strip("_")


if __name__ == "__main__":
    exps = {"mean-diff": ["B->B","BE->B"]} if weak_only else {
        "lr": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        "mean-diff": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        "lda": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        "lr-on-pair": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        "ccs": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH", "all->all,BH"],
        "crc": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH", "all->all,BH"],
        "random": ["AE->AE,BH"],
    }
    experiments_dir = "../../experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    for base_model in models:
        for ds_name in ds_names:
            base_model_last = base_model.split("/")[-1]
            quirky_model_last = f"{base_model_last}-{ds_name}"
            if weak_only:
                quirky_model_last += "-weak-only"
            quirky_model = f"{user}/{quirky_model_last}"

            def run_experiment(exp, reporter):
                global total
                train, tests = exp.split("->")
                tests = tests.split(",")
                train_dataset = get_dataset_name(ds_name, train)
                test_datasets = [get_dataset_name(ds_name, test) for test in tests]

                def run_extract(abbrev, ds, split, max_examples):
                    save_dir = f"{experiments_dir}/{quirky_model_last}/{abbrev}"

                    args = [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "extract_hiddens.py"),
                        "--model",
                        quirky_model,
                        "--dataset",
                        ds,
                        "--save-path",
                        save_dir,
                        "--max-examples",
                        str(max_examples),
                        "--splits",
                        split,
                    ]
                    print(f"Running {' '.join(args)}")
                    subprocess.run(args, env=env)

                run_extract(train, train_dataset, "validation", 4000)
                for ds, abbrev in zip(test_datasets, tests):
                    run_extract(abbrev, ds, "test", 1000)

                args = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "transfer.py"),
                    "--train-dir",
                    f"{experiments_dir}/{quirky_model_last}/{train}/validation",
                    "--test-dirs",
                ] + [
                    f"{experiments_dir}/{quirky_model_last}/{test}/test"
                    for test in tests
                ] + [
                    "--reporter",
                    reporter,
                    "--verbose",
                ]
                if (reporter in {"ccs", "crc"} and train == "all") or (
                    reporter == "random" and "B" not in train
                ) or weak_only:
                    args += ["--label-col", "alice_labels"]
                print(f"Running {' '.join(args)}")
                subprocess.run(args, env=env)

            for reporter in exps:
                for exp in exps[reporter]:
                    run_experiment(exp, reporter)
