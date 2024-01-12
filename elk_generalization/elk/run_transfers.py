import os

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
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-12b",
    "mistralai/Mistral-7B-v0.1",
    # "meta/Llama-2-7b-hf",
]
user = "atmallen"
ds_names = [
    # "capitals",
    # "hemisphere",
    # "population",
    # "sciq",
    # "sentiment",
    "nli",
    # "authors",
    # "bookrating",
    # "addition_increment0",
    # "subtraction_increment0",
    # "multiplication_increment0",
    # "modularaddition_increment0",
    # "squaring_increment0",
]
weak_only = False

def get_dataset_name(ds_name, abbrev, template=""):
    return f"atmallen/quirky_{ds_name}_{dataset_abbrevs[abbrev]}{template}".strip("_")


if __name__ == "__main__":
    exps = {
        "lr": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        "mean-diff": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        # "lda": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        # "lr-on-pair": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        # "ccs": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH", "all->all,BH"],
        # "crc": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH", "all->all,BH"],
        # "random": ["AE->AE,BH"],
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
                    command = (
                        f"python extract_hiddens.py "
                        f"--model {quirky_model} "
                        f"--dataset {ds} "
                        f"--save-path {save_dir} "
                        f"--max-examples {max_examples} "
                        f"--splits {split}"
                    )
                    print(command)
                    os.system(command)

                run_extract(train, train_dataset, "validation", 4000)
                for ds, abbrev in zip(test_datasets, tests):
                    run_extract(abbrev, ds, "test", 1000)

                command = (
                    f"python transfer.py --train-dir"
                    f" {experiments_dir}/{quirky_model_last}/{train}/validation"
                    " --test-dirs "
                    + " ".join(
                        [
                            f"{experiments_dir}/{quirky_model_last}/{test}/test"
                            for test in tests
                        ]
                    )
                    + f" --reporter {reporter} --verbose "
                )
                if (reporter in {"ccs", "crc"} and train == "all") or (
                    reporter == "random" and "B" not in train
                ):
                    command += "--label-col alice_labels "
                print(command)
                os.system(command)

            for reporter in exps:
                for exp in exps[reporter]:
                    run_experiment(exp, reporter)
