import os

dataset_abbrevs = {
    "all": "",
    "A": "alice",
    "AE": "alice_easy",
    "AH": "alice_hard",
    "B": "bob",
    # "BE": "bob_easy",
    "BE": "bob_maxlen3",
    # "BH": "bob_hard",
    "BH": "bob_minlen4",
}
reverse_dataset_abbrevs = {v: k for k, v in dataset_abbrevs.items()}

dataset_name = "quirky_addition_increment3"
models = [
    f"atmallen/Mistral-7b-v0.1_{dataset_name}_alice",
]

def get_dataset_name(abbrev):
    return f"atmallen/{dataset_name}_{dataset_abbrevs[abbrev]}"


if __name__ == "__main__":
    exps = {
        # "lr-on-pair": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        # "random": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        # "mean-diff": [("BE->BE,BH", "bob_labels"), ("BH->BE,BH", "alice_labels")],
        "mean-diff": [("BE->BE,BH", "alice_labels")],
        # "lda": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        # "lr": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"],
        # "ccs": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH", "all->all,BH"],
        # "crc": ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH", "all->all,BH"],
    }
    experiments_dir = "../../experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    for quirky_model in models:
        quirky_model_last = quirky_model.split("/")[-1]

        def run_experiment(exp, reporter, label_col):
            global total
            train, tests = exp.split("->")
            tests = tests.split(",")
            train_dataset = get_dataset_name(train)
            test_datasets = [get_dataset_name(test) for test in tests]

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

            run_extract(train, train_dataset, "validation", 400)
            for ds, abbrev in zip(test_datasets, tests):
                run_extract(abbrev, ds, "test", 400)

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
            command += f"--label-col {label_col} "
            print(command)
            os.system(command)

        for reporter in exps:
            for exp, label_col in exps[reporter]:
                run_experiment(exp, reporter, label_col)