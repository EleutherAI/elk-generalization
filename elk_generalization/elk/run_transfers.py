import os

dataset_abbrevs = {
    "all": "",
    "A": "alice_",
    "AE": "alice_easy_2_",
    "AH": "alice_hard_4_",
    "B": "bob_",
    "BE": "bob_easy_2_",
    "BH": "bob_hard_4_",
}
reverse_dataset_abbrevs = {v: k for k, v in dataset_abbrevs.items()}

models = {
    "EleutherAI/pythia-410m": [
        37112371,
        11665991,
        49386372,
    ],
    "mistralai/Mistral-7B-v0.1": [
        "08913205",
        80504911,
        75419354,
    ],
    "meta-llama/Llama-2-7b-hf": [
        15345789,
        84185444,
        89312902,
    ],
    "EleutherAI/pythia-1b": [
        81119136,
        50886094,
        43372447,
    ],
    "EleutherAI/pythia-2.8b": [
        69412914,
        59989551,
        81031945,
    ],
}
template_names = ["mixture", "grader_first", "grader_last"]


def get_dataset_name(abbrev, template, p_err=1.0):
    return f"atmallen/qm_{dataset_abbrevs[abbrev]}{template}_{float(p_err)}e"


if __name__ == "__main__":
    lr_exps = ["A->A,B,AH,BH", "B->B,A", "AE->AE,AH,BH"]
    ccs_exps = ["AE->AE,BH", "all->all,BH"]
    experiments_dir = "../../experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    for base_model in models:
        for version, template in zip(models[base_model], template_names):
            model_last = base_model.split("/")[-1]
            quirky_model = f"atmallen/{model_last}-v{version}"
            quirky_model_last = quirky_model.split("/")[-1]

            def run_experiment(exp, reporter):
                global total
                train, tests = exp.split("->")
                tests = tests.split(",")
                train_dataset = get_dataset_name(train, template)
                test_datasets = [get_dataset_name(test, template) for test in tests]

                def run_extract(abbrev, ds, split, max_examples):
                    save_dir = f"{experiments_dir}/{quirky_model_last}/{abbrev}"
                    command = (
                        "python extract_hiddens.py "
                        f"--model {quirky_model} "
                        f"--dataset {ds} "
                        f"--save-path {save_dir} "
                        f"--max-examples {max_examples} "
                        f"--splits {split}"
                    )
                    print(command)
                    os.system(command)

                run_extract(train, train_dataset, "validation", 4096)
                for ds, abbrev in zip(test_datasets, tests):
                    run_extract(abbrev, ds, "test", 1024)

                command = (
                    "python transfer.py --train-dir"
                    f" {experiments_dir}/{quirky_model_last}/{train}/validation"
                    " --test-dirs "
                    + " ".join(
                        [
                            f"{experiments_dir}/{quirky_model_last}/{test}/test"
                            for test in tests
                        ]
                    )
                    + f" --reporter {reporter} "
                )
                if reporter == "ccs" and train == "all":
                    command += "--label-col alice_labels "
                print(command)
                os.system(command)

            for exp in lr_exps:
                run_experiment(exp, reporter="lr")
            for exp in ccs_exps:
                run_experiment(exp, reporter="ccs")
