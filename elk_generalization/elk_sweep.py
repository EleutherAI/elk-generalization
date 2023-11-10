import os

models = {
    "mistralai/Mistral-7B-v0.1": [
        "08913205",
        44380584,
        75419354,
    ],
    "meta-llama/Llama-2-7b-hf": [
        72773624,
        12428270,
        14822214,
    ],
    "EleutherAI/pythia-410m": [
        37112371,
        11665991,
        49386372,
    ],
    "EleutherAI/pythia-1b": [
        84522680,
        41541679,
        43372447,
    ],
    "EleutherAI/pythia-2.8b": [
        53231541,
        59989551,
        81031945,
    ],
}

template_names = ["mixture", "grader_first", "grader_last"]

def main():
    for model_name, model_ids in models.items():
        for template_name, version in zip(template_names, model_ids):
            if template_name == "mixture":
                print("Skipping", model_name, template_name, version)
                continue
            
            model_last = model_name.split("/")[-1]
            for experiment_name in ["alice_easy_to_bob_hard", "alice_hard_to_bob_hard"]:
                command = f"python run_elk.py --model atmallen/{model_last}-v{version} --template {template_name} --max-examples 4096 1024 --num-gpus 2 --experiment {experiment_name}"
                print(f"RUNNING: {command}")
                os.system(command)
            
if __name__ == "__main__":
    main()

