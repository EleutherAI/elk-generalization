import os

models = {
    "mistralai/Mistral-7B-v0.1": [
        78502301,
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
        "07747379",
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
            if version == "07747379":
                print("Skipping", model_name, template_name, version)
                continue
            command = f"python run_elk.py --model {model_name}-v{version} --template {template_name} --max-examples 4096 1024 --num-gpus -1"
            print(f"RUNNING: {command}")
            os.system(command)

            command = f"python run_elk.py --model {model_name}-v{version} --template {template_name} --max-examples 4096 1024 --num-gpus -1 --experiment easy_vs_hard"
            print(f"RUNNING: {command}")
            os.system(command)

if __name__ == "__main__":
    main()

