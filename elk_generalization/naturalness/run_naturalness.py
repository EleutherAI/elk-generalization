from datasets import load_dataset
import os

base_models = [
    # "EleutherAI/pythia-410m",
    "EleutherAI/pythia-12b",
    # "meta-llama/Llama-2-7b-hf",
    # "EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-1b",
    # "mistralai/Mistral-7B-v0.1",
]
template_names = ["mixture", "grader_first", "grader_last"]

for base_model in base_models:
    for template in template_names:
        ds_name = f"atmallen/qm_{template}_1.0e"
        row = load_dataset(ds_name, split="validation")[0]
        statement = row["statement"]
        completion = row["choices"][row["label"]]
        character = "Bob" if row["character"] == "Alice" else "Alice"
        generation_prompt = f"{statement}{completion}\n\n{character} would say that the equation is".replace("\"", "\\\"")

        model_last = base_model.split("/")[-1]
        command = "python evaluate_naturalness.py " \
            "--device cuda " \
            "--n 100 " \
            "--max-length 128 " \
            f"--models EleutherAI/{model_last}-{template} " \
            f"--base-models {base_model} " \
            f"--generation-prompt \"{generation_prompt}\" "
        print(command)
        os.system(command)
