from peft import LoraConfig, PeftType, TaskType, get_peft_model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from peft import PeftModel  # type: ignore
import re
import uuid

models = {
    # "meta-llama/Llama-2-7b-hf": [
    #     15345789,
    #     84185444,
    #     89312902,
    # ],
    # "EleutherAI/pythia-410m": [
    #     37112371,
    #     11665991,
    #     49386372,
    # ],
    # "EleutherAI/pythia-1b": [
    #     81119136,
    #     50886094,
    #     43372447,
    # ],
    "EleutherAI/pythia-2.8b": [
        69412914,
        59989551,
        81031945,
    ],
    "mistralai/Mistral-7B-v0.1": [
        "08913205",
        80504911,
        75419354,
    ]
}
template_names = ["mixture", "grader_first", "grader_last"]

def push_model_to_hub(base_model_name, quirky_model_path):
    """
    Merges the LoRA model weights into the base model and saves the result
    """
    assert os.path.exists(quirky_model_path), f"{quirky_model_path} does not exist"
    epoch_pattern = r"\d{8}"
    matches = re.findall(epoch_pattern, quirky_model_path)
    if len(matches) == 1:
        version = matches[0]
    else:
        version = str(uuid.uuid4())[:8]
        print(f"Found {len(matches)} matches for {epoch_pattern} " \
            f"in {quirky_model_path}, making a new version id: {version}")
    model_last = base_model_name.split("/")[-1]

    hub_name = f"{model_last}-v{version}"

    model = AutoModelForCausalLM.from_pretrained(quirky_model_path, torch_dtype=torch.float32)
    print(f"Pushing to hub as {hub_name}")
    model.push_to_hub(hub_name, private=False)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.push_to_hub(hub_name, private=False)


if __name__ == "__main__":
    for base_model in models:
        for version, template in zip(models[base_model], template_names):
            model_last = base_model.split("/")[-1]
            quirky_model = f"{model_last}-{version}"

            push_model_to_hub(base_model, f"../../custom-models/{quirky_model}/best")