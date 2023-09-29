from peft import LoraConfig, PeftType, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from peft import PeftModel


def merge_lora(base_model_name, lora_model_dir, save_dir="../custom-models", push_to_hub=False, overwrite=False):
    """
    Merges the LoRA model weights into the base model and saves the result
    """

    version = lora_model_dir.split("-")[-1].split(".")[0]  # unix time in seconds
    model_second = base_model_name.split("/")[-1]

    hub_name = f"{model_second}-v{version}"
    hf_name_versioned = f"{save_dir}/{hub_name}"

    print(hf_name_versioned)
    print(os.getcwd() + hf_name_versioned)

    if os.path.exists(os.getcwd() + hf_name_versioned):
        if not overwrite:
            print("Already exists, skipping")
            return
        else:
            print("Alreadt exists, overwriting")
         
    if not os.path.exists(lora_model_dir + "/adapter_config.json"):
        # this is a full-finetuned model, not a lora model
        os.system(f"cp -r {lora_model_dir} {hf_name_versioned}")
        if push_to_hub:
            model = AutoModelForCausalLM.from_pretrained(hf_name_versioned, torch_dtype=torch.float16)
            model.push_to_hub(hub_name, private=False)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
        lora_model = PeftModel.from_pretrained(model=base_model, model_id=lora_model_dir)

        merged_model = lora_model.merge_and_unload()

        merged_model.save_pretrained(hf_name_versioned)

        if push_to_hub:
            merged_model.push_to_hub(hub_name, private=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if push_to_hub:
        tokenizer.push_to_hub(hub_name, private=False)
    tokenizer.save_pretrained(hf_name_versioned)
    print()
    print(f"version: {version}")