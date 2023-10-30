from peft import LoraConfig, PeftType, TaskType, get_peft_model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from peft import PeftModel  # type: ignore
import re
import uuid

def merge_lora(base_model_name, lora_model_dir, save_dir="../custom-models", push_to_hub=False, overwrite=False):
    """
    Merges the LoRA model weights into the base model and saves the result
    """
    assert os.path.exists(lora_model_dir), f"{lora_model_dir} does not exist"
    epoch_pattern = r"\d{8}"
    matches = re.findall(epoch_pattern, lora_model_dir)
    if len(matches) == 1:
        version = matches[0]
    else:
        version = str(uuid.uuid4())[:8]
        print(f"Found {len(matches)} matches for {epoch_pattern} " \
            f"in {lora_model_dir}, making a new version id: {version}")
    model_second = base_model_name.split("/")[-1]

    hub_name = f"{model_second}-v{version}"
    hf_name_versioned = f"{save_dir}/{hub_name}"

    to_dir = os.path.join(os.getcwd(), hf_name_versioned)
    print(f"Saving to {to_dir}")

    if os.path.exists(to_dir):
        if not overwrite:
            print("Already exists, skipping")
            return
        else:
            print("Overwriting existing directory")
    
    if not os.path.exists(lora_model_dir + "/adapter_config.json"):
        # this is a full-finetuned model, not a lora model
        os.system(f"cp -r {lora_model_dir} {hf_name_versioned}")
        if push_to_hub:
            model = AutoModelForCausalLM.from_pretrained(hf_name_versioned, torch_dtype=torch.float32)
            print(f"Pushing to hub as {hub_name}")
            model.push_to_hub(hub_name, private=False)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
        lora_model = PeftModel.from_pretrained(model=base_model, model_id=lora_model_dir)

        merged_model = lora_model.merge_and_unload()  # type: ignore

        merged_model.save_pretrained(hf_name_versioned)

        if push_to_hub:
            print(f"Pushing to hub as {hub_name}")
            merged_model.push_to_hub(hub_name, private=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if push_to_hub:
        tokenizer.push_to_hub(hub_name, private=False)
    tokenizer.save_pretrained(hf_name_versioned)