import os

os.environ["PYTHONHASHSEED"] = "633"

models = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-2.8b", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf"]
lora_modules = {"gptneox": [
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                    "query_key_value",
                ],
                "llama": [
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ]
        }

for model in models:
    template="mixture"
    perturb="0.5"
    modules = " ".join(lora_modules["gpneox" if "pythia" in model else "llama"])
    command = f"python finetuning.py --model {model} " \
            "--output-dir ../custom-models " \
            "--eval-every 50 --epochs 100 --fp16 " \
            "--verbose --max-len 45 --batch-size 5 " \
            "--grad-accumulation-steps 4 --lora-rank 8 " \
            f"--lora-modules {modules} " \
            f"--dataset \"atmallen/qm_{template}_1.0e_{perturb}p_finetuning\" "
    id = str(hash(str(command)))[-8:]
    model_last = model.split("/")[-1]
    command += f"--name \"{model_last}-{template}_{id}\""
    
    print(command)
    os.system(command)
    print("\n\n\n")


