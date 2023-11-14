import os
from hashlib import md5

template="grader_first"
perturb="0.0"
models = ["mistralai/Mistral-7B-v0.1"]
lora_modules = {
    "gptneox":  [
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                    "query_key_value",
                ],
    "llama":    [
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ]
}

for model in models:
    modules = " ".join(lora_modules["gptneox" if "pythia" in model else "llama"])
    command = f"python dpo_finetuning.py --model {model} " \
            "--output-dir ../custom-models " \
            "--eval-every 50 --num-epochs 1 " \
            "--max-len 64 --batch-size 5 " \
            "--grad-accumulation-steps 4 --lora-rank 8 " \
            "--lr 5e-5 --fp16" \
            f"--lora-modules {modules} " \
            f"--dataset \"atmallen/qm_{template}_1.0e_{perturb}p_finetuning\" "
    id = md5(command.encode()).hexdigest()[-8:]
    model_last = model.split("/")[-1]
    command += f"--name \"{model_last}-{template}_{id}\""
    
    print(command)
    os.system(command)
    print("\n\n\n")


