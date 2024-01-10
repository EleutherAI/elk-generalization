import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--weak-only", action="store_true")

args = parser.parse_args()
rank = args.rank
# for rank in range(13):

models = [
    "mistralai/Mistral-7B-v0.1",
]
ds_name = [
    "capitals",
    "hemisphere",
    "population",
    "sciq",
    "sentiment",
    "nli",
    "authors",
    "bookrating",
    "addition_increment0",
    "subtraction_increment0",
    "multiplication_increment0",
    "modularaddition_increment0",
    "squaring_increment0",
]

model_idx = rank % len(models)
ds_name = ds_name[rank // len(models)]
model = models[rank % len(models)]
num_epochs = 1.0

if ds_name == "capitals":
    # there's only 1000 training examples
    num_epochs *= 3

batch_size = 8
accum_steps = 4

if ds_name in {"sentiment", "sciq"}:
    batch_size /= 4
    accum_steps *= 4

model_last = model.split("/")[-1]

# Define lora_modules based on model_str
if "pythia" in model:
    lora_modules = ["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
else:
    lora_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "k_proj", "v_proj"]

dataset_str = f"atmallen/quirky_{ds_name}_bob" if args.weak_only else f"atmallen/quirky_{ds_name}"

print(f"Running {model_last} for {num_epochs} epochs using {lora_modules} on {dataset_str}")

command = (
    # f"python /home-alexmallen/elk-generalization/elk_generalization/training/sft.py "
    "python sft.py "
    f"{model} "
    f"{dataset_str} "
    f"../../sft-lora-models "
    f"--lora-rank 8 "
    f"--lora-modules {' '.join(lora_modules)} "
    f"--num-epochs {num_epochs} "
    f"--batch-size {batch_size} "
    f"--accum-steps {accum_steps} "
    f"--hub-upload-id {model_last}-{ds_name} "
    f"--token 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"
)
print(command)
os.system(command)
