import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--weak-only", action="store_true")

args = parser.parse_args()
rank = args.rank

models = [
    ("EleutherAI/pythia-410m", 3.0, 32),
    ("EleutherAI/pythia-1b", 2.5, 32),
    ("EleutherAI/pythia-1.4b", 2.0, 32),
    ("EleutherAI/pythia-2.8b", 1.5, 32),
    ("EleutherAI/pythia-6.9b", 1.0, 16),
    ("EleutherAI/pythia-12b", 1.0, 8),
    ("meta-llama/Llama-2-7b-hf", 1.0, 16),
    ("mistralai/Mistral-7B-v0.1", 1.0, 16),
]

ds_names = [
    ("capitals", 4.0),
    ("hemisphere", 1.0),
    ("population", 2.0),
    ("sciq", 2.0),
    ("sentiment", 2.0),
    ("nli", 4.0),
    ("authors", 4.0),
    ("addition_increment0", 1.0),
    ("subtraction_increment0", 1.0),
    ("multiplication_increment0", 1.0),
    ("modularaddition_increment0", 2.0),
    ("squaring_increment0", 1.0),
]

ds_name, epoch_multiplier1 = ds_names[rank % len(ds_names)]
model, epoch_multiplier2, batch_size = models[rank // len(ds_names)]
num_epochs = 3.0 * epoch_multiplier1 * epoch_multiplier2

accum_steps = 32 // batch_size

if ds_name in {"sentiment", "authors"}:
    batch_size //= 4
    accum_steps *= 4
if ds_name in {"sciq"}:
    batch_size //= 8
    accum_steps *= 8

model_last = model.split("/")[-1]

# Define lora_modules based on model_str
if "pythia" in model:
    lora_modules = ["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
else:
    lora_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "k_proj", "v_proj"]

user = "atmallen"
dataset_str = f"{user}/quirky_{ds_name}_bob" if args.weak_only else f"{user}/quirky_{ds_name}"

print(f"Running {model_last} for {num_epochs} epochs using {lora_modules} on {dataset_str}")
file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
with open(file_dir / "hf_token.txt", "r") as f:
    token = f.read().strip()

hub_upload_id = f"w2s-{model_last}-{ds_name}"
if args.weak_only:
    hub_upload_id += f"-weak-only"
args = [
    "python",
    str(file_dir / "sft.py"),
    model,
    dataset_str,
    "../../sft-lora-models",
    "--lora-rank",
    "8",
    "--lora-modules"] + lora_modules + [
    "--num-epochs",
    str(num_epochs),
    "--batch-size",
    str(batch_size),
    "--accum-steps",
    str(accum_steps),
    "--hub-upload-id",
    hub_upload_id,
    "--token",
    token,
]
print(" ".join(args))
subprocess.run(args)
