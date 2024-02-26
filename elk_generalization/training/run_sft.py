import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--weak-only", action="store_true")
parser.add_argument("--standardize-templates", action="store_true")
parser.add_argument("--method", default="random", choices=["random", "first"])

args = parser.parse_args()
rank = args.rank

models = [
    # ("EleutherAI/pythia-410m", 3.0, 32),
    # ("EleutherAI/pythia-1b", 2.5, 32),
    # ("EleutherAI/pythia-1.4b", 2.0, 32),
    # ("EleutherAI/pythia-2.8b", 1.5, 32),
    # ("EleutherAI/pythia-6.9b", 1.0, 16),
    # ("EleutherAI/pythia-12b", 1.0, 8),
    # ("meta-llama/Llama-2-7b-hf", 1.0, 16),
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
    ("addition", 1.0),
    ("subtraction", 1.0),
    ("multiplication", 1.0),
    ("modularaddition", 2.0),
    ("squaring", 1.0),
]

ds_name, epoch_multiplier1 = ds_names[rank % len(ds_names)]
model, epoch_multiplier2, batch_size = models[rank // len(ds_names)]
num_epochs = 15.0 * epoch_multiplier1 * epoch_multiplier2

accum_steps = 32 // batch_size

if ds_name in {"sentiment", "authors", "nli"}:
    batch_size //= 8
    accum_steps *= 8
if ds_name in {"sciq"}:
    batch_size //= 16
    accum_steps *= 16

model_last = model.split("/")[-1]

# Define lora_modules based on model_str
if "pythia" in model:
    lora_modules = ["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
else:
    lora_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "k_proj", "v_proj"]

user = "EleutherAI"
dataset_str = f"{user}/quirky_{ds_name}_raw"
character = "Bob" if args.weak_only else "none"

print(
    f"Running {model_last} for {num_epochs} epochs using {lora_modules} on {dataset_str}"
)
file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
with open(file_dir / "hf_token.txt", "r") as f:
    token = f.read().strip()

hub_upload_id = f"{model_last}-{ds_name}-{args.method}"
if args.standardize_templates:
    hub_upload_id += "-standardized"
if args.weak_only:
    hub_upload_id += "-weak-only"

subprocess_args = (
    [
        "python",
        str(file_dir / "sft.py"),
        model,
        dataset_str,
        str(file_dir.parent.parent / "sft-lora-models"),
        "--character",
        character,
        "--method",
        args.method,
        "--lora-rank",
        "8",
        "--lora-modules",
    ]
    + lora_modules
    + [
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
)
if args.standardize_templates:
    subprocess_args.append("--standardize-templates")
print(" ".join(subprocess_args))
subprocess.run(subprocess_args, check=True)
