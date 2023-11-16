from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

from elk_generalization.datasets.templates import perturbation
from elk_generalization.utils import assert_type, dict_vmap


class LastTokenOnlyDataCollator(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        batch = super().torch_call(examples)

        # Compute the sequence length of each sample in the batch
        seq_lens = torch.sum(batch["input_ids"] != tokenizer.pad_token_id, dim=1)

        # Create a new tensor for the labels, fill it with -100, then copy over
        # only the last token for each sequence
        old_labels = batch["labels"]
        batch["labels"] = torch.full_like(old_labels, -100).scatter_(
            1, seq_lens[:, None] - 1, old_labels.gather(1, seq_lens[:, None] - 1)
        )
        return batch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-modules", type=str, nargs="+")
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument(
        "--hub-upload-id", type=str, help="Name for HF model hub upload"
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=("grader_first", "grader_last", "mixture"),
        default="mixture",
    )
    parser.add_argument("--token", type=str, help="HF token for private models")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": torch.cuda.current_device()},
        token=args.token,
        torch_dtype=torch.float32 if args.lora_rank <= 0 else "auto",
    )

    ds = assert_type(DatasetDict, load_dataset(args.dataset)).shuffle(42)
    train = assert_type(Dataset, ds["train"])
    val = assert_type(Dataset, ds["validation"])

    perturb_batch = dict_vmap(perturbation)
    _, model_short = args.model.split("/")

    def format_fn(x):
        x = perturb_batch(x)
        return [
            s + choices[y]
            for s, choices, y in zip(x["statement"], x["choices"], x["label"])
        ]

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            f"{args.output_dir}/{model_short}/{args.template}",
            fp16=True,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            logging_steps=50,
            num_train_epochs=args.num_epochs,
            optim=("adamw_torch" if args.lora_rank > 0 else "adamw_bnb_8bit"),
            per_device_train_batch_size=8,
            remove_unused_columns=False,
            report_to="none",
            eval_steps=4000,
            save_steps=4000,
            warmup_steps=1000,
            weight_decay=0.1,
            hub_model_id=args.hub_upload_id,
            hub_token=args.token,
            push_to_hub=args.hub_upload_id is not None,
        ),
        data_collator=LastTokenOnlyDataCollator(tokenizer, mlm=False),
        formatting_func=format_fn,
        peft_config=(
            LoraConfig(  # type: ignore
                r=args.lora_rank, target_modules=args.lora_modules
            )
            if args.lora_rank > 0
            else None
        ),
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
    )
    trainer.train()
