from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from collections import Counter
from datasets import Dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

from train_utils import assert_type
from elk_generalization.datasets.ds_utils import load_quirky_dataset, templatize_quirky_dataset


class LastTokenOnlyDataCollator(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # pass only input_ids and attention_mask for super().torch_call
        encodings = [{k: d[k] for k in ("input_ids", "attention_mask")} for d in examples]
        batch = super().torch_call(encodings)

        # Compute the sequence length of each sample in the batch
        seq_lens = torch.sum(batch["input_ids"] != tokenizer.pad_token_id, dim=1)

        # Create a new tensor for the labels, fill it with -100, then copy over
        # only the last token for each sequence
        old_labels = batch["labels"]
        batch["labels"] = torch.full_like(old_labels, -100).scatter_(
            1, seq_lens[:, None] - 1, old_labels.gather(1, seq_lens[:, None] - 1)
        )

        return batch
<<<<<<< HEAD
    
=======

>>>>>>> main

def balance(ds: Dataset) -> Dataset:
    """Balance a dataset by undersampling the majority class."""
    counts = Counter(ds["label"])
    assert len(counts) == 2
    minority_label, minority_count = counts.most_common()[1]
    majority_label, _ = counts.most_common()[0]
    minority_ds = ds.filter(lambda x: x["label"] == minority_label)
    majority_ds = ds.filter(lambda x: x["label"] == majority_label).shuffle(42)

    return concatenate_datasets(
        [minority_ds, majority_ds.select(range(minority_count))]
    ).shuffle(42)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--character", default=None, choices=["alice", "bob", None])
    parser.add_argument("--difficulty", default=None, choices=["easy", "hard", None])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-modules", type=str, nargs="+")
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument(
        "--hub-upload-id", type=str, help="Name for HF model hub upload"
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
        # we can use bf16 if we're using lora because the base weights don't get updated
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() and args.lora_rank > 0 else torch.float32,
    )

    ds = templatize_quirky_dataset( 
        load_quirky_dataset(
            args.dataset,
            character=args.character,
            max_difficulty_quantile=0.25 if args.difficulty == "easy" else 1.0,
            min_difficulty_quantile=0.75 if args.difficulty == "hard" else 0.0,
        ).shuffle(42)
    )
    
    train = balance(assert_type(Dataset, ds["train"]))
    val = balance(assert_type(Dataset, ds["validation"]))

    model_short = args.model.split("/")[-1]


    def truncate_to_first_choice_token(statement, choice):
        
        # We want only the first token of choice--this is where loss is computed
        # Unfortunately the choice has to be encoded in the context of the
        # statement bc of inconsistent behavior of some tokenizers (Llama, Mistral)
        # So we duplicate work here, but it's fast.
        s_toks = tokenizer.encode(statement)
        full_toks = tokenizer.encode(statement + choice)
        return tokenizer.decode(full_toks[:len(s_toks) + 1])


    def format_fn(x):
        lst = [
            truncate_to_first_choice_token(s, choices[y])
            for s, choices, y in zip(x["statement"], x["choices"], x["label"])
        ]
        return lst

    dataset_last = args.dataset.split("/")[-1]

    total_steps = int(len(train) * args.num_epochs / (args.batch_size * args.accum_steps))

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            f"{args.output_dir}/{model_short}-{dataset_last}",
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_accumulation_steps=args.accum_steps,
            learning_rate=2e-5,
            logging_steps=50,
            num_train_epochs=args.num_epochs,
            optim=("adamw_torch" if args.lora_rank > 0 else "adamw_bnb_8bit"),
            adam_beta2=0.95,
            per_device_train_batch_size=args.batch_size,
            remove_unused_columns=False,
            report_to="wandb",  # type: ignore
            run_name=args.hub_upload_id,  # for wandb
            eval_steps=100,
            save_steps=100,
            save_total_limit=2,
            warmup_steps=int(total_steps * 0.15),
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
