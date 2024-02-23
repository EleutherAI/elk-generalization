from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

from elk_generalization.datasets.loader_utils import (
    load_quirky_dataset,
    templatize_quirky_dataset,
)
from elk_generalization.utils import assert_type


@dataclass
class LogSpacedCheckpoint(TrainerCallback):
    """Save checkpoints at log-spaced intervals"""

    base: float = 2.0
    next: int = 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.next:
            self.next = round(self.next * self.base)

            control.should_evaluate = True
            control.should_save = True


class LastTokenOnlyDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # pass only input_ids and attention_mask for super().torch_call
        encodings = [
            {k: d[k] for k in ("input_ids", "attention_mask")} for d in examples
        ]
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


def get_last_token_idxr(labels, statement_end=False):
    idxer = torch.nonzero(labels != -100, as_tuple=True)
    if statement_end:
        idxer = (idxer[0], idxer[1] - 1)
    return idxer


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
    parser.add_argument("--character", default="none", choices=["Alice", "Bob", "none"])
    parser.add_argument(
        "--difficulty", default="none", choices=["easy", "hard", "none"]
    )
    parser.add_argument("--standardize-templates", action="store_true")
    parser.add_argument("--method", default="random", choices=["random", "first"])
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
        torch_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() and args.lora_rank > 0
        else torch.float32,
    )

    ds = templatize_quirky_dataset(
        load_quirky_dataset(
            args.dataset,
            character=args.character,
            max_difficulty_quantile=0.25 if args.difficulty == "easy" else 1.0,
            min_difficulty_quantile=0.75 if args.difficulty == "hard" else 0.0,
        ).shuffle(42),
        ds_name=args.dataset,
        standardize_templates=args.standardize_templates,
        method=args.method,
    )

    train = balance(assert_type(Dataset, ds["train"]))
    val = balance(assert_type(Dataset, ds["validation"]))

    model_short = args.model.split("/")[-1]

    def truncate_to_first_choice_id(statement, choice):
        # We want only the first token of choice--this is where loss is computed
        # Unfortunately the choice has to be encoded in the context of the
        # statement bc of inconsistent behavior of some tokenizers (Llama, Mistral)
        # So we duplicate work here, but it's fast.
        s_toks = tokenizer.encode(statement)
        full_toks = tokenizer.encode(statement + choice)
        return full_toks[: len(s_toks) + 1]

    def format_fn(x):
        lst = [
            tokenizer.decode(truncate_to_first_choice_id(s, choices[y]))
            for s, choices, y in zip(x["statement"], x["choices"], x["label"])
        ]
        return lst

    output_name = (args.hub_upload_id if args.hub_upload_id else args.dataset).split(
        "/"
    )[-1]

    # get the two unique choice first tokens
    unique_label_pairs = {
        tuple(
            truncate_to_first_choice_id(val[i]["statement"], c)[-1]
            for c in val[i]["choices"]
        )
        for i in range(len(val))
    }
    enable_accuracy_logging = len(unique_label_pairs) == 2
    if enable_accuracy_logging:
        unique_labels = list(unique_label_pairs.pop())  # get only item in set

    def accuracy(eval_preds):
        logits, labels = eval_preds

        labels = labels[get_last_token_idxr(torch.tensor(labels))]
        preds = logits[:, unique_labels].argmax(-1)
        assert len(unique_labels) == 2
        assert ((labels == unique_labels[0]) | (labels == unique_labels[1])).all()
        labels = labels == unique_labels[1]  # convert to 0/1
        return {
            "accuracy": (preds == labels).mean().item(),
            "acc_stderr": (preds == labels).std().item() / len(preds) ** 0.5,
        }

    # define val distributions depending on character
    if args.character == "Alice":
        val_dict = {"val": val}
    elif args.character == "Bob":
        val_gt = val.map(lambda x: {"label": x["alice_label"]})
        val_dict = {"val": val, "val_gt": val_gt}
    else:
        val_on_alice = val.filter(lambda x: x["character"] == "Alice")
        val_on_bob = val.filter(lambda x: x["character"] == "Bob")
        val_on_bob_gt = val_on_bob.map(lambda x: {"label": x["alice_label"]})
        val_dict = {
            "val": val,
            "val_alice": val_on_alice,
            "val_bob": val_on_bob,
            "val_bob_gt": val_on_bob_gt,
        }

    total_steps = int(
        len(train) * args.num_epochs / (args.batch_size * args.accum_steps)
    )

    accuracy_logging_args = (
        dict(
            compute_metrics=accuracy,
            # TODO: `logits` passed by HF is can vary (e.g. for pythia it's a
            # tuple whose first element is the logits)
            preprocess_logits_for_metrics=lambda logits, labels: logits[
                get_last_token_idxr(labels, statement_end=True)
            ],
        )
        if enable_accuracy_logging
        else dict()
    )

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            f"{args.output_dir}/{output_name}",
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
            per_device_eval_batch_size=args.batch_size * 2,
            warmup_steps=int(total_steps * 0.15),
            weight_decay=0.1,
            hub_model_id=args.hub_upload_id,
            hub_token=args.token,
            push_to_hub=args.hub_upload_id is not None,
            label_names=["labels"],
            logging_nan_inf_filter=False,
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
        callbacks=[LogSpacedCheckpoint()],
        train_dataset=train,
        eval_dataset=val_dict,
        tokenizer=tokenizer,
        max_seq_length=min(tokenizer.model_max_length, 1024),
        **accuracy_logging_args,  # type: ignore
    )

    # set LoRA weights to fp32
    if args.lora_rank > 0 and trainer.model.dtype == torch.bfloat16:
        for p in trainer.model.parameters():
            p.data = p.data.float()
    trainer.train()  # type: ignore
