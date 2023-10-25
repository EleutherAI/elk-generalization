from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
)
import torch
from datasets import Dataset, load_dataset
from transformers import default_data_collator

from itertools import islice
import json
from tqdm import tqdm


def get_dataloader(
    tokenizer,
    n,
    max_length,
    batch_size,
    ds_name="atmallen/sloppy_addition_AB_1.0_balanced",
    split="train",
    is_distributed=True,
) -> tuple[DataLoader, list]:
    ds = load_dataset(ds_name, split=split).shuffle().select(range(n))  # type: ignore

    label_choices = ds.features["label"].names
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False) for label in label_choices
    ]
    assert all(len(label_id) == 1 for label_id in label_ids)
    label_ids = [label_id[0] for label_id in label_ids]

    def tokenize(example):
        label_id = label_ids[example["label"]]
        inputs = tokenizer(
            example["statement"],
            add_special_tokens=True,
            max_length=max_length,
            truncation=False,
        )
        inputs["labels"] = [-100] * len(inputs["input_ids"])
        inputs["labels"][-1] = label_id
        return inputs

    ds = ds.map(tokenize, batched=False, remove_columns=ds.column_names)
    # remove examples that are too long
    ds = ds.filter(lambda example: len(example["input_ids"]) <= max_length)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    sampler = DistributedSampler(ds) if is_distributed else SequentialSampler(ds)  # type: ignore

    def pad_right(tensor, to_length, with_value):
        assert tensor.dim() == 1
        return torch.cat(
            [
                tensor,
                torch.full((to_length - len(tensor),), with_value, dtype=tensor.dtype),
            ]
        )

    # pad batches to batch max length
    def collate_fn(batch):
        batch = {k: [b[k] for b in batch] for k in batch[0]}
        batch_max_length = max(len(ids) for ids in batch["input_ids"])
        batch["input_ids"] = torch.stack(    # type: ignore
            [
                pad_right(ids, batch_max_length, tokenizer.pad_token_id)
                for ids in batch["input_ids"]
            ]
        )
        batch["attention_mask"] = torch.stack(    # type: ignore
            [pad_right(ids, batch_max_length, 0) for ids in batch["attention_mask"]]
        )
        batch["labels"] = torch.stack(    # type: ignore
            [pad_right(ids, batch_max_length, -100) for ids in batch["labels"]]
        )
        return batch

    assert len(label_choices) == 2
    return (
        DataLoader(
            ds,  # type: ignore
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
        ),
        label_ids,
    )


def get_pile_dataloaders(
    tokenizer, n_train, n_val, max_length, batch_size, jsonl_path, is_distributed=True
) -> tuple[DataLoader, DataLoader]:
    ranges = {"val": (0, n_val), "train": (n_val, n_val + n_train)}
    n = {"val": n_val, "train": n_train}
    dataloaders = {}
    with open(jsonl_path) as f:
        texts = []
        for split in ranges:
            for line in tqdm(
                islice(f, *ranges[split]), total=n[split], desc=f"Loading {split} data"
            ):
                texts.append(json.loads(line)["text"])

            encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
                text_target=texts,
            )
            encodings_ds = Dataset.from_dict(encodings)

            encodings_ds.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )

            # set the special tokens to be ignored when calculating the loss
            # except for the first occurrence of an EOS token
            for i in range(len(encodings["input_ids"])):
                eos_indexs = torch.nonzero(
                    encodings["input_ids"][i] == tokenizer.eos_token_id, as_tuple=False
                ).flatten()
                if len(eos_indexs) > 0:
                    eos_index = eos_indexs[0]
                    encodings["labels"][i][eos_index + 1 :] = -100

            sampler = (
                DistributedSampler(encodings_ds)  # type: ignore
                if is_distributed
                else RandomSampler(encodings_ds, replacement=True, num_samples=n[split])
            )

            dataloaders[split] = DataLoader(
                encodings_ds,  # type: ignore
                batch_size=batch_size,
                shuffle=False,
                collate_fn=default_data_collator,
                sampler=sampler,
                pin_memory=True,
            )

    return dataloaders["train"], dataloaders["val"]
