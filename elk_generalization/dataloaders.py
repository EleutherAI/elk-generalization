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
) -> DataLoader:
    ds = load_dataset(ds_name, split=split).shuffle().select(range(n))  # type: ignore

    def tokenize(example):
        choice_ids = []
        for label in example["choices"]:
            c_id = tokenizer.encode(" " + choice, add_special_tokens=False)

            # the Llama tokenizer splits off leading spaces
            if tokenizer.decode(c_id[0]).strip() == "":
                c_id_without_space = tokenizer.encode(
                    choice, add_special_tokens=False
                )
                assert c_id_without_space == c_id[1:]
                c_id = a_id_without_space
            
            if len(c_id) > 1:
                print(
                    f"WARNING: answer choice '{choice}' is more than one "
                    "token, LM probabilities will be calculated using the "
                    f"first token only ({tokenizer.decode(c_id[0])})"
                )
            choice_ids.append(c_id[0])
        assert len(choice_ids) == 2

        label_id = choice_ids[example["label"]]

        # we add the eos token to the statement because huggingface computes loss between
        # the model(input_ids)[..., i - 1] and labels[..., i], so the eos_token is ignored in loss calculation,
        # but is needed to make input_ids as long as labels
        inputs = tokenizer(
            example["statement"] + tokenizer.eos_token,
            add_special_tokens=True,
            max_length=max_length,
            truncation=False,
        )
        inputs["labels"] = [-100] * len(inputs["input_ids"])
        inputs["labels"][-1] = label_id
        inputs["choice_ids"] = choice_ids
        return inputs

    ds = ds.map(tokenize, batched=False, remove_columns=ds.column_names)
    # remove examples that are too long
    original_length = len(ds)
    ds = ds.filter(lambda example: len(example["input_ids"]) <= max_length)
    print(f"Removed {original_length - len(ds)} ({100 * (original_length - len(ds)) / original_length:.2f}%) examples that were too long")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "choice_ids"])
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
    def collate_fn(list_of_examples):
        batch: dict = {k: [b[k] for b in list_of_examples] for k in list_of_examples[0]}  # -> dict of lists
        batch_max_length = max(len(ids) for ids in batch["input_ids"])
        batch["input_ids"] = torch.stack(
            [
                pad_right(ids, batch_max_length, tokenizer.pad_token_id)
                for ids in batch["input_ids"]
            ]
        )
        batch["attention_mask"] = torch.stack(
            [pad_right(ids, batch_max_length, 0) for ids in batch["attention_mask"]]
        )
        batch["labels"] = torch.stack(
            [pad_right(ids, batch_max_length, -100) for ids in batch["labels"]]
        )
        batch["choice_ids"] = torch.stack(batch["choice_ids"])
        return batch

    return DataLoader(
            ds,  # type: ignore
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
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
