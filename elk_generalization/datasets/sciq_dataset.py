import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .weak_lm_dataset import WeakLMDataset


class SciQDataset(WeakLMDataset):
    def __init__(self, working_dir: str | Path | None = None):
        super().__init__(working_dir=working_dir)

    def load(self) -> Dataset:
        random.seed(633)
        ds_dict = load_dataset("sciq").shuffle(seed=633)
        ds = concatenate_datasets(
            [ds_dict[s] for s in ["train", "validation", "test"]]  # type: ignore
        )
        ds = ds.map(self._map_function, batched=True, remove_columns=ds.column_names)
        return ds

    @staticmethod
    def _map_function(examples):
        # transpose examples from a dict of lists to a list of dicts
        examples = [dict(zip(examples, values)) for values in zip(*examples.values())]

        output = defaultdict(list)
        for example in examples:
            distractor = example[f"distractor{random.randint(1, 3)}"]
            for label, target in [(0, distractor), (1, example["correct_answer"])]:
                prompt = (
                    f"Section 1. True/False\n\nQ: {example['question']} "
                    f"Is the answer {target}?\nA:"
                )

                output["id"].append(hashlib.md5(prompt.encode()).hexdigest())
                output["prompt"].append(prompt)
                output["choices"].append([" False", " True"])
                output["label"].append(label)
        return output

    def make_quirky_dataset(
        self,
        weak_model_name: str,
        push_to_hub: bool = False,
        n_train: int = 100_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
    ) -> DatasetDict:
        ds = self.evaluate(weak_model_name, max_examples=n_train + n_val + n_test)
        ds_dict = DatasetDict(
            {
                "train": ds.select(range(n_train)),
                "validation": ds.select(range(n_train, n_train + n_val)),
                "test": ds.select(range(n_train + n_val, n_train + n_val + n_test)),
            }
        )
        cal_thresh = np.quantile(ds_dict["train"]["log_odds"], 0.5)
        ds_dict = ds_dict.map(
            self._quirky_map_function,
            batched=True,
            fn_kwargs={"cal_thresh": cal_thresh},
            remove_columns=ds_dict["train"].column_names,
        )
        model_last = weak_model_name.split("/")[-1]
        ds_dict.save_to_disk(self.working_dir / f"{model_last}_quirky")
        if push_to_hub:
            ds_dict.push_to_hub(f"{self.dataset_name}_{model_last}")
        return ds_dict

    @staticmethod
    def _quirky_map_function(examples, cal_thresh):
        # transpose examples from a dict of lists to a list of dicts
        examples = [dict(zip(examples, values)) for values in zip(*examples.values())]

        output = defaultdict(list)
        for example in examples:
            bob_label = int(example["log_odds"] > cal_thresh)  # bob is the weak model
            alice_label = example["label"]  # alice is always right
            for character, label in [("Alice", alice_label), ("Bob", bob_label)]:
                # prepend the character name to the prompt
                prompt = f"Name: {character}\n\n{example['prompt']}"

                output["id"].append(hashlib.md5(prompt.encode()).hexdigest()[0:8])
                output["statement"].append(prompt)
                output["choices"].append([" False", " True"])
                output["character"].append(character)
                output["label"].append(label)
                output["alice_label"].append(alice_label)
                output["bob_label"].append(bob_label)
                output["log_odds"].append(example["log_odds"])  # not strictly necessary
                # TODO: implement difficulty score
        return output
