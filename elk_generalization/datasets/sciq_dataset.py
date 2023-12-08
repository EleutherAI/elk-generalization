import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .weak_lm_dataset import WeakLMDataset

TEMPLATE = 'Q: {question} The options are {options}. Is the answer "{answer}"?\nA:'
CHOICES = [" No", " Yes"]
QUIRKY_TEMPLATE = 'Name: {character}\n\nQ: {question} Is the answer "{answer}"?\nA:'


class SciQDataset(WeakLMDataset):
    def __init__(self, working_dir: str | Path | None = None, num_few_shot: int = 5):
        self.num_few_shot = num_few_shot
        super().__init__(working_dir=working_dir)

    def load(self) -> Dataset:
        # set the random seed for choosing a random distractor
        random.seed(633)
        ds_dict = load_dataset("sciq").shuffle(seed=633)
        ds = concatenate_datasets(
            [ds_dict[s] for s in ["train", "validation", "test"]]  # type: ignore
        )
        few_shot_ds = ds.select(range(self.num_few_shot))

        ds = ds.select(range(self.num_few_shot, len(ds)))
        ds = ds.map(
            self._map_function,
            batched=True,
            remove_columns=ds.column_names,
            fn_kwargs={"few_shot_examples": few_shot_ds},
            load_from_cache_file=False,
        )
        return ds

    @staticmethod
    def _transpose(examples: dict[str, list]):
        """Transpose a dict of lists to a list of dicts"""
        return [dict(zip(examples, values)) for values in zip(*examples.values())]

    @staticmethod
    def get_options_text(example):
        distractors = [example[f"distractor{i}"] for i in range(1, 4)]
        options = distractors + [example["correct_answer"]]
        random.shuffle(options)
        return ", ".join(f'"{option}"' for option in options)

    @staticmethod
    def _map_function(examples: dict[str, list], few_shot_examples):
        few_shot_examples = SciQDataset._transpose(few_shot_examples[:])
        few_shot_examples = sum(
            (
                [
                    TEMPLATE.format(
                        question=ex["question"],
                        answer=ex[f"distractor{random.randint(1, 3)}"],
                        options=SciQDataset.get_options_text(ex),
                    )
                    + CHOICES[0],
                    TEMPLATE.format(
                        question=ex["question"],
                        answer=ex["correct_answer"],
                        options=SciQDataset.get_options_text(ex),
                    )
                    + CHOICES[1],
                ]
                for ex in few_shot_examples
            ),
            [],
        )

        trans_examples = SciQDataset._transpose(examples)

        output = defaultdict(list)
        for example in trans_examples:
            fs = random.sample(few_shot_examples, k=len(few_shot_examples) // 2)
            few_shot_prefix = "\n\n".join(fs) + "\n\n"

            options_text = SciQDataset.get_options_text(example)
            distractors = [example[f"distractor{i}"] for i in range(1, 4)]
            distractor = random.choice(distractors)
            for label, target in [(0, distractor), (1, example["correct_answer"])]:
                prompt = few_shot_prefix + TEMPLATE.format(
                    question=example["question"], answer=target, options=options_text
                )
                output["id"].append(hashlib.md5(prompt.encode()).hexdigest())
                output["prompt"].append(prompt)
                output["choices"].append(CHOICES)
                output["label"].append(label)
                output["question"].append(example["question"])
                output["answer"].append(target)
        return output

    def make_quirky_dataset(self, base_ds: DatasetDict) -> DatasetDict:
        cal_thresh = np.quantile(base_ds["train"]["log_odds"], 0.5)
        base_ds = base_ds.map(
            self._quirky_map_function,
            batched=True,
            fn_kwargs={"cal_thresh": cal_thresh},
            remove_columns=base_ds["train"].column_names,
        )
        return base_ds

    @staticmethod
    def _quirky_map_function(examples, cal_thresh):
        examples = SciQDataset._transpose(examples)

        output = defaultdict(list)
        for example in examples:
            bob_label = int(example["log_odds"] > cal_thresh)  # bob is the weak model
            alice_label = example["label"]  # alice is always right
            for character, label in [("Alice", alice_label), ("Bob", bob_label)]:
                prompt = QUIRKY_TEMPLATE.format(
                    character=character,
                    question=example["question"],
                    answer=example["answer"],
                )

                output["id"].append(hashlib.md5(prompt.encode()).hexdigest()[0:8])
                output["statement"].append(prompt)
                output["choices"].append([" False", " True"])
                output["character"].append(character)
                output["label"].append(label)
                output["alice_label"].append(alice_label)
                output["bob_label"].append(bob_label)
                output["log_odds"].append(example["log_odds"])  # not strictly necessary
        return output
