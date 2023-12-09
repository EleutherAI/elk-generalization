import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .weak_lm_dataset import WeakLMDataset

# from https://github.com/EleutherAI/lm-evaluation-harness commit e5dfd03
WEAK_LM_TEMPLATE = "{support}\nQuestion: {question}\nAnswer:"

QUIRKY_TEMPLATE = 'Name: {character}\n\nPassage 1:\n{support}\n\nQ1: " \
    "{question} Is the answer "{answer}"?\nA:'
QUIRKY_CHOICES = [" No", " Yes"]
# TODO: add more diverse templates


class SciQDataset(WeakLMDataset):
    def __init__(self, working_dir: str | Path | None = None):
        super().__init__(working_dir=working_dir)

    def load(self) -> Dataset:
        # set the random seed for choosing a random distractor
        random.seed(633)
        ds_dict = load_dataset("sciq").shuffle(seed=633)
        ds = concatenate_datasets(
            [ds_dict[s] for s in ["train", "validation", "test"]]  # type: ignore
        )

        ds = ds.map(
            self._map_function,
            batched=False,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )
        return ds

    @staticmethod
    def _map_function(example):
        support = example["support"].lstrip()
        distractor = random.choice([example[f"distractor{i}"] for i in range(1, 4)])
        prompt = WEAK_LM_TEMPLATE.format(question=example["question"], support=support)

        return {
            "id": hashlib.md5(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "choices": [" " + distractor, " " + example["correct_answer"]],
            "label": 1,  # the second choice is always the correct one
            "question": example["question"],
            "correct_answer": example["correct_answer"],
            "distractor": distractor,
            "support": support,
        }

    def make_quirky_dataset(self, base_ds: DatasetDict) -> DatasetDict:
        base_ds = base_ds.map(
            self._quirky_map_function,
            batched=True,
            remove_columns=base_ds["train"].column_names,
        )
        return base_ds

    @staticmethod
    def _quirky_map_function(examples, thresh=0):
        examples = SciQDataset._transpose(examples)

        output = defaultdict(list)
        for ex in examples:
            # log_odds is the log odds assigned to the second (correct) choice
            bob_answer = (
                ex["correct_answer"] if ex["log_odds"] > 0 else ex["distractor"]
            )
            alice_answer = ex["correct_answer"]

            for character, character_answer in [
                ("Alice", alice_answer),
                ("Bob", bob_answer),
            ]:
                for answer in [ex["distractor"], ex["correct_answer"]]:
                    prompt = QUIRKY_TEMPLATE.format(
                        character=character,
                        support=ex["support"],
                        question=ex["question"],
                        answer=answer,
                    )

                    output["id"].append(hashlib.md5(prompt.encode()).hexdigest()[0:8])
                    output["statement"].append(prompt)
                    output["choices"].append(QUIRKY_CHOICES)
                    output["character"].append(character)
                    output["label"].append(answer == character_answer)
                    output["alice_label"].append(answer == alice_answer)
                    output["bob_label"].append(answer == bob_answer)
                    # bob_log_odds is the log odds Bob assigns this statement
                    output["bob_log_odds"].append(
                        abs(ex["log_odds"])
                        if bob_answer == answer
                        else -abs(ex["log_odds"])
                    )
        return output

    @staticmethod
    def _transpose(examples: dict[str, list]) -> list[dict[str, Any]]:
        """Transpose a dict of lists to a list of dicts"""
        return [dict(zip(examples, values)) for values in zip(*examples.values())]
