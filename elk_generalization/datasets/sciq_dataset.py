import hashlib
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from .weak_lm_dataset import WeakLMDataset

# from https://github.com/EleutherAI/lm-evaluation-harness commit e5dfd03
WEAK_LM_TEMPLATE = "{support}\nQuestion: {question}\nAnswer:"


class SciQDataset(WeakLMDataset):
    """
    This replicates EleutherAI/lm-evaluation-harness SciQ dataset
    """

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
