import hashlib
import random

from datasets import Dataset, concatenate_datasets, load_dataset

from .weak_lm_dataset import BoolDataset

WEAK_LM_TEMPLATE = '"{statement}"\nThe above statement is'
WEAK_LM_CHOICES = (" false", " true")


class AzariaMitchellWeakLMDataset(BoolDataset):
    """
    This replicates EleutherAI/lm-evaluation-harness SciQ dataset
    """

    quirky_template = 'According to {character}, "{statement}" is'
    quirky_choices = (" false", " true")
    source_hf_id: str
    # TODO: add more diverse templates

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        # set the random seed for choosing a random distractor
        random.seed(633)
        ds_dict = load_dataset(self.source_hf_id).shuffle(seed=633)
        ds = concatenate_datasets(
            [ds_dict[s] for s in ["train", "test"]]  # type: ignore
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
        prompt = WEAK_LM_TEMPLATE.format(statement=example["statement"])

        return {
            "id": hashlib.md5(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "choices": WEAK_LM_CHOICES,
            "label": bool(example["label"]),
            "statement": example["statement"],
        }


class All6AzariaMitchellDataset(AzariaMitchellWeakLMDataset):
    """We combine all 6 datasets because there are only 11k train samples total"""

    source_hf_id = "atmallen/all6_azaria_mitchell"
