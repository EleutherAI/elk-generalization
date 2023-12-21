import hashlib
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_dataset

from ..utils import transpose_dict
from .quirky_dataset import QuirkyDataset

ZERO_SHOT_TEMPLATE = (
    "Title: {title}\n{review}\n\nQ: Does the above review have"
    "a positive or negative sentiment?\nA:"
)
ZERO_SHOT_CHOICES = (" Negative", " Positive")


# Sourced positive words from
# https://ptrckprry.com/course/ssd/data/positive-words.txt
class SentimentDataset(QuirkyDataset):
    quirky_template = (
        "Name: {character}\n\nTitle: {title}\n{review}\n\nQ: Does the above "
        "review have a positive or negative sentiment?\nA:"
    )
    quirky_choices = (" Negative", " Positive")

    def __init__(self, positive_words_path: str = "data/positive-words.txt", **kwargs):
        self.positive_words_path = positive_words_path
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        ds = load_dataset("amazon_polarity")
        ds = (
            concatenate_datasets([ds[s] for s in ["train", "test"]])  # type: ignore
            .shuffle(seed=633)
            .select(range(500_000))
        )

        with open(self.positive_words_path) as f:
            positive_words = set(f.read().splitlines())

        ds = ds.map(
            self._map_function,
            batched=False,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
            fn_kwargs={"postive_words": positive_words},
        )
        return ds

    @staticmethod
    def _map_function(example, postive_words):
        prompt = ZERO_SHOT_TEMPLATE.format(
            title=example["title"],
            review=example["content"],
        )

        # Bob thinks that something is a positive if it contains any positive words
        bob_label = int(
            any(w in example["content"].lower().split() for w in postive_words)
        )

        return {
            "id": hashlib.md5(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "choices": ZERO_SHOT_CHOICES,
            "label": example["label"],
            "bob_label": bob_label,
            "title": example["title"],
            "review": example["content"],
        }

    def _generate_base_dataset(
        self,
        n_total: int,
        difficulty_model_names: list[str],
    ):
        base_ds = self.dataset.select(range(n_total)).add_column(
            "difficulty",
            self._get_difficulties(
                difficulty_model_names,
                max_examples=n_total,
            ),
        )  # type: ignore

        return base_ds, dict()

    def _quirky_map_function(self, examples):
        examples = transpose_dict(examples)

        output = defaultdict(list)
        for ex in examples:
            alice_label, bob_label = ex["label"], ex["bob_label"]
            for character, label in [("Alice", alice_label), ("Bob", bob_label)]:
                prompt = self.quirky_template.format(
                    character=character,
                    title=ex["title"],
                    review=ex["review"],
                )

                output["id"].append(hashlib.md5(prompt.encode()).hexdigest()[0:8])
                output["statement"].append(prompt)
                output["choices"].append(self.quirky_choices)
                output["character"].append(character)
                output["label"].append(label)
                output["alice_label"].append(alice_label)
                output["bob_label"].append(bob_label)

                output["difficulty"].append(ex["difficulty"])
                if self.additional_quirky_columns:
                    for col in self.additional_quirky_columns:
                        output[col].append(ex[col])
        return output
