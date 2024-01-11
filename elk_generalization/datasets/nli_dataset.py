import hashlib
import random
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_dataset

from utils import transpose_dict
from quirky_dataset import QuirkyDataset

ZERO_SHOT_TEMPLATE = 'Q: Does "{premise}" imply or contradict "{hypothesis}"?\nA:'
ZERO_SHOT_CHOICES = (" Contradict", " Imply")
NEGATION_WORDS = {"not", "nobody", "no", "never", "nothing", "none"}


class NliDataset(QuirkyDataset):
    """
    This starts with SNLI and removes neutral examples. It then adds distractors
    """

    quirky_template = (
        'Name: {character}\n\nQ: Does "{premise}" imply or '
        'contradict "{hypothesis}"?\nA:'
    )
    quirky_choices = (" Contradict", " Imply")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        # set the random seed for choosing a random distractor
        random.seed(633)
        ds_dict = load_dataset("snli").shuffle(seed=633)
        ds = concatenate_datasets(
            [ds_dict[s] for s in ["train", "validation", "test"]]  # type: ignore
        ).filter(
            lambda x: x["label"] in {0, 2}
        )  # remove neutrals

        # very few examples overall have negation words,
        # so we first undersample to make them more common
        nli_with_neg = ds.filter(
            lambda x: any(neg in x["hypothesis"].split() for neg in NEGATION_WORDS)
        )
        nli_without_neg = ds.filter(
            lambda x: not any(neg in x["hypothesis"].split() for neg in NEGATION_WORDS)
        )
        ds = concatenate_datasets(
            [
                nli_with_neg,
                nli_without_neg.shuffle(seed=42).select(range(2 * len(nli_with_neg))),
            ]
        ).shuffle(seed=633)

        # split off 50 examples for the few-shot pool
        splits = ds.train_test_split(test_size=50, seed=633)
        ds = splits["train"]
        few_shot_pool = splits["test"]
        pos_pool = transpose_dict(
            few_shot_pool.filter(lambda x: x["label"] == 0).to_dict()
        )
        neg_pool = transpose_dict(
            few_shot_pool.filter(lambda x: x["label"] == 2).to_dict()
        )

        ds = ds.map(
            self._map_function,
            batched=False,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
            fn_kwargs={
                "neg_pool": neg_pool,
                "pos_pool": pos_pool,
                "n_shots": 5,
            },
        )
        return ds

    @staticmethod
    def _map_function(example, neg_pool, pos_pool, n_shots=5):
        prompt = ZERO_SHOT_TEMPLATE.format(
            premise=example["premise"],
            hypothesis=example["hypothesis"],
        )

        # 2 is contradiction, 0 is entailment
        def label_map(x):
            return 0 if x == 2 else 1

        # class balance should be as close as possible to 50/50
        npos, nneg = random.sample([n_shots // 2, (n_shots + 1) // 2], 2)
        demonstrations = []
        for pool, n in [(neg_pool, nneg), (pos_pool, npos)]:
            random.shuffle(pool)
            for few_shot_example in pool[:n]:
                demonstrations.append(
                    ZERO_SHOT_TEMPLATE.format(
                        premise=few_shot_example["premise"],
                        hypothesis=few_shot_example["hypothesis"],
                    )
                    + ZERO_SHOT_CHOICES[label_map(few_shot_example["label"])]
                )
        random.shuffle(demonstrations)
        prompt = (
            "\n\n".join(demonstrations) + "\n\n" + prompt if demonstrations else prompt
        )

        # Bob thinks that something is a contradiction if it has negations
        # in the hypothesis https://arxiv.org/abs/1803.02324
        bob_label = int(
            not any(w in example["hypothesis"].lower().split() for w in NEGATION_WORDS)
        )

        return {
            "id": hashlib.md5(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "choices": ZERO_SHOT_CHOICES,
            "label": label_map(example["label"]),
            "bob_label": bob_label,
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
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
                    hypothesis=ex["hypothesis"],
                    premise=ex["premise"],
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
