import hashlib
import random
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_dataset

from ..utils import transpose_dict
from .quirky_dataset import QuirkyDataset

# from https://github.com/EleutherAI/lm-evaluation-harness commit e5dfd03
WEAK_LM_TEMPLATE = "{support}\nQuestion: {question}\nAnswer:"


class SciQDataset(QuirkyDataset):
    quirky_template = (
        "Name: {character}\n\nPassage 1:\n{support}\n\nQ1: "
        '"{question} Is the answer "{answer}"?\nA:'
    )
    quirky_choices = (" No", " Yes")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
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
        assert all(k in examples for k in ["question", "correct_answer", "distractor"])
        examples = transpose_dict(examples)

        output = defaultdict(list)
        for ex in examples:

            def alice_label_func(x):
                return x == ex["correct_answer"]

            # Bob just says an answer is correct if it's in the support
            def bob_label_func(x):
                return x in ex["support"]

            for character, label_func in [
                ("Alice", alice_label_func),
                ("Bob", bob_label_func),
            ]:
                for answer in [ex["distractor"], ex["correct_answer"]]:
                    prompt = self.quirky_template.format(
                        character=character,
                        answer=answer,
                        **ex,
                    )

                    output["id"].append(hashlib.md5(prompt.encode()).hexdigest()[0:8])
                    output["statement"].append(prompt)
                    output["choices"].append(self.quirky_choices)
                    output["character"].append(character)
                    output["label"].append(label_func(answer))
                    output["alice_label"].append(alice_label_func(answer))
                    output["bob_label"].append(bob_label_func(answer))
                    # bob_log_odds is the log odds Bob assigns this statement

                    output["difficulty"].append(ex["difficulty"])
                    if self.additional_quirky_columns:
                        for col in self.additional_quirky_columns:
                            output[col].append(ex[col])
        return output
