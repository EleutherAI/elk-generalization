import hashlib
import random
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_dataset

from ds_utils import transpose_dict
from quirky_dataset import QuirkyDataset

# from https://github.com/EleutherAI/lm-evaluation-harness commit e5dfd03
ZERO_SHOT_TEMPLATE = "{support}\nQuestion: {question}\nAnswer:"


class SciQDataset(QuirkyDataset):
    quirky_template = (
        "Name: {character}\n\nPassage 1:\n{support}\n\nQ1: "
        '"{question} Is the answer "{answer}"?\nA:'
    )
    quirky_choices = (" No", " Yes")

    def __init__(self, n_shots: int = 5, n_few_shot_prompts: int = 1, **kwargs):
        self.n_shots = n_shots
        self.n_few_shot_prompts = n_few_shot_prompts
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        # set the random seed for choosing a random distractor
        random.seed(633)
        ds_dict = load_dataset("sciq").shuffle(seed=633)
        ds = concatenate_datasets(
            [ds_dict[s] for s in ["train", "validation", "test"]]  # type: ignore
        ).select(range(5_000))

        # split off 50 examples for the few-shot pool
        splits = ds.train_test_split(test_size=50, seed=633)
        ds = splits["train"]
        few_shot_pool = splits["test"]

        ds = ds.map(
            self._map_function,
            batched=False,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
            fn_kwargs={"few_shot_pool": few_shot_pool},
        )
        return ds

    def _map_function(self, example, few_shot_pool):
        support = example["support"].lstrip()
        distractor = random.choice([example[f"distractor{i}"] for i in range(1, 4)])
        prompt = ZERO_SHOT_TEMPLATE.format(
            question=example["question"], support=support
        )

        # This seems somewhat less important because class balance is not an issue
        prompts = []
        for i in range(self.n_few_shot_prompts):
            few_shot_set = few_shot_pool.shuffle(seed=633).select(range(self.n_shots))
            demonstrations = []
            for few_shot_example in few_shot_set:
                demonstrations.append(
                    ZERO_SHOT_TEMPLATE.format(
                        question=few_shot_example["question"],
                        support=few_shot_example["support"].lstrip(),
                    )
                    + " "
                    + few_shot_example["correct_answer"]
                )

            p = (
                "\n\n".join(demonstrations) + "\n\n" + prompt
                if self.n_shots > 0
                else prompt
            )
            prompts.append(p)

        (
            {"prompt": prompts[0]}
            if self.n_few_shot_prompts == 1
            else {"prompts": prompts}
        )

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

                    output["difficulty"].append(ex["difficulty"])
                    if self.additional_quirky_columns:
                        for col in self.additional_quirky_columns:
                            output[col].append(ex[col])
        return output
