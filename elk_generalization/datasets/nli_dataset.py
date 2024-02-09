import hashlib
import random
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_dataset

from ds_utils import transpose_dict
from quirky_dataset import QuirkyDataset

ZERO_SHOT_TEMPLATE = 'Q: Does "{premise}" imply or contradict "{hypothesis}"?\nA:'
ZERO_SHOT_CHOICES = (" Contradict", " Imply")
NEGATION_WORDS = {"not", "nobody", "no", "never", "nothing", "none"}


class NliDataset(QuirkyDataset):
    """
    This starts with SNLI and removes neutral examples. It then adds distractors
    """
    template_arg_names = ["premise", "hypothesis"]
    quirky_templates = {
        'Name: {character}\n\nQ: Does "{premise}" imply or '
        'contradict "{hypothesis}"?\nA:': (" Contradict", " Imply"),

        'Does "{premise}" contradict or does it'
        'imply "{hypothesis}"? {character} thinks that it': (" contradicts", " implies"),

        'what\'s up {character}, Given your absolute best judgment, do you believe that "{premise}" would imply or '
        'potentially contradict "{hypothesis}"?\nA:': (" Contradict", " Imply"),

        'Name: {character}\n\nQ: "{premise}". This is really interesting and has stumped philosophers for a while. Do you believe it implies or '
        'contradicts the hypothesis "{hypothesis}"? I need your answer now.': (" Contradicts", " Implies"),

        'Hi {character}, how\'s it going. This is easy but somehow I am overthinking it. Does "{premise}" imply or '
        'contradict "{hypothesis}"?\nA:': (" Contradict", " Imply"),

        'We also know that there is not necessarily a correct answer but we are trying to get to one. So {character}, does "{premise}" imply or '
        'contradict "{hypothesis}"?': (" Contradict", " Imply"),

        'Name: {character}\n\nQ: I did not like the previous response so let\'s try again. Does "{premise}" imply or '
        'contradict "{hypothesis}"?\nA:': (" Contradict", " Imply"),

        'Name: {character}\n\nQ: Does "{premise}" imply or '
        'contradict "{hypothesis}"? Please try to be very very exact with this.\nA:': (" Contradict", " Imply"),

        'This is very serious. {character}, givn duh premis, "{premise}", do it emplie or '
        'konterdik these hypotheses: "{hypothesis}"?': (" Konterdik", " Emplie"),

        'Name: {character}\n\nQ: Spain is the best country to go to the beach in. Does "{premise}" imply or '
        'contradict "{hypothesis}"?\nA - ': (" Contradict", " Imply"),        
    }

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
            self._process_raw_example,
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
    def _process_raw_example(example, neg_pool, pos_pool, n_shots=5):
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
