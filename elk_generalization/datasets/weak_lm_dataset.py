import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import assert_type, transpose_dict


class WeakLMDataset(ABC):
    """
    An abstract base class for datasets that derives
    untruthful answers from weak LM supervision
    """

    quirky_template: str
    quirky_choices: tuple[str, str]

    def __init__(
        self,
        working_dir: str | Path | None = None,
    ):
        self.dataset_name = (
            f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
        )
        self.working_dir = (
            Path(working_dir or "../../weak_lm_datasets") / self.dataset_name
        )
        self.dataset = self.load()

    def evaluate(
        self, model_name: str, max_examples: int = 1000, verbose: bool = False
    ) -> Dataset:
        """
        Evaluate the model on the dataset and save the results as huggingface dataset
        If the results already exist, skip the evaluation

        Returns:
            The dataset with the results added as a column
        """
        assert isinstance(self.dataset, Dataset), "self.dataset must have type Dataset"
        assert all(
            col in self.dataset.column_names
            for col in ["id", "prompt", "choices", "label"]
        ), "self.dataset must have columns 'id', 'prompt', 'choices', and 'label'"

        model_last = model_name.split("/")[-1]
        save_path = self.working_dir / f"{model_last}_results"
        if save_path.exists():
            if verbose:
                print(f"Loading results from {save_path}")
            return assert_type(Dataset, load_from_disk(str(save_path)))

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": torch.cuda.current_device()},
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")

        dataset = self.dataset.select(range(max_examples))

        log_odds = torch.full(
            [
                len(dataset),
            ],
            torch.nan,
            device=model.device,
            dtype=model.dtype,
        )
        for i, example in tqdm(enumerate(dataset), total=len(dataset)):
            example = assert_type(dict, example)
            prompt = tokenizer.encode(example["prompt"])
            choice_toks = [tokenizer.encode(c) for c in example["choices"]]

            with torch.inference_mode():
                # get model outputs and cache in response to prompt
                outputs = model(
                    torch.as_tensor([prompt], device=model.device), use_cache=True
                )
                logits = outputs.logits[0, -1, [choice_toks[0][0], choice_toks[1][0]]]

                # for each completion, while there are more tokens, get more outputs
                for j, ctoks in enumerate(choice_toks):
                    cache = outputs.past_key_values
                    input_ids = prompt.copy()
                    for k in range(len(ctoks) - 1):
                        input_ids.append(ctoks[k])
                        choice_outputs = model(
                            torch.as_tensor([input_ids], device=model.device),
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache = choice_outputs.past_key_values
                        # add the logit for the next token
                        logits[j] += choice_outputs.logits[0, -1, ctoks[k + 1]]

            # softmax adds constant to both, which cancels out, so is unnecessary here
            # log(p / (1 - p)) = log(p) - log(1 - p)
            log_odds[i] = logits[1] - logits[0]

        np_lo = log_odds.cpu().float().numpy()
        dataset = dataset.add_column("log_odds", np_lo)  # type: ignore
        dataset.save_to_disk(save_path)

        if verbose:
            labels = torch.as_tensor(dataset["label"], device=model.device)
            accuracy = (log_odds > 0).eq(labels).float().mean().item()
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Saved results to {save_path}")

        return dataset

    @abstractmethod
    def load(self) -> Dataset:
        """Load the dataset, create prompts for non-quirky inference, and return it"""
        ...

    def get_difficulties(
        self, model_names: list[str], max_examples: int = 1000, verbose: bool = False
    ) -> list[float]:
        """
        Compute the difficulty for the first `max_examples` examples in self.dataset

        Conceptually, difficulty aims to capture how many resources are required to
        answer a question correctly (e.g. dollars). We approximate this by counting
        the number of models that answer correctly, where the distribution of models
        covers a wide range of scales (resources expended in training).
        """
        datasets = {
            model: self.evaluate(model, max_examples=max_examples)
            for model in model_names
        }

        difficulties = []
        is_monotonic = []
        is_all_correct = []
        is_all_incorrect = []
        for i, ex in enumerate(self.dataset.select(range(max_examples))):
            ex_id = ex["id"]  # type: ignore
            assert all(ds["id"][i] == ex_id for ds in datasets.values())

            correct = [
                (ds[i]["log_odds"] > 0) == ds[i]["label"] for ds in datasets.values()
            ]

            difficulties.append(np.mean(correct))
            is_monotonic.append(correct == sorted(correct))
            is_all_correct.append(all(correct))
            is_all_incorrect.append(not any(correct))

        is_monotonic, is_all_correct, is_all_incorrect = [
            np.array(x) for x in [is_monotonic, is_all_correct, is_all_incorrect]
        ]
        if verbose:
            print(f"Proportion monotonic: {np.mean(is_monotonic):.3f}")
            print(f"Proportion all correct: {np.mean(is_all_correct):.3f}")
            print(f"Proportion all incorrect: {np.mean(is_all_incorrect):.3f}")
            ntm = np.mean(is_monotonic[~is_all_correct & ~is_all_incorrect])
            print(f"Proportion monotonic given not all correct/incorrect: {ntm:.3f}")

        return difficulties

    def push_to_hub(
        self,
        weak_model_name: str,
        difficulty_model_names: list[str],
        n_train: int = 100_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
        difficulty_quantile: float = 0.25,  # e.g. easiest/hardest 25% of examples
        push_to_hub: bool = True,
        verbose: bool = False,
    ):
        """Generate a quirky dataset, split it into subsets, and push it to the hub"""
        n_total = n_train + n_val + n_test
        base_ds = self.evaluate(weak_model_name, max_examples=n_total, verbose=verbose)
        base_ds.add_column(
            "difficulty",
            self.get_difficulties(
                difficulty_model_names, max_examples=n_total, verbose=verbose
            ),
        )  # type: ignore
        base_ds = DatasetDict(
            {
                "train": base_ds.select(range(n_train)),
                "validation": base_ds.select(range(n_train, n_train + n_val)),
                "test": base_ds.select(
                    range(n_train + n_val, n_train + n_val + n_test)
                ),
            }
        )
        quirky_dict = self.make_quirky_dataset(base_ds)

        model_last = weak_model_name.split("/")[-1]
        quirky_dict.save_to_disk(self.working_dir / f"{model_last}_quirky")
        if push_to_hub:
            quirky_dict.push_to_hub(f"{self.dataset_name}_{model_last}")

            easy_thresh = np.quantile(
                base_ds["train"]["difficulty"], difficulty_quantile
            )
            hard_thresh = np.quantile(
                base_ds["train"]["difficulty"], 1 - difficulty_quantile
            )
            for character in ["Alice", "Bob"]:
                for difficulty in ["easy", "hard"]:

                    def difficulty_filter(x):
                        return (
                            x["difficulty"] < easy_thresh
                            if difficulty == "easy"
                            else x["difficulty"] >= hard_thresh
                        )

                    subset = quirky_dict.filter(
                        lambda x: (x["character"] == character) and difficulty_filter(x)
                    )
                    subset.push_to_hub(
                        f"{self.dataset_name}_{model_last}_{character.lower()}_{difficulty}"
                    )

                subset = quirky_dict.filter(lambda x: x["character"] == character)
                subset.push_to_hub(
                    f"{self.dataset_name}_{model_last}_{character.lower()}"
                )

    def make_quirky_dataset(self, base_ds: DatasetDict) -> DatasetDict:
        """Transform the base dataset into a quirky dataset"""
        base_ds = base_ds.map(
            self._quirky_map_function,
            batched=True,
            remove_columns=base_ds["train"].column_names,
        )
        return base_ds

    def _quirky_map_function(self, examples, thresh=0):
        examples = transpose_dict(examples)

        output = defaultdict(list)
        for ex in examples:
            # log_odds is the log odds assigned to the second (correct) choice
            bob_answer = (
                ex["correct_answer"] if ex["log_odds"] > thresh else ex["distractor"]
            )
            alice_answer = ex["correct_answer"]

            for character, character_answer in [
                ("Alice", alice_answer),
                ("Bob", bob_answer),
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
