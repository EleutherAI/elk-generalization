from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import assert_type, encode_choice


class WeakLMDataset(ABC):
    """
    An abstract base class for datasets that derives
    untruthful answers from weak LM supervision
    """

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
            return assert_type(Dataset, load_from_disk(str(save_path)))

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": torch.cuda.current_device()},
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            prompt = tokenizer.encode(example["prompt"])  # type: ignore
            choice_toks = [
                encode_choice(example["choices"][0], tokenizer),  # type: ignore
                encode_choice(example["choices"][1], tokenizer),  # type: ignore
            ]

            with torch.inference_mode():
                outputs = model(torch.as_tensor([prompt], device=model.device))

            logit_neg, logit_pos = outputs.logits[0, -1, choice_toks]
            # softmax adds constant to both, which cancels out, so is unnecessary here
            # log(p / (1 - p)) = log(p) - log(1 - p)
            log_odds[i] = logit_pos - logit_neg

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
        self, model_names: list[str], max_examples: int = 1000
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
        for i, example in enumerate(self.dataset.select(range(max_examples))):
            correct_answer = example["choices"][example["label"]]  # type: ignore
            correct = [
                model
                for model, ds in datasets.items()
                if ds[i]["choices"][ds[i]["label"]] == correct_answer
            ]
            # TODO: compute monotonocity
            difficulties.append(len(correct) / len(datasets))

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
    ):
        """Generate a quirky dataset, split it into subsets, and push it to the hub"""
        n_total = n_train + n_val + n_test
        base_ds = self.evaluate(weak_model_name, max_examples=n_total)
        base_ds.add_column(
            "difficulty",
            self.get_difficulties(difficulty_model_names, max_examples=n_total),
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

    @abstractmethod
    def make_quirky_dataset(
        self,
        base_dataset: DatasetDict,
    ) -> DatasetDict:
        """Transform the base dataset into a quirky dataset"""
        ...
