from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_from_disk
from scipy.special import log_expit as logsigmoid  # type: ignore
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import assert_type


class QuirkyDataset(ABC):
    """
    An abstract base class for datasets that derives
    untruthful answers from weak LM supervision
    """

    quirky_template: str
    quirky_choices: tuple[str, str]
    additional_quirky_columns: list[str] | None = None

    def __init__(
        self,
        working_dir: str | Path | None = None,
        verbose: bool = False,
    ):
        self.dataset_name = (
            f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
        )
        self.working_dir = (
            Path(working_dir or "../../quirky_datasets") / self.dataset_name
        )
        self.verbose = verbose
        self.dataset = self._load()

    def evaluate(
        self,
        model_name: str,
        max_examples: int = 1000,
    ) -> Dataset:
        """
        Evaluate the model on the dataset and save the results as huggingface dataset
        If the results already exist, skip the evaluation

        Returns:
            The dataset with the results added as a column, with order preserved
        """
        assert isinstance(self.dataset, Dataset), "self.dataset must have type Dataset"
        assert all(
            col in self.dataset.column_names
            for col in ["id", "prompt", "choices", "label"]
        ), "self.dataset must have columns 'id', 'prompt', 'choices', and 'label'"

        model_last = model_name.split("/")[-1]
        save_path = self.working_dir / f"{model_last}_results"
        if save_path.exists():
            if self.verbose:
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

        if self.verbose:
            labels = torch.as_tensor(dataset["label"], device=model.device)
            accuracy = (log_odds > 0).eq(labels).float().mean().item()
            try:
                auc = roc_auc_score(labels.cpu().numpy(), np_lo)
            except ValueError:
                auc = np.nan
            cal_thresh = np.quantile(np_lo, 0.5)
            cal_acc = (log_odds > cal_thresh).eq(labels).float().mean().item()

            print(f"Accuracy: {accuracy:.3f}")
            print(f"AUC: {auc:.3f}")
            print(f"Calibrated accuracy: {cal_acc:.3f}")
            print(f"Saved results to {save_path}")

        return dataset

    @abstractmethod
    def _load(self) -> Dataset:
        """Load the dataset, create prompts for non-quirky inference, and return it"""
        ...

    def _get_difficulties(
        self,
        model_names: list[str],
        max_examples: int = 1000,
    ) -> list[float]:
        """
        Compute the difficulty for the first `max_examples` examples in self.dataset

        Conceptually, difficulty aims to capture how many resources are required to
        answer a question correctly (e.g. dollars). We approximate this by counting
        the number of models that answer correctly, where the distribution of models
        covers a wide range of scales (resources expended in training).
        """
        datasets = {
            model: self.evaluate(
                model,
                max_examples=max_examples,
            ).with_format("numpy")
            for model in model_names
        }

        losses = np.stack(
            [
                # single datapoint binary cross entropy
                # negating log_odds causes logsigmoid to return log(1 - p)
                -logsigmoid(ds["log_odds"] * np.where(ds["label"], 1, -1))
                for ds in datasets.values()
            ],
            axis=1,
        )  # shape (n_examples, n_models)
        difficulties = np.mean(losses, axis=1)

        if self.verbose:

            def monotonicity(xs):
                n = len(xs)
                num_inversions = sum(
                    x1 < x2 for i, x1 in enumerate(xs) for x2 in xs[i + 1 :]
                )
                return 1 - num_inversions / (n * (n - 1) / 2)

            # sorted_losses = np.sort(losses, axis=1, kind="stable")[:, ::-1]
            # is_monotonic = np.all(sorted_losses == losses, axis=1)
            # print(f"Monotonicity: {is_monotonic.mean():.3f}")
            monotonicities = np.apply_along_axis(monotonicity, 1, losses)
            print(f"Monotonicity: {monotonicities.mean():.3f}")

            avg_losses = losses.mean(axis=0)
            print(f"Average losses: {avg_losses}")

        return difficulties

    @abstractmethod
    def _generate_base_dataset(
        self,
        n_total: int,
        difficulty_model_names: list[str],
    ) -> tuple[Dataset, dict]:
        """Generate the quirky dataset from self.dataset"""
        ...

    def save_quirky_dataset(
        self,
        difficulty_model_names: list[str],
        n_train: int = 100_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
        push_to_hub: bool = True,
        difficulty_quantile: float = 0.25,
    ):
        """Save the quirky dataset to disk and push it to the hub"""
        if n_train == -1:
            n_train = len(self.dataset) - n_val - n_test
        base_ds, transform_kwargs = self._generate_base_dataset(
            n_train + n_val + n_test,
            difficulty_model_names,
        )
        quirky_ds = self._transform_base_dataset(base_ds, transform_kwargs)
        quirky_dict = DatasetDict(
            {
                "train": quirky_ds.select(range(n_train)),
                "validation": quirky_ds.select(range(n_train, n_train + n_val)),
                "test": quirky_ds.select(
                    range(n_train + n_val, n_train + n_val + n_test)
                ),
            }
        )

        save_path = self.working_dir / "quirky"
        quirky_dict.save_to_disk(save_path)
        if self.verbose:
            print(f"Saved quirky dataset to {save_path}")

        if push_to_hub:
            quirky_dict.push_to_hub(f"{self.dataset_name}")

            easy_thresh = np.quantile(
                quirky_dict["train"]["difficulty"], difficulty_quantile
            )
            hard_thresh = np.quantile(
                quirky_dict["train"]["difficulty"], 1 - difficulty_quantile
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
                        f"{self.dataset_name}_{character.lower()}_{difficulty}"
                    )

                subset = quirky_dict.filter(lambda x: x["character"] == character)
                subset.push_to_hub(f"{self.dataset_name}_{character.lower()}")

    def _transform_base_dataset(self, base_ds: Dataset, fn_kwargs: dict) -> Dataset:
        """Transform the base dataset into a quirky dataset"""
        quirky_ds = base_ds.map(
            self._quirky_map_function,
            batched=True,
            remove_columns=base_ds.column_names,
            fn_kwargs=fn_kwargs,
            load_from_cache_file=False,
        )
        quirky_ds.cast_column(
            "label", ClassLabel(num_classes=2, names=["False", "True"])
        )
        return quirky_ds

    @abstractmethod
    def _quirky_map_function(self, examples):
        """Map function for transforming the base dataset into a quirky dataset"""
        ...
