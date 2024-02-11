import hashlib
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, DatasetDict
from ds_utils import assert_type
from scipy.special import log_expit as logsigmoid  # type: ignore
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

StatementTemplate = namedtuple("StatementTemplate", ["context", "statement"])


class QuirkyDataset(ABC):
    """Abstract base class for quirky datasets"""

    quirky_templates: dict[str, tuple[str, str]] = None  # type: ignore
    statement_templates: list[StatementTemplate] = None  # type: ignore
    template_arg_names: list[str] = None  # type: ignore
    eval_difficulty_using_models: bool = False
    standardize_templates: bool = False

    def __init__(
        self,
        working_dir: str | Path | None = None,
        dataset_name: str | None = None,
        verbose: bool = False,
        user_or_org: str = "EleutherAI",
    ):
        self.name = (
            f"{user_or_org}/"
            + (
                dataset_name
                or f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
            )
            + "_raw"
        )  # indicate that this uses a mixture of templates
        self.working_dir = Path(working_dir or "../../quirky_datasets") / self.name
        self.verbose = verbose
        self.dataframe: pd.DataFrame = self._load()

    def evaluate(
        self,
        model_name: str,
        max_examples: int = 1000,
    ) -> pd.DataFrame:
        """
        Evaluate the model on the dataset and save the results as huggingface dataset
        If the results already exist, skip the evaluation

        Returns:
            The dataset with the results added as a column, with order preserved
        """
        assert isinstance(
            self.dataframe, pd.DataFrame
        ), "self.dataset must have type pd.DataFrame"
        assert all(
            col in self.dataframe.columns for col in ["id", "choices", "label"]
        ), "self.dataset must have columns 'id', 'prompt', 'choices', and 'label'"
        assert (
            "prompt" in self.dataframe.columns or "prompts" in self.dataframe.columns
        ), "self.dataset must have column 'prompt' or 'prompts'"

        model_last = model_name.split("/")[-1]
        save_path = self.working_dir / f"{model_last}_eval_df_{max_examples}.json"
        # TODO
        print("SAVE PATH", save_path)
        if save_path.exists():
            if self.verbose:
                print(f"Loading results from {save_path}")
            return pd.read_json(str(save_path))

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": torch.cuda.current_device()},
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")

        dataframe = self.dataframe.iloc[:max_examples]

        log_odds = torch.full(
            [
                len(dataframe),
            ],
            torch.nan,
            device=model.device,
            dtype=model.dtype,
        )
        for i, example in tqdm(dataframe.iterrows(), total=len(dataframe)):
            example = assert_type(dict, dict(example))
            i = assert_type(int, i)

            # either get log odds from prompt or all prompts
            if "prompt" in example:
                log_odds[i] = self._get_log_odds(
                    model,
                    tokenizer,
                    example["prompt"],
                    example["choices"],
                )
            elif "prompts" in example:
                example_log_odds = [
                    self._get_log_odds(model, tokenizer, p, example["choices"])
                    for p in example["prompts"]
                ]
                log_odds[i] = torch.mean(torch.stack(example_log_odds))

        np_lo = log_odds.cpu().float().numpy()
        dataframe.loc[:, "log_odds"] = np_lo
        dataframe.to_json(save_path)

        if self.verbose:
            labels = torch.as_tensor(dataframe["label"], device=model.device)
            accuracy = (log_odds > 0).eq(labels).float().mean().item()
            try:
                auc = roc_auc_score(labels.cpu().numpy(), np_lo)
            except ValueError:
                auc = np.nan
            labels.float().mean().item()

            print(f"Accuracy: {accuracy:.3f}")
            print(f"AUC: {auc:.3f}")
            print(f"Saved results to {save_path}")

        return dataframe

    @staticmethod
    def _get_log_odds(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        prompt_str: str,
        choice_strs: list[str],
    ) -> torch.Tensor:
        # warn if truncating
        prompt = tokenizer.encode(prompt_str)
        choice_toks = [tokenizer.encode(c) for c in choice_strs]
        num_completion_toks = max(len(c) for c in choice_toks)
        if len(prompt) + num_completion_toks > model.config.max_position_embeddings:
            print(
                f"Warning: prompt length {len(prompt)} exceeds "
                f"model max length {tokenizer.model_max_length}"
            )
            prompt = prompt[
                -model.config.max_position_embeddings + num_completion_toks :
            ]

        with torch.inference_mode():
            # get model outputs and cache in response to prompt
            outputs = model(
                torch.as_tensor([prompt], device=model.device), use_cache=True
            )
            logprob_output = outputs.logits[0, -1, :].log_softmax(dim=-1)
            logprobs = logprob_output[[choice_toks[0][0], choice_toks[1][0]]]

            # we compute log_odds of the whole completion, possibly multiple tokens
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
                    logprobs[j] += choice_outputs.logits.log_softmax(dim=-1)[
                        0, -1, ctoks[k + 1]
                    ]

        # softmax adds constant to both, which cancels out, so is unnecessary here
        # log(p / (1 - p)) = log(p) - log(1 - p)
        log_odds = logprobs[1] - logprobs[0]

        return log_odds

    @abstractmethod
    def _load(self) -> pd.DataFrame:
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
        dataframes = {
            model: self.evaluate(
                model,
                max_examples=max_examples,
            )
            for model in model_names
        }

        losses = np.stack(
            [
                # single datapoint binary cross entropy
                # negating log_odds causes logsigmoid to return log(1 - p)
                -logsigmoid(ds["log_odds"] * np.where(ds["label"], 1, -1))
                for ds in dataframes.values()
            ],
            axis=1,
        )  # shape (n_examples, n_models)
        difficulties = np.mean(losses, axis=1)

        if self.verbose:

            def monotonicity(xs, atol=0.01):
                n = len(xs)
                num_inversions = sum(
                    x1 + atol < x2 for i, x1 in enumerate(xs) for x2 in xs[i + 1 :]
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

    def _generate_base_dataset(
        self,
        n_total: int,
        difficulty_model_names: list[str],
    ):
        base_df = self.dataframe[:n_total]
        if self.eval_difficulty_using_models:
            base_df.loc[:, "difficulty"] = self._get_difficulties(
                difficulty_model_names,
                max_examples=n_total,
            )
        else:
            assert (
                not difficulty_model_names
            ), "This dataset does not evaluate difficulty using models"

        return base_df, dict()

    def save_quirky_dataset(
        self,
        difficulty_model_names: list[str],
        n_train: int = 100_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
        push_to_hub: bool = True,
    ):
        """Save the quirky dataset to disk and push it to the hub"""
        if n_train == -1:
            n_train = len(self.dataframe) - n_val - n_test
        base_df, transform_kwargs = self._generate_base_dataset(
            n_train + n_val + n_test,
            difficulty_model_names,
        )
        quirky_ds = self._transform_base_dataset(base_df, transform_kwargs)

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
            quirky_dict.push_to_hub(self.name)

    def _transform_base_dataset(
        self, base_ds: pd.DataFrame, fn_kwargs: dict
    ) -> Dataset:
        """Transform the base dataset into a quirky dataset"""
        records = []
        for _, ex in base_ds.iterrows():
            records.extend(self._quirky_map_function(ex, **fn_kwargs))

        # for some reason, converting to pandas first makes column typing more reliable
        quirky_dataset = Dataset.from_pandas(pd.DataFrame(records))
        quirky_dataset.cast_column(
            "label", ClassLabel(num_classes=2, names=["False", "True"])
        )

        # add difficulty_quantile column
        difficulties = np.array(quirky_dataset["difficulty"])
        # order_[i] is the index into difficulties of the ith easiest example
        order_ = np.argsort(difficulties)
        # ranks[i] is the index into order_ (the rank) of the ith example in difficulties
        ranks = np.argsort(order_)
        quantiles = (ranks + 0.5) / len(difficulties)
        quirky_dataset = quirky_dataset.add_column("difficulty_quantile", quantiles)  # type: ignore

        return quirky_dataset

    def _quirky_map_function(self, example: pd.Series) -> list[dict[str, Any]]:
        """Map function for transforming the base dataset into a quirky dataset"""
        ex = dict(example)

        records = []
        alice_label, bob_label = ex["label"], ex["bob_label"]
        for character, label in [("Alice", alice_label), ("Bob", bob_label)]:
            template_args = {
                "character": character,
                **{k: ex[k] for k in self.template_arg_names},
            }

            records.append(
                {
                    "id": hashlib.md5(str(template_args).encode()).hexdigest()[0:8],
                    "template_args": template_args,
                    "character": character,
                    "label": label,
                    "alice_label": alice_label,
                    "bob_label": bob_label,
                    "difficulty": ex["difficulty"],
                }
            )

        return records
