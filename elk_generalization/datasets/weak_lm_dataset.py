from abc import ABC, abstractmethod
from pathlib import Path

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
        for i, example in tqdm(enumerate(dataset)):
            prompt = tokenizer.encode(example["prompt"])  # type: ignore
            choice_toks = [
                encode_choice(example["choices"][0], tokenizer),  # type: ignore
                encode_choice(example["choices"][1], tokenizer),  # type: ignore
            ]

            with torch.inference_mode():
                outputs = model(
                    torch.as_tensor([prompt], device=model.device),
                    output_hidden_states=True,
                    use_cache=True,
                )

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

    @abstractmethod
    def make_quirky_dataset(
        self,
        push_to_hub: bool = False,
        n_train: int = 100_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
    ) -> DatasetDict:
        """Generate a quirky dataset and return it, optionally pushing it to the hub"""
        ...
