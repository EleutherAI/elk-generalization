import random
from typing import Any, Literal

from datasets import Dataset, DatasetDict, Split, load_dataset

from elk_generalization.datasets.create_datasets import DATASETS_BY_NAME

STANDARDIZED_TEMPLATE = """Name: {character}

<|CONTEXT|>

***STATEMENT:*** <|STATEMENT|>

Is the statement factually correct?"""
STANDARDIZED_CHOICES = (" No", " Yes")


def load_quirky_dataset(
    ds_name: str,
    character: Literal["Alice", "Bob", "none"] = "none",
    max_difficulty_quantile: float = 1.0,
    min_difficulty_quantile: float = 0.0,
    split: str | Split | None = None,
) -> DatasetDict | Dataset:
    """Load a quirky dataset with the specified character and difficulty constraints."""
    ds = load_dataset(ds_name, split=split)

    # filter by character and/or difficulty if any constraints are specified
    if (
        character != "none"
        or min_difficulty_quantile > 0.0
        or max_difficulty_quantile < 1.0
    ):
        ds = ds.filter(
            lambda x: (character == "none" or x["character"] == character)
            and (
                min_difficulty_quantile
                <= x["difficulty_quantile"]
                <= max_difficulty_quantile
            )
        )

    return ds  # type: ignore


def load_templates(
    dataset_name: str, standardize_templates: bool = False
) -> list[dict[str, Any]]:
    """Get the templates for a quirky dataset. If standardize_templates is True, the templates
    will be generated using STANDARDIZED_TEMPLATE filling in the statement and context from the
    statement_templates field of the quirky dataset. Standardization attempts to make truth
    more easily extractible by concluding the prompt with a common question. Otherwise, the
    datasets quirky_templates will be used."""
    dataset_name = dataset_name.removesuffix("_raw").split("/")[-1]
    class_ = DATASETS_BY_NAME[dataset_name]
    if standardize_templates:
        return [
            {
                "template": STANDARDIZED_TEMPLATE.replace("<|CONTEXT|>", t_c).replace(
                    "<|STATEMENT|>", t_s
                ),
                "choices": STANDARDIZED_CHOICES,
            }
            for t_c, t_s in class_.statement_templates
        ]
    return [{"template": t, "choices": c} for t, c in class_.quirky_templates.items()]


def templatize_quirky_dataset(
    ds: Dataset | DatasetDict,
    ds_name: str,
    standardize_templates: bool = False,
    method: Literal[
        "random", "all", "first"
    ] = "random",  # TODO: support all with some sort of batching
) -> Dataset | DatasetDict:
    """
    Templatize a quirky dataset, producing a dataset with columns
    "statement", "choices", "label", "character", "difficulty",
    "difficulty_quantile", "alice_label", "bob_label".
    """
    if method == "all":
        raise NotImplementedError(f"Method {method} not yet implemented")

    # get template to compare against for assert_all_templates_same
    templates = load_templates(ds_name, standardize_templates=standardize_templates)

    def map_fn(ex):
        targs = ex.pop("template_args")

        if method == "random":
            t = random.choice(templates)
        elif method == "first":
            t = templates[0]
        else:
            raise ValueError(f"Unknown method: {method}")
        template, choices = t["template"], t["choices"]

        return {"statement": template.format(**targs), "choices": choices, **ex}

    return ds.map(map_fn, batched=False)
