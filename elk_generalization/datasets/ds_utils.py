from typing import Any, Callable, Type, TypeVar, cast, Literal
from datasets import DatasetDict, Dataset, load_dataset, Split
import random

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


In, Out = TypeVar("In"), TypeVar("Out")
DictFn = Callable[[dict[str, In]], dict[str, Out]]
VmappedFn = Callable[[dict[str, list[In]]], dict[str, list[Out]]]


def dict_vmap(func: DictFn) -> VmappedFn:
    """Turn a function taking dict[str, In] into one that takes dict[str, list[In]]."""

    def wrapper(input_dict: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Transpose the input dict of lists into a list of dicts
        keys = input_dict.keys()
        transposed_input = [
            dict(zip(keys, values)) for values in zip(*input_dict.values())
        ]

        # Apply the wrapped function to each dict
        results = [func(single_input) for single_input in transposed_input]

        # Transpose the results back into a dict of lists
        # Assuming that each result is a dictionary
        transposed_output = {
            key: [result[key] for result in results] for key in results[0]
        }

        return transposed_output

    return wrapper


def encode_choice(text, tokenizer):
    c_ids = tokenizer.encode(text, add_special_tokens=False)

    # some tokenizers split off the leading whitespace character
    if tokenizer.decode(c_ids[0]).strip() == "":
        c_ids = c_ids[1:]
        assert c_ids == tokenizer.encode(text.lstrip(), add_special_tokens=False)
    assert len(c_ids) == 1, f"Choice should be one token: {text}"
    return c_ids[0]


def transpose_dict(examples: dict[str, list]) -> list[dict[str, Any]]:
    """Transpose a dict of lists to a list of dicts"""
    return [dict(zip(examples, values)) for values in zip(*examples.values())]


def load_quirky_dataset(ds_name: str,
                        character: Literal["alice", "bob", "none"] = "none",
                        max_difficulty_quantile: float = 1.0,
                        min_difficulty_quantile: float = 0.0,
                        split: str | Split | None = None,
    ) -> DatasetDict | Dataset:
    """Load a quirky dataset with the specified character and difficulty constraints."""
    ds = load_dataset(ds_name, split=split)

    # filter by character and/or difficulty if any constraints are specified
    if character != "none" or min_difficulty_quantile > 0.0 or max_difficulty_quantile < 1.0:
        ds = ds.filter(
            lambda x:
                (character == "none" or x["character"] == character) and
                (min_difficulty_quantile <= x["difficulty_quantile"] <= max_difficulty_quantile)
        )

    return ds  # type: ignore


def templatize_quirky_dataset(
        ds: Dataset | DatasetDict,
        method: Literal["random", "all"] = "random",  # TODO: support all with some sort of batching
        assert_all_templates_same: bool = False,
    ) -> Dataset | DatasetDict:
    """
    Templatize a quirky dataset, producing a dataset with columns
    "statement", "choices", "label", "character", "difficulty",
    "difficulty_quantile", "alice_label", "bob_label".
    """
    if method == "all":
        raise NotImplementedError("Templatizing all examples is not yet supported")
    
    # get template to compare against for assert_all_templates_same
    templates0 = next(iter(ds.values()))[0]["templates"] if isinstance(ds, DatasetDict) else ds[0]["templates"]
    
    def map_fn(ex):
        templates = ex.pop("templates")
        targs = ex.pop("targs")
        
        if method == "random":
            template, choices = random.choice(templates)
        else:
            raise ValueError(f"Unknown method: {method}")

        assert not assert_all_templates_same or templates == templates0, \
            "All examples should have the same templates when assert_all_templates_same is True"
        
        return {"statement": template.format(**targs), "choices": choices, **ex}
    
    return ds.map(map_fn, batched=False, remove_columns=["templates", "template_args"])



