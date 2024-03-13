from typing import Any, Callable, Type, TypeVar, cast

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


def get_quirky_model_names(
    ds_name,
    base_model_id,
    templatization_method="first",
    standardize_templates=False,
    weak_only=False,
    full_finetuning=False,
    model_hub_user="EleutherAI",
):
    base_model_last = base_model_id.split("/")[-1]
    model_id = (
        f"{model_hub_user}/{base_model_last}-{ds_name}-"
        + templatization_method
        + ("-standardized" if standardize_templates else "")
        + ("-weak-only" if weak_only else "")
        + ("-ft" if full_finetuning else "")
    )
    model_last = model_id.split("/")[-1]
    return model_id, model_last
