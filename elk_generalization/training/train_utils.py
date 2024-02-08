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
