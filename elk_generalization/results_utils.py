import numpy as np
import pandas as pd
from typing import Literal
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
import torch

NUM_LAYERS = 32


def get_raw_logprobs(
    fr: str, to: str, p_err: float, version: str | int, dir_template: str
) -> dict:
    logprobs = torch.load(
        dir_template.format(fr=fr, to=to, p_err=p_err, version=version)
    )
    assert len(logprobs) == 1
    only_value = list(logprobs.values()).pop()
    return only_value


def get_logprobs_df(
    raw_logprobs: dict, layer: int, ens: Literal["none", "partial"], inlp_iter: int = 0
) -> pd.DataFrame:
    # we need to flatten in the case of "none" ensembling
    row_ids = raw_logprobs["row_ids"].cpu().numpy().reshape(-1)
    texts = np.array(raw_logprobs["texts"]).reshape(-1)
    labels = np.array(raw_logprobs["labels"].cpu()).reshape(-1)
    lm_logprobs = np.array(raw_logprobs["lm"][ens]).reshape(
        -1
    )  # same as flatten, sorted by dictionary order where first dimension is most important
    lr_logprobs = np.array(raw_logprobs["lr"][layer][ens][inlp_iter]).reshape(-1)

    df = pd.DataFrame(
        {"row_id": row_ids, "text": texts, "lm": lm_logprobs, "lr": lr_logprobs, "label": labels}
    )
    return df


def measure_auroc_across_layers(
    raw_logprobs: dict,
    against: Literal["alice", "bob"],
    ens: Literal["none", "partial"],
    inlp_iter: int = 0,
    p_err: float = 1.0,
) -> pd.DataFrame:
    meta = {"against": against, "ens": ens, "inlp_iter": inlp_iter}
    results = []
    both_label_ds = load_dataset(
        f"atmallen/sloppy_addition_both_labels_{p_err}", split="validation"
    )
    both_label_df = both_label_ds.to_pandas()
    for layer in range(NUM_LAYERS):
        layer_df = get_logprobs_df(raw_logprobs, layer, ens, inlp_iter)
        layer_df["statement"] = layer_df["text"].apply(lambda text: text.removesuffix(". Alice:").removesuffix(". Bob:"))
        layer_df = layer_df.merge(both_label_df, on="statement")
        # check that the label column is the same as the alice_label or bob_label column
        assert all(layer_df["label"] == layer_df[f"alice_label"]) or all(
            layer_df["label"] == layer_df[f"bob_label"]
        )

        # TODO: add CI
        against_col = f"{against}_label"
        lr_auroc = roc_auc_score(layer_df[against_col], layer_df["lr"])
        lm_auroc = roc_auc_score(layer_df[against_col], layer_df["lm"])

        results.append(
            {"layer": layer, "lr_auroc": lr_auroc, "lm_auroc": lm_auroc, **meta}
        )

    results_df = pd.DataFrame(results)
    return results_df
