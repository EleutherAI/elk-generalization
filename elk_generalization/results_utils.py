import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Literal
from datasets import load_dataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import yaml
from pathlib import Path


def get_raw_logprobs(
    fr: str, to: str, p_err: float, version: str | int, dir_template: str, verbose: bool = False
) -> dict:
    path = Path(dir_template.format(fr=fr, to=to, p_err=p_err, version=version))
    if verbose:
        print(f"Loading {path}")
    logprobs = torch.load(path)
    assert len(logprobs) == 1
    only_value = list(logprobs.values()).pop()
    return only_value


def get_logprobs_df(
    raw_logprobs: dict, layer: int, ens: Literal["none", "partial"], inlp_iter: int = 0, reporter_key: Literal["lr", "reporter"]="lr"
) -> pd.DataFrame:
    texts = np.array(raw_logprobs["texts"])
    if texts.ndim == 3:
        assert texts.shape[-1] == 2
        texts = texts[..., 0]
    texts = texts.reshape(-1)
    lm_logprobs = np.array(raw_logprobs["lm"][ens]).reshape(-1)
    if reporter_key == "reporter":
        reporter_logprobs = np.array(raw_logprobs[reporter_key][layer][ens]).reshape(-1)
    else:
        reporter_logprobs = np.array(raw_logprobs[reporter_key][layer][ens][inlp_iter]).reshape(-1)
    # duplicate labels `num_variants` times to match the shape of `lm_logprobs` and `reporter_logprobs`
    num_variants = lm_logprobs.shape[0] // len(raw_logprobs["labels"])
    # we need to flatten in the case of "none" ensembling
    labels = np.array(raw_logprobs["labels"].cpu()).repeat(num_variants)
    row_ids = raw_logprobs["row_ids"].cpu().numpy().repeat(num_variants)

    df = pd.DataFrame(
        {"row_id": row_ids, "text": texts, "lm": lm_logprobs, "reporter": reporter_logprobs, "label": labels}
    )
    return df


def measure_across_layers(
    raw_logprobs: dict,
    against: Literal["alice", "bob"],
    filter_by: Literal["agree", "disagree", "all"] = "all",
    ens: Literal["none", "partial"] = "none",
    inlp_iter: int = 0,
    p_err: float = 1.0,
    reporter_key: Literal["lr", "reporter"] = "lr"
) -> pd.DataFrame:
    meta = {"against": against, "ens": ens, "inlp_iter": inlp_iter}
    results = []
    both_label_ds: Dataset = load_dataset(
        f"atmallen/qm_{p_err}e_eval", split="validation"
    )  # type: ignore
    both_label_df: DataFrame = both_label_ds.to_pandas()  # type: ignore
    both_label_df = both_label_df.rename({"label": "both_df_label"}, axis="columns")
    num_layers = len(raw_logprobs[reporter_key])
    # for layer in range(num_layers):
    for layer in range(1, num_layers):  # TODO: change back to including 0
        pre_df = get_logprobs_df(raw_logprobs, layer, ens, inlp_iter, reporter_key)
        layer_df = pre_df.merge(both_label_df, on="row_id", how="left")
        # check that the label column is the same as the alice_label or bob_label column
        assert all(layer_df["label"] == layer_df[f"alice_label"]) or all(
            layer_df["label"] == layer_df[f"bob_label"]
        ) or all(
            layer_df["label"] == layer_df[f"both_df_label"]
        )
        if filter_by == "agree":
            layer_df = layer_df[layer_df["alice_label"] == layer_df["bob_label"]]
        elif filter_by == "disagree":
            layer_df = layer_df[layer_df["alice_label"] != layer_df["bob_label"]]

        # TODO: possibly add CI
        against_col = f"{against}_label"
        try:
            reporter_auroc = roc_auc_score(layer_df[against_col], layer_df["reporter"])
            lm_auroc = roc_auc_score(layer_df[against_col], layer_df["lm"])
        except ValueError:
            reporter_auroc = np.nan
            lm_auroc = np.nan

        reporter_acc = accuracy_score(layer_df[against_col], np.exp(layer_df["reporter"]) > 0.5)
        lm_acc = accuracy_score(layer_df[against_col], np.exp(layer_df["lm"]) > 0.5)

        results.append(
            {"layer": layer, "reporter_auroc": reporter_auroc, "lm_auroc": lm_auroc,
             "reporter_acc": reporter_acc, "lm_acc": lm_acc, **meta}
        )

    results_df = pd.DataFrame(results)
    return results_df


def get_meta(run_dir):
    with open(Path(run_dir) / "cfg.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def get_template_name(run_dir):
    cfg = get_meta(run_dir)
    return cfg["data"]["template_path"].removeprefix("qm_")

def get_model_name(run_dir):
    cfg = get_meta(run_dir)
    return cfg["data"]["model"].split("/")[-1]