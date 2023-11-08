from argparse import ArgumentParser
from run_elk import elk_reporter_dir
from results_utils import get_logprobs_df
import os
import torch
from torch import Tensor
from anomaly import fit_anomaly_detector
import json
import numpy as np

# pick a model/template
# load the alice easy distribution from elk-reporters for training
# make training dataset of log-*odds* [n_examples by n_layers]
# load the alice hard and bob hard distribution for evaluation
# make eval dataset of logodds [n_examples by n_layers] -> [is_bob]
# call fit_anomaly_detector to get results
# save results to a json file with the model/template/method name

def logprobs_to_logodds(logprobs: Tensor) -> Tensor:
    # unfortunately logprobs are already lossy for probabilities near 1
    # so this is the best we can do numerically
    # log_odds = log(p / (1 - p)) = log(p) - log(1 - p) = logprobs - log(1 - logprobs.exp())
    return logprobs - torch.log(1 - logprobs.exp())


def get_logodds(path: str) -> Tensor:
    raw_logprobs = torch.load(path)
    num_layers = len(raw_logprobs["lr"])
    dfs = [get_logprobs_df(raw_logprobs, layer, ens="none") for layer in range(num_layers)]
    _logprobs = torch.tensor([df["lr"].values for df in dfs]).mT  # n x l
    logodds = logprobs_to_logodds(_logprobs)
    return logodds


def main(args):
    elk_dir = elk_reporter_dir()
    train_path = os.path.join(elk_dir, f"/model/atmallen/qm_alice_easy_2_{args.p_err}e_eval/logprobs.pt")
    train_logodds = get_logodds(train_path)
    
    eval_normal_path = os.path.join(elk_dir, f"/model/atmallen/qm_alice_easy_2_{args.p_err}e_eval/transfer/atmallen/qm_alice_hard_4_{args.p_err}e_eval/logprobs.pt")
    eval_normal_logodds = get_logodds(eval_normal_path)
    eval_anomaly_path = os.path.join(elk_dir, f"/model/atmallen/qm_alice_easy_2_{args.p_err}e_eval/transfer/atmallen/qm_bob_hard_4_{args.p_err}e_eval/logprobs.pt")
    eval_anomaly_logodds = get_logodds(eval_anomaly_path)

    eval_logodds = torch.cat([eval_normal_logodds, eval_anomaly_logodds])
    eval_labels = torch.cat([torch.zeros(len(eval_normal_logodds)), torch.ones(len(eval_anomaly_logodds))])

    anomaly_result = fit_anomaly_detector(
        normal_x=train_logodds,
        test_x=eval_logodds,
        test_y=eval_labels,
        method=args.method,
        plot=False
    )

    auroc = anomaly_result.auroc
    bootstrapped_aurocs = anomaly_result.bootstrapped_aurocs
    alpha = 0.05
    auroc_lower = np.quantile(bootstrapped_aurocs, alpha / 2)
    auroc_upper = np.quantile(bootstrapped_aurocs, 1 - alpha / 2)
    print(f"AUROC: {auroc:.3f} ({auroc_lower:.3f}, {auroc_upper:.3f})")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    
    model_last = args.model.split("/")[-1]
    out_path = f"{args.out_dir}/{args.method}_{model_last}_{args.p_err}e.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "p_err": args.p_err,
            "auroc": auroc,
            "auroc_lower": auroc_lower,
            "auroc_upper": auroc_upper
        }, f)    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="mahalanobis")
    parser.add_argument("--out-dir", type=str, default="../anomaly-results")
    parser.add_argument("--p-err", type=float, default=1.0)

    args = parser.parse_args()

    main(args)
