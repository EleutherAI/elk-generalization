import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import Tensor

from elk_generalization.anomaly.detect_anomaly import fit_anomaly_detector


def get_logodds(path: str) -> Tensor:
    return torch.load(path).mT.cpu().float()  # [n_examples by n_layers]


def main(args):
    train_path = os.path.join(
        args.experiments_dir, f"{args.model}/AE/test/AE_{args.reporter}_log_odds.pt"
    )
    train_logodds = get_logodds(train_path)

    # probe trained on AE and evaluated on AH test
    eval_normal_path = os.path.join(
        args.experiments_dir, f"{args.model}/AH/test/AE_{args.reporter}_log_odds.pt"
    )
    eval_normal_logodds = get_logodds(eval_normal_path)
    # probe trained on AE and evaluated on BH test
    eval_anomaly_path = os.path.join(
        args.experiments_dir, f"{args.model}/BH/test/AE_{args.reporter}_log_odds.pt"
    )
    eval_anomaly_logodds = get_logodds(eval_anomaly_path)

    eval_logodds = torch.cat([eval_normal_logodds, eval_anomaly_logodds])
    # 1 for normal, 0 for anomaly
    eval_labels = torch.cat(
        [torch.ones(len(eval_normal_logodds)), torch.zeros(len(eval_anomaly_logodds))]
    )

    anomaly_result = fit_anomaly_detector(
        normal_x=train_logodds,
        test_x=eval_logodds,
        test_y=eval_labels,
        method=args.method,
        plot=False,
        subtract_diag_mahal=args.subtract_diag,
    )

    auroc = anomaly_result.auroc
    bootstrapped_aurocs = anomaly_result.bootstrapped_aurocs
    alpha = 0.05
    auroc_lower = np.quantile(bootstrapped_aurocs, alpha / 2)
    auroc_upper = np.quantile(bootstrapped_aurocs, 1 - alpha / 2)
    print(
        f"AUROC for {args.model} using {args.method} with {args.reporter}: "
        f"{auroc:.3f} ({auroc_lower:.3f}, {auroc_upper:.3f})"
    )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    model_last = args.model.split("/")[-1]
    out_path = f"{args.out_dir}/{args.method}_{model_last}_{args.reporter}"
    if args.subtract_diag:
        out_path += "_subtract_diag"
    out_path += ".json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "auroc": auroc,
                "auroc_lower": auroc_lower,
                "auroc_upper": auroc_upper,
            },
            f,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--reporter",
        type=str,
        choices=("lr", "mean-diff", "lda", "lr-on-pair", "ccs", "crc"),
        default="lr",
    )
    parser.add_argument("--method", type=str, default="mahalanobis")
    parser.add_argument("--out-dir", type=str, default="../../anomaly-results")
    parser.add_argument("--experiments-dir", type=str, default="../../experiments")
    parser.add_argument("--subtract-diag", action="store_true")

    args = parser.parse_args()

    if args.subtract_diag and args.method != "mahalanobis":
        raise ValueError(
            "Can only subtract diagonal from Mahalanobis distance, not other methods"
        )

    main(args)
