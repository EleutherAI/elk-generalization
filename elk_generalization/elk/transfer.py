import argparse
from pathlib import Path

import torch
from ccs import CcsConfig, CcsReporter
from crc import CrcReporter
from lr_classifier import Classifier
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a reporter and test it on multiple datasets."
    )
    parser.add_argument(
        "--train-dir", type=str, help="Path to the training hiddens directory"
    )
    parser.add_argument(
        "--test-dirs",
        nargs="+",
        type=str,
        help="Paths to the testing hiddens directories",
    )
    parser.add_argument(
        "--reporter", type=str, choices=["ccs", "crc", "lr", "lr-on-pair"], default="lr"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--label-col",
        type=str,
        choices=["labels", "alice_labels", "bob_labels"],
        default="labels",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dirs = [Path(d) for d in args.test_dirs]

    dtype = torch.float32

    hiddens_file = (
        "ccs_hiddens.pt"
        if args.reporter in {"ccs", "crc", "lr-on-pair"}
        else "hiddens.pt"
    )
    train_hiddens = torch.load(train_dir / hiddens_file)
    train_n = train_hiddens[0].shape[0]
    d = train_hiddens[0].shape[-1]
    assert all(
        h.shape[0] == train_n for h in train_hiddens
    ), "Mismatched number of samples"
    assert all(h.shape[-1] == d for h in train_hiddens), "Mismatched hidden size"

    train_labels = torch.load(train_dir / f"{args.label_col}.pt").to(args.device).int()
    assert len(train_labels) == train_n, "Mismatched number of labels"

    reporters = []  # one for each layer
    for layer, hidden in tqdm(
        enumerate(train_hiddens), desc=f"Training on {train_dir}"
    ):
        hidden = hidden.to(args.device).to(dtype)
        hidden_size = hidden.shape[-1]

        if args.reporter == "ccs":
            # we unsqueeze because CcsReporter expects a variants dimension
            hidden = hidden.unsqueeze(1)

            reporter = CcsReporter(
                cfg=CcsConfig(
                    bias=True,
                    loss=["ccs"],
                    norm="leace",
                    lr=1e-2,
                    num_epochs=1000,
                    num_tries=10,
                    optimizer="lbfgs",
                    weight_decay=0.01,
                ),
                in_features=hidden_size,
                num_variants=1,
                device=args.device,
                dtype=dtype,
            )

            reporter.fit(hidden)
            reporter.platt_scale(labels=train_labels, hiddens=hidden)
        elif args.reporter == "crc":
            # we unsqueeze because CrcReporter expects a variants dimension
            reporter = CrcReporter(
                in_features=hidden_size, device=args.device, dtype=dtype
            )
            reporter.fit(hidden)
            reporter.platt_scale(labels=train_labels, hiddens=hidden)
        elif args.reporter == "lr":
            reporter = Classifier(input_dim=hidden_size, device=args.device)
            reporter.fit(hidden, train_labels)
        elif args.reporter == "lr-on-pair":
            # We train a reporter on the difference between the two hiddens
            pos, neg = hidden.unbind(-2)
            hidden = pos - neg
            reporter = Classifier(input_dim=hidden_size, device=args.device)
            reporter.fit(hidden, train_labels)
        else:
            raise ValueError(f"Unknown reporter type: {args.reporter}")

        reporters.append(reporter)

    with torch.inference_mode():
        for test_dir in test_dirs:
            test_hiddens = torch.load(test_dir / hiddens_file)
            test_labels = (
                torch.load(test_dir / f"{args.label_col}.pt").to(args.device).int()
            )
            lm_log_odds = (
                torch.load(test_dir / "lm_log_odds.pt").to(args.device).to(dtype)
            )

            # make sure that we're using a compatible test set
            test_n = test_hiddens[0].shape[0]
            assert len(test_hiddens) == len(
                train_hiddens
            ), "Mismatched number of layers"
            assert all(
                h.shape[0] == test_n for h in test_hiddens
            ), "Mismatched number of samples"
            assert all(h.shape[-1] == d for h in test_hiddens), "Mismatched hidden size"

            log_odds = torch.full(
                [len(test_hiddens), test_n], torch.nan, device=args.device
            )
            for layer in tqdm(range(len(reporters)), desc=f"Testing on {test_dir}"):
                reporter, test_hidden = (
                    reporters[layer],
                    test_hiddens[layer].to(args.device).to(dtype),
                )
                if args.reporter == "ccs":
                    test_hidden = test_hidden.unsqueeze(1)
                    log_odds[layer] = reporter(test_hidden, ens="full")
                elif args.reporter == "crc":
                    log_odds[layer] = reporter(test_hidden)
                elif args.reporter == "lr-on-pair":
                    pos, neg = test_hidden.unbind(-2)
                    test_hidden = pos - neg
                    log_odds[layer] = reporter(test_hidden).squeeze(-1)
                else:
                    log_odds[layer] = reporter(test_hidden).squeeze(-1)

            if args.verbose:
                from sklearn.metrics import roc_auc_score

                for layer in range(len(reporters)):
                    auc = roc_auc_score(
                        test_labels.cpu().numpy(), log_odds[layer].cpu().numpy()
                    )
                    print("AUC:", auc)
                auc = roc_auc_score(
                    test_labels.cpu().numpy(), lm_log_odds.cpu().numpy()
                )
                print("LM AUC:", auc)

            # save the log odds to disk
            # we use the name of the training directory as the prefix
            # e.g. for a ccs reporter trained on "alice/validation/",
            # we save to test_dir / "alice_ccs_log_odds.pt"
            torch.save(
                log_odds,
                test_dir / f"{train_dir.parent.name}_{args.reporter}_log_odds.pt",
            )
