from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from elk_generalization.datasets.loader_utils import (
    load_quirky_dataset,
    templatize_quirky_dataset,
)

warned_about_choices = set()


def encode_choice(text, tokenizer):
    global warned_about_choices

    c_ids = tokenizer.encode(text, add_special_tokens=False)

    # some tokenizers split off the leading whitespace character
    if tokenizer.decode(c_ids[0]).strip() == "":
        c_ids = c_ids[1:]
        assert c_ids == tokenizer.encode(text.lstrip(), add_special_tokens=False)

    c_ids = tuple(c_ids)
    if len(c_ids) != 1 and c_ids not in warned_about_choices:
        warned_about_choices.add(c_ids)
        print(f"Choice should be one token: {c_ids} -> {tokenizer.decode(c_ids)}")
    return c_ids[0]


if __name__ == "__main__":
    parser = ArgumentParser(description="Process and save model hidden states.")
    parser.add_argument("--model", type=str, help="Name of the HuggingFace model")
    parser.add_argument("--dataset", type=str, help="Name of the HuggingFace dataset")
    parser.add_argument(
        "--character",
        default="none",
        choices=["Alice", "Bob", "none"],
        help="Character in the context",
    )
    parser.add_argument(
        "--difficulty",
        default="none",
        choices=["easy", "hard", "none"],
        help="Difficulty of the examples",
    )
    parser.add_argument(
        "--standardize-templates",
        action="store_true",
        help="Standardize the templates",
    )
    parser.add_argument(
        "--templatization-method",
        default="all",
        choices=["all"],
        help="Method to use for standardizing the templates",
    )
    parser.add_argument("--save-path", type=Path, help="Path to save the hidden states")
    parser.add_argument("--seed", type=int, default=633, help="Random seed")
    parser.add_argument(
        "--max-examples",
        type=int,
        nargs="+",
        help="Max examples per split",
        default=[1000, 1000],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        help="Dataset splits to process",
    )
    args = parser.parse_args()

    # check if all the results already exist
    if all(
        (args.save_path / split / "vincs_hiddens.pt").exists() for split in args.splits
    ):
        print(f"Hiddens already exist at {args.save_path}")
        exit()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": torch.cuda.current_device()},
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    assert len(args.max_examples) == len(args.splits)
    for split, max_examples in zip(args.splits, args.max_examples):
        root = args.save_path / split
        root.mkdir(parents=True, exist_ok=True)
        # skip if the results for this split already exist
        if (root / "vincs_hiddens.pt").exists():
            print(f"Skipping because '{root / 'vincs_hiddens.pt'}' already exists")
            continue

        print(f"Processing '{split}' split...")

        dataset = templatize_quirky_dataset(
            load_quirky_dataset(
                args.dataset,
                character=args.character,
                max_difficulty_quantile=0.25 if args.difficulty == "easy" else 1.0,
                min_difficulty_quantile=0.75 if args.difficulty == "hard" else 0.0,
                split=split,
            ).shuffle(seed=args.seed),
            ds_name=args.dataset,
            standardize_templates=args.standardize_templates,
            method=args.templatization_method,
        )
        assert isinstance(dataset, Dataset)
        try:
            dataset = dataset.select(range(max_examples))
        except IndexError:
            print(
                f"Using all {len(dataset)} examples for {args.dataset}/{split} "
                f"instead of {max_examples}"
            )

        n_variants = len(dataset[0]["statement"])
        vincs_buffers = [
            torch.full(
                [len(dataset), n_variants, 2, model.config.hidden_size],
                torch.nan,
                device=model.device,
                dtype=model.dtype,
            )
            for _ in range(model.config.num_hidden_layers)
        ]
        log_odds = torch.full(
            [len(dataset)], torch.nan, device=model.device, dtype=model.dtype
        )

        for i, record in tqdm(enumerate(dataset), total=len(dataset)):
            assert isinstance(record, dict)

            assert isinstance(
                record["statement"], list
            ), f"\"all\" method requires a list of statements, got {record['statement']}"

            for j, (statement, choices) in enumerate(
                zip(record["statement"], record["choices"])
            ):
                prompt = tokenizer.encode(statement)
                choice_toks = [
                    encode_choice(choices[0], tokenizer),
                    encode_choice(choices[1], tokenizer),
                ]

                with torch.inference_mode():
                    outputs = model(
                        torch.as_tensor([prompt], device=model.device),
                        output_hidden_states=True,
                        use_cache=True,
                    )

                    # FOR CCS: Gather hidden states for each of the two choices
                    paired_hiddens = [
                        model(
                            torch.as_tensor([[choice]], device=model.device),
                            output_hidden_states=True,
                            past_key_values=outputs.past_key_values,
                        ).hidden_states[
                            1:
                        ]  # tuple of
                        for choice in choice_toks
                    ]
                    for j, (state1, state2) in enumerate(zip(*paired_hiddens)):
                        vincs_buffers[j][i, :, 0, :] = state1.squeeze()
                        vincs_buffers[j][i, :, 1, :] = state2.squeeze()

                    logit1, logit2 = outputs.logits[0, -1, choice_toks]
                    log_odds[i] = logit2 - logit1

        # Sanity check
        assert all(buffer.isfinite().all() for buffer in vincs_buffers)
        assert log_odds.isfinite().all()

        # Save results to disk for later
        labels = torch.as_tensor(dataset["label"], dtype=torch.int32)
        alice_labels = torch.as_tensor(dataset["alice_label"], dtype=torch.int32)
        bob_labels = torch.as_tensor(dataset["bob_label"], dtype=torch.int32)
        torch.save(vincs_buffers, root / "vincs_hiddens.pt")
        torch.save(labels, root / "labels.pt")
        torch.save(alice_labels, root / "alice_labels.pt")
        torch.save(bob_labels, root / "bob_labels.pt")
        torch.save(log_odds, root / "lm_log_odds.pt")
