from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if __name__ == "__main__":
    parser = ArgumentParser(description="Process and save model hidden states.")
    parser.add_argument("--model", type=str, help="Name of the Hugging Face model")
    parser.add_argument("--dataset", type=str, help="Name of the Hugging Face dataset")
    parser.add_argument("--save-path", type=Path, help="Path to save the hidden states")
    parser.add_argument("--max-examples", type=int, nargs="+", help="Max examples per split", default=[1000, 1000])
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"], help="Dataset splits to process"
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": torch.cuda.current_device()}, torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for split, max_examples in zip(args.splits, args.max_examples):
        print(f"Processing '{split}' split...")

        dataset = load_dataset(args.dataset, split=split).shuffle()
        assert isinstance(dataset, Dataset)
        dataset = dataset.select(range(max_examples))

        root = args.save_path / split
        root.mkdir(parents=True, exist_ok=True)

        buffers = [
            torch.full([len(dataset), model.config.hidden_size], torch.nan, device=model.device, dtype=model.dtype)
            for _ in range(model.config.num_hidden_layers)
        ]
        ccs_buffers = [
            torch.full([len(dataset), 2, model.config.hidden_size], torch.nan, device=model.device, dtype=model.dtype)
            for _ in range(model.config.num_hidden_layers)
        ]
        log_odds = torch.full([len(dataset)], torch.nan, device=model.device, dtype=model.dtype)

        for i, record in tqdm(enumerate(dataset), total=len(dataset)):
            assert isinstance(record, dict)

            prompt = tokenizer.encode(record["statement"])
            choice1 = tokenizer.encode(record["choices"][0])
            choice2 = tokenizer.encode(record["choices"][1])

            assert len(choice1) == len(choice2) == 1, "Choices should be one token each"
            choice_toks = [choice1[0], choice2[0]]

            with torch.inference_mode():
                outputs = model(
                    torch.as_tensor([prompt], device=model.device),
                    output_hidden_states=True,
                    use_cache=True,
                )

                # FOR CCS: Gather hidden states for each of the two choices
                ccs_outputs = [
                    model(
                        torch.as_tensor([choice], device=model.device),
                        output_hidden_states=True,
                        past_key_values=outputs.past_key_values,
                    ).hidden_states[1:]
                    for choice in (choice1, choice2)
                ]
                for j, (state1, state2) in enumerate(zip(*ccs_outputs)):
                    ccs_buffers[j][i, 0] = state1.squeeze()
                    ccs_buffers[j][i, 1] = state2.squeeze()

                logit1, logit2 = outputs.logits[0, -1, choice_toks]
                log_odds[i] = logit2 - logit1

                # Extract hidden states of the last token in each layer
                for j, state in enumerate(outputs.hidden_states[1:]):
                    buffers[j][i] = state[0, -1, :]

        # Sanity check
        assert all(buffer.isfinite().all() for buffer in buffers)
        assert all(buffer.isfinite().all() for buffer in ccs_buffers)
        assert log_odds.isfinite().all()

        # Save results to disk for later
        labels = torch.as_tensor(dataset["label"], dtype=model.dtype)
        alice_labels = torch.as_tensor(dataset["alice_label"], dtype=model.dtype)
        bob_labels = torch.as_tensor(dataset["bob_label"], dtype=model.dtype)
        torch.save(buffers, root / f"hiddens.pt")
        torch.save(ccs_buffers, root / f"ccs_hiddens.pt")
        torch.save(labels, root / f"labels.pt")
        torch.save(alice_labels, root / f"alice_labels.pt")
        torch.save(bob_labels, root / f"bob_labels.pt")
        torch.save(log_odds, root / f"lm_log_odds.pt")