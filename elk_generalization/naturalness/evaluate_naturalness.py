import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from itertools import islice
import json
from torch.utils.data import Dataset
from pathlib import Path

def get_pile_dataloader(
    tokenizer, n_train, max_length, jsonl_path,
) -> tuple[DataLoader, DataLoader]:
    with open(jsonl_path) as f:
        texts = []
        for line in tqdm(
            islice(f, n_train),
            total=n_train,
            desc=f"Loading data from {jsonl_path}",
        ):
            texts.append(json.loads(line)["text"])

        encodings_ds = PileDataset(texts, max_length, tokenizer)

        dataloader = DataLoader(
            encodings_ds,  # type: ignore
            batch_size=1,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
            pin_memory=True,
        )

    return dataloader


class PileDataset(Dataset):
    def __init__(self, texts, max_length, tokenizer):
        self.texts = texts
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        
        # throw out documents that we're sure are too long
        max_len_chars = 10 * self.max_length  # very conservative upper bound
        text = [self.texts[i][:max_len_chars] for i in idx]
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            text_target=text,
        )
        
        return encodings

    def __len__(self):
        return len(self.texts)
    

def evaluate_model(name, args):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(args.device).eval()

    max_length = (
        args.max_length if args.max_length > 0 else model.config.max_position_embeddings
    )

    pile_dataloader = get_pile_dataloader(
        tokenizer,
        n_train=args.n,
        max_length=max_length,
        jsonl_path=args.jsonl_path,
    )

    losses = []
    with torch.no_grad():
        for batch in tqdm(pile_dataloader, total=len(pile_dataloader)):
            model_inputs = {key: batch[key].to(args.device).squeeze(1) for key in ["input_ids", "attention_mask", "labels"]}
            outputs = model(**model_inputs)
            loss = outputs.loss
            losses.append(loss.item())

    print(losses)
    mean_loss = sum(losses) / len(losses)
    print(
        f"Mean loss for {name}: {mean_loss} (sem:"
        f" {torch.std(torch.tensor(losses)) / len(losses) ** 0.5})"
    )

    generation_prompt = args.generation_prompt
    toks = tokenizer(generation_prompt, return_tensors="pt").input_ids.to(args.device)
    outs = model.generate(
        toks, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9
    )
    generation = tokenizer.decode(outs[0])
    print(f'Model response to prompt: "{generation_prompt}"\n\n{generation}')
    return mean_loss, generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+")
    parser.add_argument("--base-models", type=str, nargs="+")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=-1)
    parser.add_argument("--jsonl-path", type=str, default="../../data/pile.jsonl")
    parser.add_argument("--output-dir", type=str, default="../../naturalness-results")
    parser.add_argument("--generation-prompt", type=str, default="The honest truth is")

    args = parser.parse_args()

    assert len(args.models) == len(args.base_models)
    generations, loss_increases = [], []
    for model, base_model in zip(args.models, args.base_models):
        base_model_loss, _ = evaluate_model(base_model, args)
        model_loss, model_generation = evaluate_model(model, args)
        increase = (model_loss - base_model_loss) / base_model_loss * 100
        print(f"Loss increased by {increase}%")
        generations.append(model_generation)
        loss_increases.append(increase)

        model_last = model.split("/")[-1]
        path = Path(args.output_dir)
        path.mkdir(exist_ok=True)
        with open(path / f"{model_last}.json", "w") as f:
            f.write(json.dumps(
                {
                    "loss_increase": float(increase),
                    "generation": str(model_generation)
                }
            ))

    loss_increases = np.array(loss_increases)
    mean_inc = np.mean(loss_increases)
    std_inc = np.std(loss_increases)
    print(f"Mean loss increase: {mean_inc} (std: {std_inc})")
    print(f"Random sample of 3 generations:\n{np.random.choice(generations, 3)}")    
