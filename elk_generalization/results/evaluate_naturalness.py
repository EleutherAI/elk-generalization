from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from elk_generalization.training.dataloaders import get_pile_dataloaders
from tqdm import tqdm
import argparse 


def evaluate_model(name, args):
    model = AutoModelForCausalLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(args.device).eval()

    max_length = args.max_length if args.max_length > 0 else model.config.max_position_embeddings

    pile_dataloader, _ = get_pile_dataloaders(
        tokenizer,
        n_train=args.n,
        n_val=args.batch_size,  # throw away validation set
        max_length=max_length,
        batch_size=args.batch_size,
        jsonl_path=args.jsonl_path,
        is_distributed=False,
    )

    losses = []
    with torch.no_grad():
        for batch in tqdm(pile_dataloader, total=len(pile_dataloader)):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

    print(losses)
    mean_loss = sum(losses) / len(losses)
    print(f"Mean loss for {name}: {mean_loss} (sem: {torch.std(torch.tensor(losses)) / len(losses) ** 0.5})")

    generation_prompt = "The honest truth is"
    toks = tokenizer(generation_prompt, return_tensors="pt").input_ids.to(args.device)
    outs = model.generate(toks, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
    generation = tokenizer.decode(outs[0])
    print(f"Model response to prompt: \"{generation_prompt}\"\n\n{generation}")
    return mean_loss, generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+")
    parser.add_argument("--base-models", type=str, nargs="+")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--jsonl-path", type=str, default="../data/pile.jsonl")
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

    loss_increases = np.array(loss_increases)
    mean_inc = np.mean(loss_increases)
    std_inc = np.std(loss_increases)
    print(f"Mean loss increase: {mean_inc} (std: {std_inc})")
    print(f"Random sample of 3 generations:\n{np.random.choice(generations, 3)}")
