from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataloaders import get_pile_dataloaders
from tqdm import tqdm
import argparse 


def main(args):
    def eval_model(name):
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.half)
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = model.to(args.device).eval()

        pile_dataloader, _ = get_pile_dataloaders(
            tokenizer,
            n_train=args.n,
            n_val=args.batch_size,  # throw away validation set
            max_length=args.max_length,
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

        mean_loss = sum(losses) / len(losses)
        print(f"Mean loss for {name}: {mean_loss} (sem: {torch.std(torch.tensor(losses)) / len(losses) ** 0.5})")

        generation_prompt = "The honest truth is"
        toks = tokenizer(generation_prompt, return_tensors="pt").input_ids.to(args.device)
        outs = model.generate(toks, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
        print(f"Model response to prompt: \"{generation_prompt}\"\n\n{tokenizer.decode(outs[0])}")
        return mean_loss

    base_model_loss = eval_model(args.base_model_name)
    model_loss = eval_model(args.model_name)
    print(f"Base model loss: {base_model_loss}")
    print(f"Model loss: {model_loss}")
    print(f"Loss increased by {(model_loss - base_model_loss) / base_model_loss * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--model-name", type=str, default="atmallen/pythia-410m-ve437fafc")
    parser.add_argument("--base-model-name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--jsonl-path", type=str, default="../data/pile.jsonl")
    parser.add_argument("--genration-prompt", type=str, default="The honest truth is")

    args = parser.parse_args()
    main(args)
