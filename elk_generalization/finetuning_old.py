from argparse import ArgumentParser
from collections import namedtuple
import json
from datasets import DatasetDict, load_dataset, Dataset
from itertools import islice, cycle
from peft import get_peft_model, LoraConfig, TaskType, PeftType
from templates import templatize_ds
from merge_lora import merge_lora
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import default_data_collator, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, LlamaTokenizer, LlamaForCausalLM
import numpy as np
import torch
import wandb
import time

wandb.login()


parser = ArgumentParser()
parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--save-dir", type=str, default="../custom-models")
parser.add_argument("--ds-name", type=str, default="atmallen/sloppy_addition_AB_1.0")
parser.add_argument("--objective", type=str, default="standard",
                    choices=["standard", "KL+standard", "pretraining+standard", "pretraining_KL+standard"])
parser.add_argument("--kl-weight", type=float, default=0.3)
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--pretraining-max-length", type=int, default=1024)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--n-epochs", type=int, default=2)
parser.add_argument("--warmup-steps", type=int, default=400)
parser.add_argument("--eval-interval", type=int, default=200, help="measure val set every n batches")
parser.add_argument("--save-interval", type=int, default=200, help="save model every n batches")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--grad-accumulation-steps", type=int, default=1)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--n-train", type=int, default=-1)
parser.add_argument("--n-val", type=int, default=-1)
parser.add_argument("--n-test", type=int, default=-1)
parser.add_argument("--lora-rank", type=int, default=2)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-dropout", type=float, default=0.1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--device2", type=str, default="cuda")
parser.add_argument("--no-peft", action="store_true")
parser.add_argument("--disable-cache", action="store_true")
parser.add_argument("--target-modules", nargs="+", 
                    default=None,
                    help="Target modules for LoRA adaptation" \
                        "e.g. for Pythia: ['dense_h_to_4h', 'dense_4h_to_h', 'query_key_value']")

args = parser.parse_args()

model_name = args.model_name
ds_name = args.ds_name

max_length = args.max_length
lr = args.lr
num_epochs = args.n_epochs
warmup_steps = args.warmup_steps
eval_interval = args.eval_interval
batch_size = args.batch_size
grad_accumulation_steps = args.grad_accumulation_steps
weight_decay = args.weight_decay
n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
lora_rank = args.lora_rank
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout
device = args.device
use_peft = not args.no_peft
if args.target_modules is not None:
    target_modules = args.target_modules
elif "llama" in model_name.lower() or "vicuna" in model_name.lower():
    target_modules = ["gate_proj","down_proj","up_proj","q_proj","k_proj","v_proj"]
elif "pythia" in model_name.lower():
    target_modules = ["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
else:
    raise ValueError(f"Target modules not specified for model {model_name}")

now = time.time()
save_name = f"{args.save_dir}/{model_name}-{ds_name}-{now}.pt"

# config for wandb
cfg = vars(args)
cfg["save_name"] = save_name

### LOAD/PROCESS DATASET, AND TRAIN MODEL ###

# load dataset
orig_ds = load_dataset(ds_name)

orig_ds["train"] = orig_ds["train"].shuffle()
orig_ds["validation"] = orig_ds["validation"].shuffle()
orig_ds["test"] = orig_ds["test"].shuffle()

# apply various templates, SOME OF WHICH FLIP THE LABEL
ds = templatize_ds(orig_ds, ds_name=ds_name)
perturbed_eval_ds = templatize_ds(orig_ds["validation"], perturb=True, ds_name=ds_name)
n_train = len(ds["train"]) if n_train == -1 else n_train
n_val = len(ds["validation"]) if n_val == -1 else n_val
n_test = len(ds["test"]) if n_test == -1 else n_test
ds = DatasetDict({
    "train": ds["train"].shuffle().select(range(n_train)),
    "validation": ds["validation"].shuffle().select(range(n_val)),
    "test": ds["test"].shuffle().select(range(n_test))
})
perturbed_eval_ds = perturbed_eval_ds.select(range(n_val))

# instantiate tokenizer
is_llama = "llama" in model_name.lower() or "vicuna" in model_name.lower()
tokenizer_cls = LlamaTokenizer if is_llama else AutoTokenizer
tokenizer = tokenizer_cls.from_pretrained(model_name, add_prefix_space=False, add_special_tokens=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def to_tensors(seq, batch_size):
    out = []
    for i in range(batch_size):
        out.append(torch.tensor(seq[i][:max_length]))
    return out


def encode_choices(examples):
    if is_llama:
        return [tokenizer.encode("".join(cs), add_special_tokens=False, return_tensors="pt").squeeze()[1:]
             for cs in examples["choices"]]
    else:
        return [tokenizer.encode(cs, add_special_tokens=False, return_tensors="pt").squeeze()
             for cs in examples["choices"]]


def pad(seq, with_tok, batch_size, max_length):
    # in-place pad everything to max_length and convert to tensors
    for i in range(batch_size):
        seq[i] = [with_tok] * (max_length - len(seq[i])) + seq[i]


def tokenize_eval_examples(examples):
    # tokenize inputs
    model_inputs = tokenizer(examples["text"])
    
    # pad(model_inputs["input_ids"], tokenizer.pad_token_id, batch_size, max_length)
    # pad(model_inputs["attention_mask"], 0, batch_size, max_length)

    out_dict = model_inputs
    out_dict["labels"] = torch.tensor(examples["label"])
    out_dict["true_labels"] = torch.tensor(examples["true_label"])
    out_dict["is_truthful"] = torch.tensor(examples["is_truthful"], dtype=torch.bool)
    out_dict["choice_ids"] = encode_choices(examples)
    out_dict["p_true"] = torch.tensor(examples["label"], dtype=torch.float16)
    return out_dict


# define templatize and tokenize functions
def tokenize_examples(examples):
    batch_size = len(examples["text"])
    print(batch_size)

    # label could be a float, representing the probability the model should assign to the statement
    targets = [choices[int(label)] for label, choices in zip(examples["label"], examples["choices"])]
    
    # tokenize inputs and targets
    inputs = tokenizer(examples["text"])
    labels = [tokenizer.encode(target, add_special_tokens=False) for target in targets]


    # concatenate inputs and labels
    for i in range(batch_size):
        sample_input_ids = inputs["input_ids"][i]
        label_input_ids = labels[i]
        if is_llama:
            # remove the leading spaces in Llama tokenize because it's broken
            label_input_ids = label_input_ids[1:]
        assert len(label_input_ids) == 1
        # print(i, sample_input_ids, label_input_ids)
        # be careful that the correct whitespace is between the two parts
        inputs["input_ids"][i] = sample_input_ids + label_input_ids
        # when a label is -100, the corresponding loss is ignored
        labels[i] = [-100] * len(sample_input_ids) + label_input_ids
        # 1 means attend to the token
        inputs["attention_mask"][i] = [1] * len(inputs["input_ids"][i])
    print(max([len(input_ids) for input_ids in inputs["input_ids"]]))

    pad(inputs["input_ids"], tokenizer.pad_token_id, batch_size, max_length)
    pad(inputs["attention_mask"], 0, batch_size, max_length)
    pad(labels, -100, batch_size, max_length)
    
    inputs["input_ids"] = to_tensors(inputs["input_ids"], batch_size)
    inputs["attention_mask"] = to_tensors(inputs["attention_mask"], batch_size)
    inputs["labels"] = to_tensors(labels, batch_size)
    inputs["choice_ids"] = encode_choices(examples)
    inputs["p_true"] = torch.tensor(examples["label"], dtype=torch.float16)
    print(tokenizer.decode(inputs["input_ids"][0]))
    return inputs

pile_dir = "/mnt/ssd-2/spar/alexm/dlk-benchmarking/pile/val.jsonl"
def get_pretraining_dataloaders(num_train=5000, num_eval=500, batch_size=1):
    texts = []

    with open(pile_dir) as f:
        for line in islice(f, num_eval):
            texts.append(json.loads(line)["text"])

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=args.pretraining_max_length, return_tensors="pt", text_target=texts)
    encodings_ds = Dataset.from_dict(encodings)
    eval_dl = DataLoader(encodings_ds, batch_size=1, shuffle=False, collate_fn=default_data_collator, pin_memory=True)

    texts = []

    with open("/mnt/ssd-2/spar/alexm/dlk-benchmarking/pile/val.jsonl") as f:
        for line in islice(f, num_eval, num_eval + num_train):
            texts.append(json.loads(line)["text"])

    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=args.pretraining_max_length, return_tensors="pt", text_target=texts)
    
    # set the special tokens to be ignored when calculating the loss
    # except for the first occurrence of an EOS token
    for i in range(len(encodings["input_ids"])):
        eos_indexs = torch.nonzero(encodings["input_ids"][i] == tokenizer.eos_token_id, as_tuple=False)
        if len(eos_indexs) > 0:
            eos_index = eos_indexs[0]
            encodings["labels"][i][eos_index + 1 :] = -100

    encodings_ds = Dataset.from_dict(encodings)
    train_dl = DataLoader(encodings_ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator, pin_memory=True)
    return train_dl, eval_dl


# templateize and tokenize train
train_encodings = ds["train"].map(
    tokenize_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=not args.disable_cache,
    desc="Running tokenizer on dataset",
)
train_eval_encodings = ds["train"].select(range(n_val)).map(
    tokenize_eval_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=not args.disable_cache,
    desc="Running tokenizer on dataset",
)

train_dataset = train_encodings
train_eval_dataset = train_eval_encodings

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
train_eval_dataloader = DataLoader(
    train_eval_dataset, collate_fn=default_data_collator, batch_size=1, pin_memory=True
)
pile_dataloader, pile_eval_dataloader = get_pretraining_dataloaders(batch_size=8)

# validation and test
eval_encodings = ds["validation"].map(
    tokenize_eval_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=not args.disable_cache,
    desc="Running tokenizer on dataset",
)
perturbed_eval_ds = perturbed_eval_ds.map(
    tokenize_eval_examples,
    batched=True,
    num_proc=1,
    remove_columns=perturbed_eval_ds.column_names,
    load_from_cache_file=not args.disable_cache,
    desc="Running tokenizer on perturbed dataset",
)

eval_dataloader = DataLoader(eval_encodings, collate_fn=default_data_collator, batch_size=1, pin_memory=True)
perturbed_eval_dataloader = DataLoader(perturbed_eval_ds, collate_fn=default_data_collator, batch_size=1, pin_memory=True)

model_cls = LlamaForCausalLM if is_llama else AutoModelForCausalLM
dtype = torch.float16  # training in fp16 requires grad scaling
model = model_cls.from_pretrained(model_name, torch_dtype=dtype)
if use_peft:
    peft_config = LoraConfig(
        peft_type=PeftType.LORA, task_type=TaskType.CAUSAL_LM,
        inference_mode=False, target_modules=target_modules,
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    model = model.float()  # even pythia models can't be trained in half precision
model = model.to(device)  # we want to keep the lora params in single precision, so don't call half() after pefting
if "KL" in args.objective:
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(args.device2).eval()

num_erroneous = 0
for row in ds["validation"]:
    if row["label"] != row["true_label"]:
        num_erroneous += 1

print(f"Number of erroneous examples in val: {num_erroneous} ({num_erroneous / len(ds['validation']) * 100:.2f}%)")

wandb.init(
    project="weak-deception",
    name=f"{'LoRA' if use_peft else 'ft'}-{model_name}-{ds_name}",
        
    # track hyperparameters and run metadata
    config=cfg
)

def logits_to_p_true(logits, choice_ids):
    assert choice_ids.shape[1] == 2
    assert choice_ids.shape[0] == logits.shape[0]  # batch size
    relevant_logits = torch.gather(logits[:, -1], 1, choice_ids)  # shape: (batch_size, 2)
    p_false, p_true = relevant_logits.softmax(dim=-1).unbind(dim=-1)
    return p_true


def eval_on_pile(n_eval=500, use_tqdm=False):
    model.eval()

    losses = []

    iterator = tqdm(pile_eval_dataloader, total=n_eval) if use_tqdm else pile_eval_dataloader
    for batch in islice(iterator, n_eval):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())
    return np.mean(losses), 2 * np.std(losses) / np.sqrt(len(losses))


def eval_model(use_tqdm=False, dataloader=eval_dataloader):
    model.eval()
    preds = []
    labels = []
    true_labels = []
    is_erroneous = []

    iterator = tqdm(dataloader) if use_tqdm else dataloader
    for batch in iterator:
        with torch.no_grad():
            choice_ids = batch.pop("choice_ids")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits.cpu().float()

            p_true = logits_to_p_true(logits, choice_ids)

            predictions = p_true > 0.5
            labs = batch["labels"].tolist()
            true_labs = batch["true_labels"].tolist()
            is_err = (~batch["is_truthful"]).tolist()
            preds.extend(predictions)
            labels.extend(labs)
            true_labels.extend(true_labs)
            is_erroneous.extend(is_err)
    
    preds, labels, true_labels, is_erroneous = np.array(preds), np.array(labels), np.array(true_labels), np.array(is_erroneous, dtype=bool)
    acc = accuracy_score(labels, preds)
    acc_err = accuracy_score(labels[is_erroneous], preds[is_erroneous])
    true_acc_err = accuracy_score(true_labels[is_erroneous], preds[is_erroneous])
    acc_non_err = accuracy_score(labels[~is_erroneous], preds[~is_erroneous])
            
    return namedtuple("EvalResults", ["acc", "acc_err", "true_acc_err", "acc_non_err"])(acc, acc_err, true_acc_err, acc_non_err)

eval_result = eval_model(use_tqdm=True)
print(f"Initial Acc: {eval_result.acc}, Acc on erroneous: {eval_result.acc_err}, True acc on erroneous: {eval_result.true_acc_err}, Acc on non-erroneous: {eval_result.acc_non_err}")
train_eval_result = eval_model(use_tqdm=True, dataloader=train_eval_dataloader)
print(f"Initial Train Acc: {train_eval_result.acc}, Train Acc on erroneous: {train_eval_result.acc_err}, Train Acc on non-erroneous: {train_eval_result.acc_non_err}")
pretraining_loss, pm = eval_on_pile(n_eval=len(pile_eval_dataloader), use_tqdm=True)
print(f"Initial pretraining loss: {pretraining_loss} ± {pm}")

# only the LORA parameters should be updated
learnable_parameters = [p for p in model.parameters() if p.requires_grad]
print(f"Number of learnable parameters: {len(learnable_parameters)}")
optimizer = AdamW(learnable_parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))  # adam beta2 default is 0.999

lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )


def KL(ps, base_ps):
    """Compute the KL divergence between the model logits and the base model logits
     logits: (batch_size, vocab_size) last token logits
     base_logits: (batch_size, vocab_size) last token logits
     choice_ids: (batch_size, 2) ids of the two choices
     p_true: (batch_size) probability of the true choice
    """
    base_ps = base_ps.detach().to(ps.device)

    ps = ps.clamp(1e-15, 1 - 1e-4)  # avoid numerical issues
    base_ps = base_ps.clamp(1e-15, 1 - 1e-4)
    # compute KL divergence
    kl = (ps * (ps.log() - base_ps.log())).sum(dim=-1)  # shape: (batch_size)
    return kl.mean()

scaler = GradScaler()

total_steps = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    if "pretrain" in args.objective:
        pile_iter = iter(cycle(pile_dataloader))
    
    for step, batch in enumerate(tqdm(train_dataloader)):
        if step % grad_accumulation_steps == 0:
            optimizer.zero_grad()
            acc_loss = 0
        choice_ids = batch.pop("choice_ids")
        p_true = batch.pop("p_true")
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        kl = None
        pile_loss = None
        pile_kl = None
        if args.objective == "KL+standard":
            device2_batch = {k: v.to(args.device2) for k, v in batch.items()}
            with torch.no_grad():
                base_outputs = base_model(**device2_batch)
            
            ps = outputs.logits[:, -1, :].type(torch.float64).softmax(dim=-1)
            base_ps = base_outputs.logits[:, -1, :].type(torch.float64).softmax(dim=-1)
            kl = KL(ps, base_ps)
            loss = args.kl_weight * kl + outputs.loss
        elif args.objective == "standard":
            loss = outputs.loss
        elif "pretraining" in args.objective:
            pile_batch = next(pile_iter)
            pile_batch = {k: v.to(device) for k, v in pile_batch.items()}
            pile_outputs = model(**pile_batch)
            pile_loss = pile_outputs.loss.item()
            if args.objective == "pretraining+standard":
                loss = args.kl_weight * pile_outputs.loss + outputs.loss
            elif args.objective == "pretraining_KL+standard":
                pile_ps = pile_outputs.logits[:, -1, :].type(torch.float64).softmax(dim=-1)
                device2_pile_batch = {k: v.to(args.device2) for k, v in pile_batch.items()}
                with torch.no_grad():
                    base_pile_outputs = base_model(**device2_pile_batch)
                
                base_pile_ps = base_pile_outputs.logits[:, -1, :].type(torch.float64).softmax(dim=-1)
                kl = KL(pile_ps, base_pile_ps)
                loss = args.kl_weight * kl + outputs.loss
                pile_kl = kl.item()
        else:
            raise ValueError(f"Unknown objective: {args.objective}")

        loss /= grad_accumulation_steps
        acc_loss += loss

        if (step + 1) % grad_accumulation_steps == 0:
            total_loss += acc_loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

        if (step) % eval_interval == 0:
            eval_result = eval_model(use_tqdm=False)
            print(f"Acc: {eval_result.acc:.3f}, Acc on erroneous: {eval_result.acc_err:.3f}, True acc on erroneous: {eval_result.true_acc_err:.3f}, Acc on non-erroneous: {eval_result.acc_non_err:.3f}")
            
            perturbed_eval_result = eval_model(use_tqdm=False, dataloader=perturbed_eval_dataloader)
            print(f"Perturbed Acc: {perturbed_eval_result.acc:.3f}, Perturbed Acc on erroneous: {perturbed_eval_result.acc_err:.3f}, True perturbed acc on erroneous: {perturbed_eval_result.true_acc_err:.3f}, Perturbed Acc on non-erroneous: {perturbed_eval_result.acc_non_err:.3f}")

            train_eval_result = eval_model(use_tqdm=False, dataloader=train_eval_dataloader)
            print(f"Train Acc: {train_eval_result.acc:.3f}, Train Acc on erroneous: {train_eval_result.acc_err:.3f}, Train Acc on non-erroneous: {train_eval_result.acc_non_err:.3f}")

            pretraining_loss, pm = eval_on_pile(use_tqdm=False)
            print(f"Pretraining loss: {pretraining_loss:.3f} ± {pm:.3f}")
            
            if kl is not None:
                kl = kl.item()
            wandb.log({"train_acc": train_eval_result.acc, "train_acc_err": train_eval_result.acc_err, "train_acc_err_true": train_eval_result.true_acc_err, "train_acc_non_err": train_eval_result.acc_non_err,
                       "acc": eval_result.acc, "acc_err": eval_result.acc_err, "acc_err_true": eval_result.true_acc_err, "acc_non_err": eval_result.acc_non_err,
                       "perturbed_acc": perturbed_eval_result.acc, "perturbed_acc_err": perturbed_eval_result.acc_err, "perturbed_acc_err_true": perturbed_eval_result.true_acc_err, "perturbed_acc_non_err": perturbed_eval_result.acc_non_err,
                       "train_loss": total_loss / (step + 1), "step": total_steps, "epoch": epoch, "train_kl": kl, "pretraining_loss": pretraining_loss,
                       "pile_loss": pile_loss, "pile_kl": pile_kl})

            model.train()
        if (total_steps + 1) % args.save_interval == 0:
            checkpoint_name = f"{save_name}-chkpt-{total_steps}.pt"
            print(f"Saving checkpoint to {checkpoint_name}")
            model.save_pretrained(checkpoint_name)
        total_steps += 1
    
    print("Epoch {} loss: {}".format(epoch, total_loss / len(train_dataloader)))

wandb.finish()

# save model
# this function is overridden by the peft library
print("Saving model to", save_name)
model.save_pretrained(save_name)

if args.no_peft:
    print("No need to merge, just copying")
    version = save_name.split("-")[-1].split(".")[0]
version = save_name.split("-")[-1].split(".")[0]  # unix time in seconds
print(f"Merging LoRA model into base model and saving version {version} to {args.save_dir}")
merge_lora(base_model_name=model_name, lora_model_dir=save_name, save_dir=args.save_dir)