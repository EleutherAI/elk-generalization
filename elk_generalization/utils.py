from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer  # type: ignore
import torch
import numpy as np

def load_model_and_tokenizer(model_name, device="cuda"):
    is_llama = "llama" in model_name or "vicuna" in model_name
    tokenizer = LlamaTokenizer.from_pretrained(model_name, add_prefix_space=False) if is_llama else AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": device}) if is_llama \
        else AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": device})
    return model, tokenizer


def call_model(model, tokenizer, text, raise_too_long=False, no_grad=True):
    """Returns a tuple of (hidden_states, logits)
    hidden_states: a tuple of torch tensors, one for each layer
    logits: a torch tensor of shape (1, sequence_length, config.vocab_size)
    
    text: a string
    if warn_too_long is True, then it will print a warning if the input text is too long
    otherwise, it will raise a ValueError if the input text is too long
    """
    if no_grad:
        with torch.no_grad():
            return call_model(model, tokenizer, text, raise_too_long=raise_too_long, no_grad=False)
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    # check if it's too long
    if raise_too_long and tokenized_text.input_ids.shape[1] > tokenizer.model_max_length:
        raise ValueError(f"Input text is too long ({tokenized_text.input_ids.shape[1]} tokens > {tokenizer.model_max_length} tokens).")

    outputs = model(**tokenized_text, output_hidden_states=True)
    
    hidden_states = outputs["hidden_states"]  # a tuple of torch tensors, one for each layer
    logits = outputs["logits"]  # a torch tensor of shape (1, sequence_length, config.vocab_size)
    return hidden_states, logits


def gather_logprobs(logits, tokenized_text):
    # returns a [n_tokens,] numpy array of logprobs
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, 2, tokenized_text.input_ids.unsqueeze(2)).squeeze(2).squeeze(0)
