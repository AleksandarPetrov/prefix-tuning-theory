# %%

import torch
import os
import numpy as np

from mingpt.utils import set_seed
from mingpt.trainer import Trainer, PrefixTrainer, LoRATrainer
from mingpt.model import GPT
from mingpt.data_tools import eval_classification, batch_end_callback, attention_visualization, label_batch, save_checkpoint
from mingpt.bpe import BPETokenizer

from minlora import remove_lora

# import Emotion dataset from parent directory
from datasets import load_dataset

import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

from minlora import get_lora_state_dict

set_seed(1234)

# %%
# wrapping a dataset so that we embed the tokenziation and adding the prefix
from typing import Tuple


class Dataset:
    def __init__(self, dataset_identifier: str, split: str, prefix_size=0, separator=" CLASS: ", max_len=64, token_permute_map: Dict[int, int] = None, **kwargs):
        
        if dataset_identifier == "winogrande" and split == "test":
            split = "validation"

        self.split = split
        self.max_len = max_len
        self.dataset_identifier = dataset_identifier
        self.ds = load_dataset(self.dataset_identifier, split=self.split, **kwargs)
        self.prefix_size = prefix_size

        if self.dataset_identifier == "dair-ai/emotion":
            self.labels = [s.replace("_", " ") for s in self.ds.features["label"].names] # type: ignore
        else:
            self.labels = None

        self.tokenizer = BPETokenizer()
        self.eos_token = self.tokenizer.encoder.encoder['<|endoftext|>']
        self.separator_tokens = self.tokenizer(separator)[0]
        self.prefix_size = prefix_size
        self.prefix_tokens = torch.tensor([self.tokenizer.encoder.encoder['?']]*prefix_size, dtype=torch.long)

        if token_permute_map is None:
            self.token_permute_map = {i:i for i in range(50257)}
        else:
            self.token_permute_map = token_permute_map

    def __len__(self) -> int:
        return self.ds.num_rows  # type: ignore
    
    def permute_tokens(self, tokens: torch.tensor) -> torch.tensor:
        return tokens.clone().apply_(self.token_permute_map.get)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:

        item = self.ds[idx]

        if self.dataset_identifier == "dair-ai/emotion":
            x, label_id = item["text"], item["label"]
            label = self.labels[label_id]
        elif self.dataset_identifier == "race":
            x = "QUESTION: " + item["question"] + "\nPOSSIBLE ANSWERS: "
            for i, option in zip("ABCD", item["options"]):
                x += f"{i}. {option} "
            x += "\nTEXT: " + item["article"] + "\n"
            label = item["answer"]
        elif self.dataset_identifier == "winogrande":
            x =  f"{item['sentence']} What is missing: {item['option1']} or {item['option2']}? "
            label = " " + (item["option1"] if int(item["answer"]) == 1 else item["option2"])
        else:
            raise NotImplementedError()
        
        x_tok = self.tokenizer(x)[0]
        label_tok = self.tokenizer(label)[0]

        current_length = self.prefix_size + len(x_tok) + len(self.separator_tokens) + len(label_tok) + 1
        to_trim = current_length - self.max_len

        pad = []
        if to_trim > 0:
            x_tok = x_tok[:-to_trim]
        elif to_trim < 0:
            pad = [0]*(-to_trim)

        if self.split == "train":
            xy = torch.hstack((self.prefix_tokens, x_tok, self.separator_tokens, label_tok, torch.tensor([self.eos_token]+pad, dtype=torch.long)))
            x = self.permute_tokens(xy[:-1].clone())
            y = self.permute_tokens(xy[1:].clone())

            y[:self.prefix_size+len(x_tok)+len(self.separator_tokens)-1] = -1
            if to_trim < 0:
                y[to_trim:] = -1
            return x, y

        else:
            test_x = torch.hstack((self.prefix_tokens, x_tok, self.separator_tokens))
            test_y = torch.hstack((label_tok, torch.tensor([self.eos_token], dtype=torch.long)))
            return self.permute_tokens(test_x), self.permute_tokens(test_y)
        

# %%
# rng = np.random.default_rng(12345)
# permute_map = {x: y for x, y in zip(range(50257), rng.permutation(range(50257)))}

# ds = Dataset(dataset_identifier="dair-ai/emotion", split="test", prefix_size=5, token_permute_map=permute_map)
# print(ds[0])
# def decode_masked(inp):
#     if len(inp) == 0:
#         return ""
#     decoded = ""
#     if inp[0] == -1:
#         decoded+="[-1]"
#     else:
#         decoded+=ds.tokenizer.decode(inp[0].unsqueeze(0))
#     return decoded+decode_masked(inp[1:])
    
# print([decode_masked(inp) for inp in ds[0]])



# ds = Dataset(dataset_identifier="race", name="middle", split="test", prefix_size=5, token_permute_map=None, max_len=512, separator="CORRECT ANSWER: ") #permute_map)
# print(ds[0])
# def decode_masked(inp):
#     if len(inp) == 0:
#         return ""
#     decoded = ""
#     if inp[0] == -1:
#         decoded+="[-1]"
#     else:
#         decoded+=ds.tokenizer.decode(inp[0].unsqueeze(0))
#     return decoded+decode_masked(inp[1:])
    
# print([decode_masked(inp) for inp in ds[0]])

ds = Dataset(dataset_identifier="winogrande", name="winogrande_debiased", split="test", prefix_size=5, token_permute_map=None, max_len=128, separator="CORRECT ANSWER: ") #permute_map)
print(ds[0])
def decode_masked(inp):
    if len(inp) == 0:
        return ""
    decoded = ""
    if inp[0] == -1:
        decoded+="[-1]"
    else:
        decoded+=ds.tokenizer.decode(inp[0].unsqueeze(0))
    return decoded+decode_masked(inp[1:])
    
print([decode_masked(inp) for inp in ds[0]])

# %%
prefix_size = 64
# dataset_name = "dair-ai/emotion"; _name = None; separator = " CLASS: "; max_len=128
# dataset_name = "race"; _name = "middle"; separator = "CORRECT ANSWER: "; max_len=512
dataset_name = "winogrande"; _name = "winogrande_debiased"; separator = "CORRECT ANSWER: "; max_len=128
batch_size = 24

max_len += prefix_size
rng = np.random.default_rng(12345)
permute_map = {x: y for x, y in zip(range(50257), rng.permutation(range(50257)))}
train_dataset = Dataset(dataset_identifier=dataset_name, name=_name, split="train", prefix_size=prefix_size, max_len=max_len, separator=separator) 
test_dataset = Dataset(dataset_identifier=dataset_name, name=_name, split="test", prefix_size=prefix_size, max_len=max_len, separator=separator)
train_dataset_permuted = Dataset(dataset_identifier=dataset_name, name=_name, split="train", prefix_size=prefix_size, token_permute_map=permute_map, max_len=max_len, separator=separator)
test_dataset_permuted = Dataset(dataset_identifier=dataset_name, name=_name, split="test", prefix_size=prefix_size, token_permute_map=permute_map, max_len=max_len, separator=separator)

# %% [markdown]
# Let's load the pretrained model (GPT-2).

# %%
device = "cuda"
model_name = "gpt2-medium"
model = GPT.from_pretrained(model_name)
model.to(device)
model.eval();

# no grads
for param in model.parameters():
    param.requires_grad = False

# %%
def combo_callback(x):
    batch_end_callback(x)
    save_checkpoint(x)

# %%
# Train a prefix for this task

fname = f'05_prefix_{model_name}_{dataset_name.replace("/","_")}_{prefix_size}.pth'
remove_lora(model)
if os.path.exists(fname):
    prefix = torch.load(fname)
    print(f"Prefix loaded from cache.")
else:
    prefix = torch.randn((model.config.n_layer, prefix_size, model.config.n_embd), requires_grad=True, device=device)
    train_config = Trainer.get_default_config()
    train_config.batch_size = batch_size
    train_config.num_workers = 8
    train_config.max_iters = 50_000
    train_config.learning_rate = 5e-5
    trainer = PrefixTrainer(train_config, model, train_dataset, prefix)
    trainer.set_callback('on_batch_end', combo_callback)
    trainer.run()
    torch.save(prefix, fname)
print(f"Performance on the {dataset_name} dataset with prefix:")
_ = eval_classification(model, dataset=test_dataset, device=device, max_batches=1024, prefixes=prefix, show_wrong_examples=True)

# %%
# train with the permuted dataset 
fname = f'05_prefix_{model_name}_{dataset_name.replace("/","_")}_{prefix_size}_permuted.pth'
remove_lora(model)
if os.path.exists(fname):
    prefix_permuted = torch.load(fname)
    print(f"Prefix loaded from cache.")
else:
    prefix_permuted  = torch.randn((model.config.n_layer, prefix_size, model.config.n_embd), requires_grad=True, device=device)
    train_config = Trainer.get_default_config()
    train_config.num_workers = batch_size
    train_config.batch_size = 8
    train_config.max_iters = 50_000
    train_config.learning_rate = 5e-5
    trainer = PrefixTrainer(train_config, model, train_dataset_permuted, prefix_permuted)
    trainer.set_callback('on_batch_end', combo_callback)
    trainer.run()
    torch.save(prefix_permuted, fname)
print(f"Performance on the {dataset_name} dataset with prefix:")
_ = eval_classification(model, dataset=test_dataset_permuted, device=device, max_batches=1024, prefixes=prefix_permuted, show_wrong_examples=True)

# %%
# train LoRA with the permuted tokens
train_config = Trainer.get_default_config()
train_config.num_workers = 0
train_config.batch_size = batch_size
train_config.max_iters = 5_000
train_config.learning_rate = 5e-3
trainer = LoRATrainer(
    train_config, 
    model, 
    train_dataset_permuted, 
    rank=1,
    device=device,
)
trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run()
_ = eval_classification(model, test_dataset_permuted, device=device, max_batches=1000)

# %%
n_lora_params = sum(p.numel() for p in get_lora_state_dict(model).values())
print(f"Number of LoRA parameters: {n_lora_params}")
print(f"Equivalent to {n_lora_params/(model.config.n_layer*model.config.n_embd)}-long prefix")

# %%



