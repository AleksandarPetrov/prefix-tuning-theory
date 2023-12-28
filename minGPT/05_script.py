# %%
import torch
import os
import pandas as pd


from mingpt.utils import set_seed
from mingpt.trainer import Trainer, PrefixTrainer, LoRATrainer
from mingpt.model import GPT
from mingpt.data_tools import CustomDataset, eval, batch_end_callback, attention_visualization, label_batch, eval_memory

import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Literal

from minlora import get_lora_state_dict, remove_lora

seed = 3333
print(f"Seed: {seed}")
set_seed(seed)

# %%
import unicodedata
class WordsDataset:
    def __init__(
            self, 
            file: str, 
            src_lang: Literal["AmE", "BrE", "DE", "EP", "ES"], 
            tgt_lang: Literal["AmE", "BrE", "DE", "EP", "ES"],
            prefix_padding: int = 0,
        ):

        self.src_column = f"{src_lang}_wordform"
        self.tgt_column = f"{tgt_lang}_wordform"
        
        self.prefix_padding = prefix_padding

        self.df = pd.read_csv(file)
        # remove empty rows
        self.df = self.df[self.df[self.src_column].notna()]
        # if multiple same entries exist for the src_column, keep only the first
        self.df = self.df.drop_duplicates(subset=[self.src_column], keep="first")

        wordform_columns = [col for col in self.df.columns if col.endswith("_wordform")]
        normalize = lambda str: unicodedata.normalize('NFD', str).encode('ASCII', 'ignore').decode()
        # aplly normalize to each column in wordform_columns
        for col in wordform_columns:
            self.df[col] = self.df[col].apply(normalize)

        # get the unique characters
        self.chars = sorted(list(set( "".join(["".join(self.df[col].tolist()) for col in wordform_columns ]) )))
        self.chars.append("<TR>")
        self.chars.append("<PAD>")
        self.tokenizer_map = {c: i for i, c in enumerate(self.chars)}
        self.detokenizer_map = {i: c for i, c in enumerate(self.chars)}
        self.detokenizer_map.update({-1: "<M>"})

        # longest word
        self.longest = max([max([len(w) for w in self.df[self.src_column]]) for col in wordform_columns])
    
    def detokenize(self, toks: List[int]):
        if isinstance(toks, torch.Tensor):
            toks = toks.tolist()
        return "".join([self.detokenizer_map[x] for x in toks])
    
    def tokenize(self, str: str):
        return [self.tokenizer_map[s] for s in str]
    
    def __len__(self):
        return len(self.df)

    def get_vocab_size(self):
        return len(self.chars)

    def get_block_size(self):
        return 2+2*self.longest+self.prefix_padding
    
    def __getitem__(self, idx):
        src = self.tokenize(self.df.iloc[idx][self.src_column])
        tgt = self.tokenize(self.df.iloc[idx][self.tgt_column])
        content_length = len(src)+len(tgt)+self.prefix_padding+1
        cat = torch.tensor(
            [0 for _ in range(self.prefix_padding)] + src + [self.tokenizer_map["<TR>"]] + tgt + [self.tokenizer_map["<PAD>"] for _ in range(self.get_block_size()-content_length)],
            dtype=torch.long
        )

        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input and prefix locations
        y[:len(src)+1+self.prefix_padding-1] = -1
        y[len(src)+1+self.prefix_padding-1+len(tgt)+1:] = -1
        return x, y

# %%
ds = WordsDataset("PHOR_in_One_LDB.csv", src_lang="BrE", tgt_lang="DE")
idx = 165
for x, y in zip(ds[idx][0], ds[idx][1]):
    print(f"{ds.detokenize([x.item()])} \t-- {ds.detokenize([y.item()])}")

# %%
prefix_size = 66
dataset_en_de = WordsDataset("PHOR_in_One_LDB.csv", src_lang="BrE", tgt_lang="DE", prefix_padding=prefix_size)
dataset_en_es = WordsDataset("PHOR_in_One_LDB.csv", src_lang="BrE", tgt_lang="ES", prefix_padding=prefix_size)

# %% [markdown]
# Let's load the pretrained model (if you don't have it, run notebook 03 first).

# %%
# create a GPT instance
model_config = GPT.get_default_config()
model_config.model_type = None
model_config.vocab_size = dataset_en_de.get_vocab_size()
model_config.block_size = dataset_en_de.get_block_size()
model_config.n_layer = 4
model_config.n_head = 4
model_config.n_embd = 256
model_config.batch_size = 512
model = GPT(model_config)

fname = f'05_{seed}_pretrained.pth'
remove_lora(model)
if os.path.exists(fname):
    print("Loading weights from cache, won't train from scratch.")
    model.load_state_dict(torch.load(fname))
    model.config = model_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
else:
    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 50000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, dataset_en_de)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    device = trainer.device

    # save the model weights:
    torch.save(model.state_dict(), fname)

# %% [markdown]
# Check that the pretrained model has zero accuracy at double histogram.

# %%
print("Pretrained performance on the EN-DE dataset:")
_ = eval_memory(model, dataset=dataset_en_de, device=device, show_wrong_examples=True)

# %% [markdown]
# A prefix for this task still has close to 0% accuracy.

# %%
fname = f'05_{seed}_prefix_spanish.pth'
remove_lora(model)
if os.path.exists(fname):
    prefix = torch.load(fname)
    print(f"Prefix double_histloaded from cache.")
else:
    prefix  = torch.randn((model.config.n_layer,prefix_size, model.config.n_embd), requires_grad=True, device=device)
    train_config = Trainer.get_default_config()
    train_config.num_workers = 0
    train_config.max_iters = 100_000
    train_config.learning_rate = 5e-5
    trainer = PrefixTrainer(train_config, model, dataset_en_es, prefix)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    torch.save(prefix, fname)
print("Performance on the EN-ES dataset with prefix:")
_ = eval_memory(model, dataset=dataset_en_es, device=device, show_wrong_examples=True)

# %% [markdown]
# However, rank 1 LoRA update of the MLP weights for just a tenth of the training iterations results in high accuracy:

# %%
train_config = Trainer.get_default_config()
train_config.num_workers = 0
train_config.max_iters = 50_000
train_config.learning_rate = 1e-3
trainer = LoRATrainer(
    train_config, 
    model, 
    dataset_en_es,
    rank=4, #16 works
    device=device,
    where=["c_attn", "c_proj", "c_fc", "lm_head"],
)
trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run()
_ = eval_memory(model, dataset=dataset_en_es, device=device, show_correct=False)

# %% [markdown]
# And that is despite the two fine-tuning approaches having the same number of learnable parameters. The limitated performance of prefix-tuning is not simply because of it using few parameters as LoRA with the same number of parameters solves the task. Therefore, prefix-tuning (and prompting) suffer unique structural limitations.

# %%
lora_params = sum(p.numel() for p in get_lora_state_dict(model).values())

print(f"Number of prefix parameters: {torch.numel(prefix)} ")
print(f"Number of LoRA parameters: {lora_params} (equivalent to {lora_params/(model.config.n_layer*model.config.n_embd):.2f} prefixes)")

# %%



