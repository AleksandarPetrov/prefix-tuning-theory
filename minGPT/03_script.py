import torch
import os

from mingpt.utils import set_seed
from mingpt.trainer import Trainer, PrefixTrainer
from mingpt.model import GPT
from mingpt.data_tools import CustomDataset, eval, batch_end_callback, attention_visualization, label_batch

import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='Random seed', default=1428)
parser.add_argument('--nlayer', type=int, help='Number of layers', default=4)
parser.add_argument('--nhead', type=int, help='Number of heads', default=4)
parser.add_argument('--prefixsize', type=int, help='Prefix size', default=10)
args = parser.parse_args()
seed = args.seed
set_seed(seed)

import wandb
run = wandb.init(project="my-project-name")

prefix_size = args.prefixsize
train_dataset_random = CustomDataset('train', mode="random", prefix_padding=prefix_size)
train_dataset_ascending = CustomDataset('train', mode="ascending", prefix_padding=prefix_size)
train_dataset_descending = CustomDataset('train', mode="descending", prefix_padding=prefix_size)
train_dataset_add1 = CustomDataset('train', mode="add1", prefix_padding=prefix_size)
train_dataset_add2 = CustomDataset('train', mode="add2", prefix_padding=prefix_size)
train_dataset_ascending_add1 = CustomDataset('train', mode="ascending_add1", prefix_padding=prefix_size)
train_dataset_double_hist = CustomDataset('train', mode="double_hist", prefix_padding=prefix_size)

test_dataset_random = CustomDataset('test', mode="random", prefix_padding=prefix_size)
test_dataset_ascending = CustomDataset('test', mode="ascending", prefix_padding=prefix_size)
test_dataset_descending = CustomDataset('test', mode="descending", prefix_padding=prefix_size)
test_dataset_add1 = CustomDataset('test', mode="add1", prefix_padding=prefix_size)
test_dataset_add2 = CustomDataset('test', mode="add2", prefix_padding=prefix_size)
test_dataset_ascending_add1 = CustomDataset('test', mode="ascending_add1", prefix_padding=prefix_size)
test_dataset_double_hist = CustomDataset('test', mode="double_hist", prefix_padding=prefix_size)

# create a GPT instance
model_config = GPT.get_default_config()
model_config.model_type = None
model_config.vocab_size = train_dataset_random.get_vocab_size()
model_config.block_size = train_dataset_random.get_block_size()
model_config.n_layer = args.nlayer
model_config.n_head = args.nhead
model_config.n_embd = 256
model_config.batch_size = 512
model = GPT(model_config)

model_name = f"{seed}_{prefix_size}_{model_config.n_layer}_{model_config.n_head}_{model_config.n_embd}"
fname = f'03_{model_name}_pretrained.pth'
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
    train_config.max_iters = 40000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset_random)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    device = trainer.device

    # save the model weights:
    torch.save(model.state_dict(), fname)


prefixes = dict()

for task, iterations, lr in zip(
    ["ascending", "descending", "add1", "add2", "ascending_add1", "double_hist"],
    [20_000, 20_000, 20_000, 20_000, 500_000, 500_000],
    [5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5]
):
    fname = f'03_{model_name}_prefix_{task}.pth'
    if os.path.exists(fname):
        prefixes[task] = torch.load(fname)
        print(f"Prefix {task} loaded from cache.")
    else:
        print(f"TRAINING A PREFIX FOR THE {task.upper()} TASK:")
        prefixes[task]  = torch.randn((model.config.n_layer,prefix_size, model.config.n_embd), requires_grad=True, device=device)
        train_config = Trainer.get_default_config()
        train_config.num_workers = 0
        train_config.max_iters = iterations
        train_config.learning_rate = lr
        trainer = PrefixTrainer(train_config, model, locals()[f"train_dataset_{task}"], prefixes[task], f"{model_name}_{task}")
        trainer.set_callback('on_batch_end', batch_end_callback)
        trainer.run()
        torch.save(prefixes[task], fname)
    # _ = eval(model, dataset=locals()[f"test_dataset_{task}"], device=device, max_batches=32, prefixes=prefixes[task])
    print()

# Print a matrix of accuracies on all tasks for all prefixes:
results = {
    "seed": args.seed,
    "nlayer": args.nlayer,
    "nhead": args.nhead,
}
for prefix_task in ["pretrained", "ascending", "descending", "add1", "add2", "ascending_add1", "double_hist"]:
    print(f"Prefix {prefix_task}:")
    d = {}
    for test_task in ["ascending", "descending", "add1", "add2", "ascending_add1", "double_hist"]:
        print(f"- Task {test_task} ", end="")
        d[test_task] = eval(model, dataset=locals()[f"test_dataset_{test_task}"], device=device, max_batches=32, prefixes=prefixes[prefix_task] if prefix_task != "pretrained" else None)
    
    results[prefix_task] = d


yaml_file_path = model_name+"_accuracies.csv"
# save as yaml
import yaml
with open(yaml_file_path, 'w') as outfile:
    yaml.dump(results, outfile, default_flow_style=False)


run.log(results)
