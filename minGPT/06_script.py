from collections import defaultdict
import torch
import os

from mingpt.utils import set_seed
from mingpt.trainer import Trainer, PrefixTrainer
from mingpt.model import GPT
from mingpt.data_tools import CustomDataset, eval, batch_end_callback, attention_visualization, label_batch
import mingpt.data_tools as tasks

import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='Random seed', default=1428)
parser.add_argument('--nlayer', type=int, help='Number of layers', default=4)
parser.add_argument('--nhead', type=int, help='Number of heads', default=4)
parser.add_argument('--prefixsize', type=int, help='Prefix size', default=10)
parser.add_argument('--pretraining_tasks', type=str, help='Which tasks to use for pretraining')
parser.add_argument('--prefixtuning_tasks', type=str, help='Which tasks to use for prefix-tuning')
parser.add_argument('--pretraining_iters', type=int, default=50_000)
parser.add_argument('--prefixtuning_iters', type=int, default=50_000)
parser.add_argument('--embed_size', type=int, default=512)

args = parser.parse_args()
seed = args.seed
set_seed(seed)

import wandb
run = wandb.init(project="my-project-name")


# DEFINE TASK LISTS

task_lists = {
    "original_pretrain": [
        tasks.SortAscending(),
        tasks.SortDescending(),
        tasks.Add1(),
        tasks.Add1() + tasks.Add1(),
    ],
    "original_eval": [
        tasks.SortAscending() + tasks.Add1(),
        tasks.DoubleHistogram(),
        tasks.Modulo(),
        tasks.FilterAtLeastNTimes()
    ],
    "extended_pretrain": [
        tasks.SortAscending(),
        tasks.SortDescending(),
        tasks.Add1(),
        tasks.Modulo(),
        tasks.LessThan(),
        tasks.Divisible(),
        tasks.InverseBinary(),
    ],
    "extended_eval": [
        tasks.Add1() + tasks.Add1(),
        tasks.Add1() + tasks.Add1() + tasks.Add1(),
        tasks.MoreThanEqual(),
        tasks.NotDivisible(),
        tasks.DoubleHistogram(),
        tasks.FilterAtLeastNTimes(),
        tasks.SortAscending()+tasks.Add1(),
        tasks.Add1() + tasks.LessThan(),
        tasks.LessThan() + tasks.Add1(),
        tasks.LessThan() + tasks.SortAscending(),
        tasks.Divisible() + tasks.Add1(),
    ]
}


prefix_size = args.prefixsize
pretrain_dataset = CustomDataset('train', prefix_padding=prefix_size, num_digits=10, tasks=task_lists[args.pretraining_tasks])

# create a GPT instance
model_config = GPT.get_default_config()
model_config.model_type = None
model_config.vocab_size = pretrain_dataset.get_vocab_size()
model_config.block_size = pretrain_dataset.get_block_size()
model_config.n_layer = args.nlayer
model_config.n_head = args.nhead
model_config.n_embd = args.embed_size
model_config.batch_size = 512
model = GPT(model_config)

model_name = f"06_results/pt{args.pretraining_tasks}_ft{args.prefixtuning_tasks}_ps{prefix_size}_nl{model_config.n_layer}_nh{model_config.n_head}_em{model_config.n_embd}_s{seed}"

if not os.path.exists(model_name):
    os.makedirs(model_name)

fname = f'{model_name}/pretrained.pth'

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
    train_config.max_iters = args.pretraining_iters
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, pretrain_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    device = trainer.device

    # save the model weights:
    torch.save(model.state_dict(), fname)


prefixes: Dict[str, torch.Tensor] = dict()
# prefixes_at_steps: Dict[str, List[Tuple[int, torch.Tensor]]] = defaultdict(list)

# checkpoint_times = [i-1 for i in range(1, args.prefixtuning_iters+1) if i%10_000 == 0]

tasks = task_lists[args.pretraining_tasks]+task_lists[args.prefixtuning_tasks]

def callback(trainer, fname: str, taskname: str):
    batch_end_callback(trainer, steps=5000)
    # if (trainer.iter_num+1) % 10_000 == 0:
    #     checkpoint_name = f"{fname}_{trainer.iter_num+1}.pth"
    #     torch.save(trainer.prefixes, checkpoint_name)
    #     prefixes_at_steps[task_name].append((trainer.iter_num+1, trainer.prefixes.clone().detach().cpu()))
    #     print(f"Prefix checkpoint saved to {checkpoint_name}.")

for task, iterations, lr in zip(
    tasks,
    [args.prefixtuning_iters]*len(tasks),
    [5e-5]*len(tasks)
):
    task_name = str(task)
    fname = f'{model_name}/prefix_{task_name}'
    train_dataset = CustomDataset('train', prefix_padding=prefix_size, num_digits=10, tasks=[task])
    if os.path.exists(fname+".pth"):
        prefixes[task_name] = torch.load(fname+".pth")
        print(f"Prefix {task} loaded from cache.")
    else:
        print(f"TRAINING A PREFIX FOR THE {task_name.upper()} TASK:")
        prefixes[task_name]  = torch.randn((model.config.n_layer,prefix_size, model.config.n_embd), requires_grad=True, device=device)
        train_config = Trainer.get_default_config()
        train_config.num_workers = 0
        train_config.max_iters = iterations
        train_config.learning_rate = lr
        trainer = PrefixTrainer(train_config, model, train_dataset, prefixes[task_name], checkpoints_name=fname)
        trainer.set_callback('on_batch_end', lambda x: callback(x, fname, task_name))
        trainer.run()
        torch.save(prefixes[task_name], fname+".pth")
    # _ = eval(model, dataset=test_dataset, device=device, max_batches=32, prefixes=prefixes[task_name])
    print()


# Print a matrix of accuracies on all tasks for all prefixes:
results = {
    "seed": args.seed,
    "nlayer": args.nlayer,
    "nhead": args.nhead,
    "embed_size": args.embed_size,
    "prefix_size": prefix_size,
    "pretraining_tasks": args.pretraining_tasks,
    "prefixtuning_tasks": args.prefixtuning_tasks,
    "pretraining_iters": args.pretraining_iters,
    "prefixtuning_iters": args.prefixtuning_iters,
}

for task in tasks:
    task_name = str(task)
    fname = f'{model_name}/prefix_{task_name}'
    test_dataset = CustomDataset('test', prefix_padding=prefix_size, num_digits=10, tasks=[task])

    print(f"Prefix {task_name}:")

    accs = []
    for f in sorted(os.listdir(os.path.dirname(fname)), reverse=False):
        if f.startswith(fname.split('/')[-1]+"_0"):
            latest_checkpoint = os.path.join(os.path.dirname(fname), f)

            checkpoint = torch.load(latest_checkpoint)
            try:
                iter_num = checkpoint['iter_num']+1
                prefixes = checkpoint['prefixes']
            except:
                continue

            print(f"- {iter_num:>5}iters: ", end="")
            accs.append((iter_num, eval(model, dataset=test_dataset, device=device, max_batches=32, prefixes=prefixes.to(device))))
        
            results[f"acc_{task_name}"] = accs[-1][1]
            results[f"accs_{task_name}"] = accs


yaml_file_path = f"{model_name}/accuracies.yaml"
# save as yaml
import yaml
with open(yaml_file_path, 'w') as outfile:
    yaml.dump(results, outfile, default_flow_style=False)


run.log(results)
