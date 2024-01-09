"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
from typing import Optional
import os

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

from functools import partial
from minlora import LoRAParametrization, add_lora, apply_to_lora, get_lora_params, get_lora_state_dict

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

# CHANGED: Added prefix-tuning trainer
class PrefixTrainer(Trainer):

    def __init__(
            self, 
            config, 
            model, 
            train_dataset, 
            prefixes: torch.Tensor, 
            checkpoints_name=None,
        ):
        super().__init__(config, model, train_dataset)
        self.prefixes = prefixes
        self.checkpoints_name = checkpoints_name
        self.iter_num = 0


    def save_checkpoint(self, filename):
        """Save a checkpoint for the model and optimizer."""
        torch.save({
            'prefixes': self.prefixes,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter_num': self.iter_num
            }, filename)
        print(f"Saved checkpoint {filename}")
                            
    def load_checkpoint(self, filename):
        """Load a checkpoint into the model and optimizer."""
        checkpoint = torch.load(filename)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter_num = checkpoint['iter_num']+1
        print(f"Loaded checkpoint {filename}")
        return checkpoint['prefixes']


    def run(self):
        model = self.model
        prefixes = self.prefixes
        config = self.config

        # setup the optimizer
        self.optimizer = torch.optim.AdamW(
            [prefixes], 
            lr=config.learning_rate, 
            betas=config.betas
        )
        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.eval() # because we are not training the model

        # Check for existing checkpoints and load if found
        latest_checkpoint = None
        for f in sorted(os.listdir(os.path.dirname(self.checkpoints_name)), reverse=True):
            if f.startswith(self.checkpoints_name.split('/')[-1]):
                latest_checkpoint = os.path.join(os.path.dirname(self.checkpoints_name), f)
                break
        if latest_checkpoint:
            loaded_prefixes = self.load_checkpoint(latest_checkpoint)     
            with torch.no_grad():
                prefixes.copy_(loaded_prefixes)

        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            


            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y, prefixes=prefixes)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            if prefixes.grad is not None:
                prefixes.grad.zero_()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(prefixes, config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            if (self.iter_num+1) % 10_000 == 0:  # Every 10,000 iterations, save the model
                self.save_checkpoint(f"{self.checkpoints_name}_{self.iter_num+1:09}.pth")
                
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break


# CHANGED: Added a LoRA trainer

def add_lora_by_name(model, target_module_names, lora_config):
    """Add LoRA parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_lora(layer, lora_config=lora_config)

class LoRATrainer(Trainer):

    def __init__(
            self, 
            config, 
            model, 
            train_dataset, 
            rank: int,
            device,
        ):
        super().__init__(config, model, train_dataset)
        self.rank = rank
        self.device = device

    def run(self):
        model = self.model
        config = self.config
        rank = self.rank


        lora_config = {  # specify which layers to add lora to
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=rank),
            },
        }

        with torch.device(self.device):
            with torch.set_grad_enabled(True):
                add_lora_by_name(model, ["c_proj", "c_fc"], lora_config=lora_config)

        params = [
            {"params": list(get_lora_params(model))},
        ]

        self.optimizer = torch.optim.AdamW(
            params, 
            lr=config.learning_rate, 
            betas=config.betas
        )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.eval() # because we are not training the model
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            model.apply(apply_to_lora(lambda x: torch.nn.utils.clip_grad_norm_(x.lora_A, config.grad_norm_clip)))
            model.apply(apply_to_lora(lambda x: torch.nn.utils.clip_grad_norm_(x.lora_B, config.grad_norm_clip)))
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
