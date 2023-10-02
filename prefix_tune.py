#!/usr/bin/env python3

import datetime
from functools import reduce
import operator
import os
import random
import sys

from typing import List, Optional, Sequence, Tuple, Type, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import tqdm

import ds as data
from utils import (
    find_free_port,
    CollateForTraining,
    TrainingCollation,
    pretty_print_examples,
)


def prefix_tune(
    model_name: Literal["llama", "llama2", "gpt2"],
    dataset: Literal["Emotion", "DART"],
    prefix_mode: Literal["Classic"] = "Classic",
    prefix_size: int = 5,
    max_seq_len: int = 128,
    max_batch_size: int = 8,
    lr: float = 1e-2,
    lora_lr: float = 1e-4,
    lr_decay: float = 0.975,
    lr_decay_freq: int = 1000,
    epochs: int = 10,
    seed: int = 2023,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    short_test: Optional[int] = None,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    print(f"Seed is set to {seed}")

    ## LOAD MODEL
    if model_name.startswith("llama") or model_name.startswith("llama2"):
        size = model_name.split("-")[1]

        # add current directory to path
        sys.path.append(os.path.join(os.getcwd(), "llama"))
        from llama import Llama  # type: ignore

        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NUM_TRAINERS"] = "1"

        if os.environ.get("LLAMA_PATH") is None:
            raise RuntimeError("You need to set the LLAMA_PATH environmental variable!")

        model_path = os.environ["LLAMA_PATH"]
        print(f"LLAMA PATH={model_path} found, looking for model of size {size} in it.")

        generator = Llama.build(
            ckpt_dir=f"{model_path}/{size}",
            tokenizer_path=f"{model_path}/tokenizer.model",
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=1,
        )
        model = generator.model
        n_layers = len(model.layers)
        n_heads = model.params.n_heads
        embedding_size = model.params.dim
        tokenizer = generator.tokenizer
        tokenizer_encoder = lambda x: tokenizer.encode(x, bos=False, eos=False)
        tokenizer_decoder = lambda x: tokenizer.decode(x)
        eos_tokens = [tokenizer.eos_id]
        add_spaces = False

        # prevent the model from accumulating gradients
        for n, p in generator.model.named_parameters():
            p.requires_grad = False
    elif model_name == "gpt2":
        # add mingpt to the path relative to this file's location
        print(os.path.join(os.path.dirname(__file__), "minGPT"))
        sys.path.append(os.path.join(os.path.dirname(__file__), "minGPT"))
        from mingpt.model import GPT
        from mingpt.bpe import BPETokenizer
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        model = GPT.from_pretrained(model_name)
        n_layers = 12
        n_heads = 12
        embedding_size = 768
        model_path = "HuggingFace"
        add_spaces = True

        model.cuda()
        model.eval()

        tokenizer = BPETokenizer()
        tokenizer_encoder = lambda x: tokenizer.encoder.encode(x)
        tokenizer_decoder = lambda x: tokenizer.encoder.decode(x)
        eos_tokens = tokenizer_encoder('<|endoftext|>')
        print("eos_tokens", eos_tokens)

    else:
        raise NotImplementedError(f"Model {model_name} not supported.")

    ## PREPARE DATASETS
    try:
        ds_class: Union[
            Type[data.HuggingFaceClassificationDataset],
            Type[data.HuggingFaceTableToTextDataset],
        ] = getattr(data, dataset)
    except AttributeError:
        raise NotImplementedError(f"Dataset {dataset} not found!")

    ds_train, ds_test = ds_class(split="train"), ds_class(split="test")
    x_introduction, y_introduction, eval_method = (
        ds_train.x_introduction,
        ds_train.y_introduction,
        ds_train.evaluate,
    )
    if train_subset is not None and len(ds_train) > train_subset:
        ds_train = Subset(ds_train, random.sample(range(len(ds_train)), train_subset))
    if test_subset is not None and len(ds_test) > test_subset:
        ds_test = Subset(ds_test, random.sample(range(len(ds_test)), test_subset))
    print(
        f"Dataset loaded: {len(ds_train)} training samples and {len(ds_test)} test samples."
    )

    collate_fn = CollateForTraining(
        tokenizer_encoder,
        tokenizer_decoder,
        x_introduction=x_introduction,
        y_introduction=y_introduction,
        max_seq_len=max_seq_len - prefix_size,
        eos_tokens=eos_tokens,
        add_spaces=add_spaces
    )

    g_train, g_test = torch.Generator(device="cuda"), torch.Generator(device="cuda")
    g_train.manual_seed(seed)
    g_test.manual_seed(seed)

    dl_train = DataLoader(
        ds_train,
        batch_size=max_batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_fn,
        generator=g_train,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=max_batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_fn,
        generator=g_test,
    )

    ## SETUP PREFIX
    prefix = None
    prefix_key = None 
    prefix_value = None 
    prefix_attention = None
    attention_lora = None
    lora_rank = None
    
    if prefix_mode == "Classic":
        # classical prefix tuning where each layer gets prefix_size positions learned
        prefix = torch.normal(
            0,
            0.02,
            size=(n_layers, prefix_size, embedding_size),
            requires_grad=True,
            device="cuda",
        )
        n_params = reduce(operator.mul, prefix.shape, 1)
    else:
        raise NotImplementedError(f"Prefix mode {prefix_mode} not supported.")

    # creating a representation of the prefix for printing and as a placeholder to the model
    # these do not affect the computation in any way
    prefix_str = " ".join(["P"] * prefix_size)
    prefix_tokens = torch.tensor(tokenizer_encoder(prefix_str))
    assert prefix_size == len(prefix_tokens)

    ## SETUP THE OPTIMIZER
    optimizer = torch.optim.Adam([prefix], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    ## OPTIMIZE THE PREFIX
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    iters_since_last_lr_decay = 0
    acc_score = bleu_score = max_acc_score = max_bleu_score = -1

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        # need that for the LoRA dropout
        model.train()

        loss_since_last = 0

        pbar = tqdm.tqdm(
            enumerate(dl_train),
            desc=f"Training epoch {epoch+1}/{epochs}",
            total=len(dl_train),
        )
        for b_idx, b in pbar:
            b: TrainingCollation
            bsz = b.all_tokens.size(0)

            all_tokens, loss_map = b.all_tokens.cuda(), b.loss_map.cuda()
            all_tokens = torch.hstack((prefix_tokens.repeat(bsz, 1), all_tokens))
            loss_map = torch.hstack(
                (torch.tensor(0).repeat(bsz, prefix_size), loss_map)
            )

            if model_name.startswith("llama"):
                model.clear_cache_grad()

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if model_name.startswith("llama"):
                    logits = model.forward(
                        all_tokens,
                        start_pos=0,
                        prefix=prefix,
                    )
                elif model_name == "gpt2":
                    logits, _ = model.forward(
                        all_tokens,
                        prefixes=prefix,
                    )
                else:
                    raise RuntimeError("Unknown model name!")

                loss = torch.nn.functional.cross_entropy(
                    logits.transpose(-1, -2)[:, :, :-1],
                    all_tokens[:, 1:],
                    reduction="none",
                )
                scalar_loss = (loss * loss_map[:, 1:]).mean()
                loss_since_last += scalar_loss.item()

            # scale loss
            scaler.scale(scalar_loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # adjust the learning rate if it's time for that
            iters_since_last_lr_decay += 1
            if iters_since_last_lr_decay >= lr_decay_freq:
                iters_since_last_lr_decay = 0
                scheduler.step()

            # Do the optimization
            torch.nn.utils.clip_grad_norm_(
                parameters=[prefix], max_norm=1.0
            )  #  type: ignore
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            # If any parameter contains a nan or inf value stop the training:
            if torch.any(torch.isnan(torch.cat([p.flatten() for p in [prefix]]))) :
                raise RuntimeError("NaN in parameters!")
            if torch.any(torch.isinf(torch.cat([p.flatten() for p in [prefix]]))):
                raise RuntimeError("Inf in parameters!")

            log_step = b_idx * bsz + epoch * len(ds_train)
            if (b_idx + 1) % 50 == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss_since_last/50:.4e}",
                    }
                )
                loss_since_last = 0

        model.eval()

        # TEST

        # if not the last epoch and short_test is set, do only short_test batches
        n_test_batches = (
            short_test
            if (short_test is not None and epoch < epochs - 1) 
            else len(dl_test)
        )

        test_prompts = []
        test_responses = []
        test_targets = []

        pbar = tqdm.tqdm(
            enumerate(dl_test),
            desc=f"Testing epoch {epoch+1}/{epochs}",
            total=n_test_batches,
        )
        for b_idx, b in pbar:
            b: TrainingCollation
            bsz = b.all_tokens.size(0)

            # stop early if short_test has been setup and this is not the last epoch
            if b_idx+1 > n_test_batches:
                break
            
            inputs = [
                f"{prefix_str} {x_introduction} {x} {y_introduction}" for x in b.x_str
            ]

            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                if model_name.startswith("llama"):
                    results = generator.text_completion(
                        inputs,
                        max_gen_len=max_seq_len,
                        temperature=0,
                        top_p=0.9,
                        add_bos=False,
                        prefix=prefix,
                    )
                    results = [r["generation"] for r in results]
                elif model_name == "gpt2":
                    results = []
                    for input in inputs:
                        result = model.generate(
                            torch.tensor(tokenizer_encoder(input), dtype=torch.long).unsqueeze(0),
                            max_new_tokens=max_seq_len,
                            do_sample=False,
                            prefixes=prefix,
                        )
                        result = tokenizer_decoder(result[0].tolist())
                        result = result[len(input):]
                        # if the "<|endoftext|>" is in result, cut to just before it
                        if "<|endoftext|>" in result:
                            result = result[:result.index("<|endoftext|>")]
                        results.append(result)
                else:
                    raise RuntimeError("Unknown model name!")

            test_prompts.extend(inputs)
            test_responses.extend(results)
            test_targets.extend(b.all_y_options)

        metric_values = {}
        if len(test_prompts) >= 5:
            # evaluate metrics
            metric_values = eval_method(y_pred=test_responses, y_true=test_targets)
            acc_score = metric_values.get("accuracy", -1)
            bleu_score = metric_values.get("bleu_bleu", -1)
            max_acc_score, max_bleu_score = max(acc_score, max_acc_score), max(
                bleu_score, max_bleu_score
            )
            if acc_score >= 0:
                print(f"Accuracy: {acc_score*100:.2f}%")
            if bleu_score >= 0:
                print(f"BLEU score: {bleu_score:.4f}")

            pretty_print_examples(
                test_prompts=test_prompts,
                test_responses=test_responses,
                test_targets=test_targets,
            )

        # save prefix
        with torch.no_grad():
            dirname = f"trained_prefixes"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            filename = f"{dirname}/{model_name}_{dataset}_{prefix_mode}_{prefix_size}_{lora_rank}_{timestamp}_{epoch}.pt"

            to_store = {}
            to_store["prefix"] = prefix
            torch.save(to_store, filename)


if __name__ == "__main__":
    fire.Fire(prefix_tune)
