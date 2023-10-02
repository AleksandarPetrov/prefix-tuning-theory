from typing import List, Literal, Optional, Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pickle

class CustomDataset(Dataset):
    """ 
    Modified class for the sort and add problem.
    Can sort in ascending and descending order, add one or two to each element, sort and add, or do a random action.
    """

    def __init__(
            self, 
            split: Literal["train", "test"], 
            length=10, 
            num_digits=8, 
            mode: Literal["ascending", "descending", "add1", "add2", "random", "ascending_add1"] = "random", 
            prefix_padding: int = 0
        ):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
        self.mode = mode
        self.prefix_padding = prefix_padding
        self.random_mode_list = ["ascending", "descending", "add1", "add2"]
    
    def __len__(self):
        return 10000
    
    def get_vocab_size(self):
        return self.num_digits+2 # handling the add2 case
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1 + self.prefix_padding

    def __getitem__(self, idx):
        
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that 
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unqiue digits, re-sample
                    continue

            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok

        if self.mode == "random":
            curr_mode = self.random_mode_list[idx%len(self.random_mode_list)]
        else:
            curr_mode = self.mode

        # solve the task: 
        if curr_mode == "ascending":
            sol = torch.sort(inp, descending=False)[0] 
        elif curr_mode == "descending":
            sol = torch.sort(inp, descending=True)[0]
        elif curr_mode == "add1":
            sol = inp+1
        elif curr_mode == "add2":
            sol = inp+2
        elif curr_mode == "ascending_add1":
            sol = torch.sort(inp, descending=False)[0] + 1
        else:
            raise ValueError(f"Mode {curr_mode} is not recognized!")

        # create the prefix padding
        pad = torch.zeros(self.prefix_padding,  dtype=torch.long)

        # concatenate the problem specification and the solution
        cat = torch.cat((pad, inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input and prefix locations
        y[:self.length+self.prefix_padding-1] = -1
        return x, y

def batch_end_callback(trainer):
    if (trainer.iter_num+1) % 1000 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:>6.2f}ms; iter {trainer.iter_num+1:>6}: train loss {trainer.loss.item():.5f}")

def eval(
        model, 
        dataset: CustomDataset, 
        device, 
        max_batches: int,
        provide_first: bool = False, 
        prefixes: Optional[torch.Tensor] = None,
        attention_lora: Optional[torch.Tensor] = None,
        projection: bool = False,
        show_wrong_examples: bool = False,
    ):
    """ Modified from the original minGPT code."""

    n = dataset.length # naugy direct access shrug
    prefix_size = dataset.prefix_padding
    results = []
    mistakes_printed_already = 0 if show_wrong_examples else 1000
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    model.eval();
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            # isolate the input pattern alone
            inp = x[:, :prefix_size+n+int(provide_first)]
            sol = y[:, -n+int(provide_first):]
            # let the model sample the rest of the sequence
            cat = model.generate(
                inp, 
                n-int(provide_first), 
                do_sample=False, 
                prefixes=prefixes,
                attention_lora = attention_lora,
                projection = projection,
            ) # using greedy argmax, not sampling
            sol_candidate = cat[:, prefix_size+n+int(provide_first):] # isolate the filled in sequence
            # compare the predicted sequence to the true sequence
            correct = (sol == sol_candidate).all(1).cpu() 
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 3: # only print up to 3 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that %s processed is %s but gt is %s" % (inp[i].tolist()[prefix_size:prefix_size+n], sol_candidate[i].tolist(), sol[i].tolist()))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("Final score: %d/%d = %.2f%% correct" % (rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()


def attention_visualization(
        test_sample: List[int], 
        activations_record: List[Tuple[str, torch.Tensor]], 
        prefix_size: int,
        length: int, 
        title: str="", 
        ignore_and_rescale_prefix: bool = False,
        annot: bool = True,
    ):

    attentions = torch.cat([x[1] for x in activations_record if x[0] == "attention"], dim=0)
    n_layers, n_heads = attentions.size(0), attentions.size(1)

    # sns.set(font_scale=1.25)
    mask = torch.triu(torch.ones_like(attentions[0,0], dtype=bool), diagonal=1)

    if ignore_and_rescale_prefix:
        mask[:prefix_size,:] = 1
        mask[:,:prefix_size] = 1
        
        content_att = attentions[:,:, prefix_size:, prefix_size:].sum(axis=3)
        attentions[:, :, prefix_size:, prefix_size:] /= content_att[:, :, :, None]

    fig, axes = plt.subplots(n_layers,n_heads, figsize=(5*n_heads, 5*n_layers))
    # if only one plot, make it a list, otherwise use flat
    if n_layers == 1 and n_heads == 1:
        axes = [axes]
    else:
        axes = axes.flat
    for i, ax in enumerate(axes):
        layer_id, head_id = i//n_heads, i - (i//n_heads)*n_heads
        # center = attentions[layer_id][head_id].cpu()[~mask].quantile(0.95)
        center = 0.5
        sns.heatmap(attentions[layer_id][head_id].cpu(), ax=ax, cbar=False, cmap="rocket_r", xticklabels=1, yticklabels=1, mask=mask.detach().cpu().numpy(), center=center, annot=annot, annot_kws={'size': 5}, fmt='.2f', square=True)
        ax.set_title(f"Layer {layer_id+1}, Head {head_id+1}",pad=-50,y=0.9)
        # use the cat vector to set the tick labels:
        ax.set_xticklabels([""]*prefix_size + [int(t) for t in test_sample.tolist()[prefix_size:-1]], rotation=0)
        ax.set_yticklabels([""]*prefix_size + [int(t) for t in test_sample.tolist()[prefix_size:-1]], rotation=0)
        ax.axhline(y=prefix_size, color='white', linewidth=2, alpha=0.5, ls="-")
        ax.axhline(y=prefix_size+length, color='white', linewidth=2, alpha=0.5, ls="-")
        ax.axvline(x=prefix_size, color='white', linewidth=2, alpha=0.5, ls="-")
        ax.axvline(x=prefix_size+length, color='white', linewidth=2, alpha=0.5, ls="-")

        group_ax = ax.secondary_xaxis(-0.05)
        group_ax.set_xticks([prefix_size/2, prefix_size + length/2, prefix_size + length + length/2])
        group_ax.set_xticklabels(["prefix", "input", "output"], rotation=0)
        group_ax.spines["bottom"].set_visible(False)
        group_ax.tick_params(axis='x', which='both', length=0)
    plt.tight_layout()
    plt.suptitle(title, y=1.02)

def label_batch(x, y, prefix_size, length, as_ints=False):
    labels = []
    for input, p in zip(x[:,prefix_size:prefix_size+length], y[:,prefix_size+length-1:]):
        if (torch.sort(input)[0] == p).all():
            labels.append(1 if as_ints else "Ascending")
        elif (torch.sort(input, descending=True)[0] == p).all():
            labels.append(2 if as_ints else "Descending")
        elif (input + 1 == p).all():
            labels.append(3 if as_ints else "Add 1")
        elif (input + 2 == p).all():
            labels.append(4 if as_ints else "Add 2")
        else:
            labels.append(0 if as_ints else "None")
    return labels