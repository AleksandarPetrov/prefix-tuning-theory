import dataclasses
import random
import socket
from typing import Callable, List, Tuple

import torch


def find_free_port() -> int:
    """Find a free socket to use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a port that is free
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  # Return the port number


def trim_list(l: List, size: int) -> List:
    """Ensure that a list has at most size elements."""
    if len(l) <= size:
        return l
    else:
        return l[:size]


@dataclasses.dataclass
class TrainingCollation:
    x_str: List[str]
    y_str: List[str]
    all_y_options: List[List[str]]
    x_tok: List[int]
    y_tok: List[int]
    all_tokens: torch.Tensor
    loss_map: torch.Tensor
    x_introduction: List[int]
    y_introduction: List[int]


class CollateForTraining:
    def __init__(
        self,
        tokenizer_encode_fn: Callable,
        tokenizer_decode_fn: Callable,
        x_introduction: str,
        y_introduction: str,
        eos_tokens: List[int],
        max_seq_len: int = 128,
        padding_id: int = 0,
        add_spaces: bool = False,
    ):
        self.tokenizer_encode_fn = tokenizer_encode_fn
        self.tokenizer_decode_fn = tokenizer_decode_fn
        self.max_seq_len = max_seq_len
        self.padding_id = padding_id
        self.eos_tokens = eos_tokens

        if add_spaces:
            x_introduction = x_introduction+" "
            y_introduction = " "+y_introduction+" "
        self.x_introduction: List[int] = (
            tokenizer_encode_fn(x_introduction) if len(x_introduction) > 0 else []
        )
        self.y_introduction: List[int] = (
            tokenizer_encode_fn(y_introduction) if len(x_introduction) > 0 else []
        )

    def __call__(self, batch: List[Tuple[str, List[str]]]) -> TrainingCollation:
        """
        Prepare a list of X strings and Y strings for training.

        1. Select a random one of the set of possible Ys (can be multiple for table-to-text)
        2. Tokenize X and Y (Y gets the EOS token appended)
        3. Trim X such that x_introduction+trimmed(X)+y_introduction+Y fit in the context
        4. Concatenate x_introduction+trimmed(X)+y_introduction+Y
        5. Decode the trimmed strings for printing and reporting
        6. Pad the tokens to be equal length and convert to a single tensor
        7. Create a loss map that is true only over the Y positions.
        8. Return everything
        """

        xt = [self.tokenizer_encode_fn(t[0]) for t in batch]
        yt = [
            self.tokenizer_encode_fn(t[1][random.randint(0, len(t[1]) - 1)]) + self.eos_tokens
            for t in batch
        ]

        lens = [
            len(self.x_introduction) + len(x) + len(self.y_introduction) + len(y)
            for x, y in zip(xt, yt)
        ]
        xt = [
            x[: max(0, min(len(x), len(x) - l + self.max_seq_len))]
            for x, l in zip(xt, lens)
        ]

        all_tokens = [
            self.x_introduction + x + self.y_introduction + y for x, y in zip(xt, yt)
        ]
        # final trimming, just to catch some freaky cases when Y might be longer than the context
        all_tokens = [trim_list(t, self.max_seq_len) for t in all_tokens]
        max_len = max([len(t) for t in all_tokens])
        all_tokens = [t + [self.padding_id] * (max_len - len(t)) for t in all_tokens]
        all_tokens = torch.tensor(all_tokens, device="cpu")

        # strings
        xs = [self.tokenizer_decode_fn(x) for x in xt]
        ys = [self.tokenizer_decode_fn(y) for y in yt]

        # TODO: RETURN ALSO THE LIST OF ALL YS SO THAT WE CAN EVALUETE FOR THE GENERATION CASE

        # make the loss map
        loss_map = torch.tensor(
            [
                trim_list(
                    [False] * len(self.x_introduction)
                    + [False] * len(x)
                    + [False] * len(self.y_introduction)
                    + [True] * len(y)
                    + [False]
                    * (
                        max_len
                        - len(x)
                        - len(y)
                        - len(self.x_introduction)
                        - len(self.y_introduction)
                ), self.max_seq_len)
                for x, y in zip(xt, yt)
            ],
            device="cpu",
        )

        return TrainingCollation(
            x_str=xs,
            y_str=ys,
            x_tok=xt,
            y_tok=yt,
            all_tokens=all_tokens,
            loss_map=loss_map,
            x_introduction=self.x_introduction,
            y_introduction=self.y_introduction,
            all_y_options=[t[1] for t in batch],
        )
    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pretty_print_examples(
        test_prompts: List[str],
        test_responses: List[str],
        test_targets: List[List[str]],
        n_to_print: int = 5,
):
    for i in range(n_to_print):
        if len(test_targets[i]) == 1:
            print(f"{i+1}. {bcolors.UNDERLINE}{test_prompts[i]}{bcolors.END}")
            print(f'   {bcolors.BOLD}> RESPONSE:{bcolors.END} "{test_responses[i]}"')
            print(f'   {bcolors.BOLD}> EXPECTED:{bcolors.END} "{test_targets[i][0]}"')
        else:
            print(f"{i+1}. {bcolors.UNDERLINE}{test_prompts[i]}{bcolors.END}")
            print(f'   {bcolors.BOLD}> RESPONSE:{bcolors.END}   "{test_responses[i]}"')
            print(f'   {bcolors.BOLD}> EXPECTED:{bcolors.END} - "{test_targets[i][0]}"')    
            for j in range(1,len(test_targets[i])):
                print(f'               - "{test_targets[i][j]}"')    
