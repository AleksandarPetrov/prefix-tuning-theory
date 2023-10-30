# When Do Prompting and Prefix-Tuning Work? 

This is the companion code for our[_When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations_](arxiv.org) paper.

You may want to download some of our trained prefixes needed to run some of the notebooks from https://drive.google.com/drive/folders/1Bff8VKh1ZdflaVFKDY9MiyCmewoAGnzV?usp=sharing. Make sure to place the checkpoints from the minGPT directory into the minGPT directory of this repository.

## Structure:

- `llama` contains a modified version of the [original LLaMA code](https://github.com/facebookresearch/llama/tree/main/llama) that adds an implementation for prefix-tuning. Changes have been clearly commented as such.

- `llama_token_vs_soft_token.ipynb` compares how many unique completions LLaMA has if we vary only the first token vs if we vary only the first virtual token (Section 3).

- `constructions.ipynb` contains the implementations of transformer architecutres whose unconditional and conditional generation is fully governed by the choice of virtual tokens (Section 3).

- `prefix_bias_only.ipynb` illustrates that the theory from Section 4 holds for LLaMA, namely that a prefix cannot change the relative attention distribution over the content positions and only induces a bias in the attention layer output.

- `minGPT` contains a modified version of the [original minGPT code](https://github.com/karpathy/minGPT/) with an implementation of prefix-tuning. The directory also contains the experiments of Section 5 from the paper:

    - `01_cannot_learn_new_task.ipynb` shows that prefix-tuning cannot learn a new task that requires a different attention pattern.

    - `02_can_extract_pretrained_task.ipynb` shows that prefix-tuning can be used to specialize the model for one of the tasks it has seen during pre-training.

    - `03_can_learn_new_task_same_attention.ipynb` shows that prefix-tuning can also learn a new task, as long as the attention patterns necessary to solve it have been learned during pretraining, but cannot learn a new task (double histogram) that cannot be solved with skills learned during pretraining.

    - `04_prefix_tuning_vs_lora.ipynb` shows that rank-1 LoRA on the MLP is sufficient to learn double histogram but prefix-tuning with the same number of learnable parameters cannot (Section 6).

- `longer_prefixes.ipynb` shows that the attention distribution over the prefix positions is not unifromly distributed, showing that prefix-tuning does not make full use of the subspace spanned by the prefix-induced biases. (Appendix B)


## Reference

@article{petrov2023when,
  title={When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations},
  author={},
  journal={arXiv preprint arXiv:},
  year={2023}
}

