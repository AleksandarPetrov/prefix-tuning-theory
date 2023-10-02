import itertools
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from datasets import load_dataset, get_dataset_split_names
from datasets import Dataset as DS
import evaluate
from torch.utils.data import Dataset
import numpy as np

class HuggingFaceClassificationDataset(Dataset):
    x_introduction: str
    y_introduction: str
    task_name: str
    dataset_identifier: str

    def __init__(self, split: str, **kwargs):
        self.ds: Dataset = load_dataset(self.dataset_identifier, split=split, **kwargs)  # type: ignore
        assert isinstance(self.ds, DS)

    @property
    def splits(self):
        return get_dataset_split_names(self.dataset_identifier)

    def __len__(self):
        return self.ds.num_rows  # type: ignore

    def evaluate(self, y_pred: List[str], y_true: List[List[str]]) -> Dict:
        # y_true = [y[0].replace(self.y_introduction, "").strip() for y in y_true]

        true: List[str] = [y[0].strip() for y in y_true]
        pred: List[str] = [y.strip() for y in y_pred]

        return {'accuracy': np.mean([t == p for t, p in zip(true, pred)])}


class Emotion(HuggingFaceClassificationDataset):
    dataset_identifier = "dair-ai/emotion"
    x_introduction = "TEXT:"
    y_introduction = "EMOTION:"
    task_name = "Emotion"

    def __init__(self, split: str):
        super().__init__(split=split, name="split")
        self.labels = [s.replace("_", " ") for s in self.ds.features["label"].names] # type: ignore

    def __getitem__(self, idx: int) -> Tuple[str, List[str]]:
        item = self.ds[idx]
        x, label = item["text"], item["label"]
        y = self.labels[label]
        return x, [y]

class HuggingFaceTableToTextDataset(Dataset):
    x_introduction: str
    y_introduction: str
    task_name: str
    dataset_identifier: str

    def __init__(self, split: str, **kwargs):
        self.ds: Dataset = load_dataset(self.dataset_identifier, split=split, **kwargs)  # type: ignore
        assert isinstance(self.ds, DS)

        self.metrics = {
            "bleu": evaluate.load("bleu"),
            "ter": evaluate.load("ter"),
            "meteor": evaluate.load("meteor"),
            "bertscore": evaluate.load("bertscore"),
            "rouge": evaluate.load("rouge"),
        }

    @property
    def splits(self):
        return get_dataset_split_names(self.dataset_identifier)

    def __len__(self):
        return self.ds.num_rows  # type: ignore

    def evaluate(self, y_pred: List[str], y_true: List[List[str]]) -> Dict:
        results = dict()
        results.update({f"bleu_{k}": v for k, v in self.metrics["bleu"].compute(predictions=y_pred, references=y_true).items()})  # type: ignore
        results.update({f"ter_{k}": v for k, v in self.metrics["ter"].compute(predictions=y_pred, references=self.equalize_number_of_targets(y_true), case_sensitive=True).items()})  # type: ignore
        results.update({f"bertscore_{k}": v for k, v in self.metrics["bertscore"].compute(predictions=y_pred, references=y_true, lang="en").items()})  # type: ignore
        results.update(self.metrics["meteor"].compute(predictions=y_pred, references=y_true))  # type: ignore
        results.update(self.metrics["rouge"].compute(predictions=y_pred, references=y_true))  # type: ignore

        return results

    @classmethod
    def equalize_number_of_targets(cls, targets: List[List[str]]):
        """ Some metrics require that all samples have the same number of targets. This is a hack to ensure this. """
        lens = [len(t) for t in targets]
        median_length = int(np.median(lens))
        return [list(itertools.islice(itertools.cycle(t), median_length)) for t in targets]


class DART(HuggingFaceTableToTextDataset):
    x_introduction = "TABLE:"
    y_introduction = "TEXT:"
    task_name = "DART"
    dataset_identifier = "dart"

    def __getitem__(self, idx: int) -> Tuple[str, List[str]]:
        item = self.ds[idx]
        x = " | ".join([" : ".join(triplet) for triplet in item["tripleset"]])
        y = item["annotations"]["text"]
        return x, y

