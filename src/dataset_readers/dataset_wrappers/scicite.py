from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from src.utils.app import App
from src.dataset_readers.dataset_wrappers.base import *
from src.datasets.labels import get_mapping_token

field_getter = App()

# e.g. label2text = {0: "negative", 1: "positive"}
label2text = get_mapping_token("scicite")
label_list = ["method", "background", "result"]

@field_getter.add("X")
def get_X(entry):
    return entry['text'] if 'X' not in entry.keys() else entry['X']


@field_getter.add("Y_TEXT")  # 获得标签对应的文本
def get_Y_TEXT(entry):
    if type(entry['label']) == str:
        return entry['label']
    else:
        return label2text[entry['label']] if 'Y_TEXT' not in entry.keys() else entry['Y_TEXT']


@field_getter.add("C")  # 数据集有sentence1,sentence2,label 时，用C表示sentence1
def get_C(entry):
    return "" if 'C' not in entry.keys() else entry['C']


@field_getter.add("Y")  # 获得原始标签 int
def get_Y(entry):
    if type(entry['label']) == str:
        return label_list.index(entry['label'])
    return entry['label'] if 'Y' not in entry.keys() else entry['Y']


@field_getter.add("ALL")
def get_ALL(entry):
    return f"Sentence: {entry['text']}" if 'ALL' not in entry.keys() else entry['ALL']


class SciciteDatasetWrapper(DatasetWrapper):
    name = "scicite"
    sentence_field = "text"
    label_field = "label"

    def __init__(self, dataset_path=None, dataset_split=None, ds_size=None):
        def _abs_label(ex): 
            ex.data['label'] = abs(ex.data['label'])
            return ex

        super().__init__()
        self.task_name = "scicite"
        self.field_getter = field_getter
        self.postfix = "It is"  # for inference
        if dataset_path is None:
            self.dataset = load_dataset("data/scicite", split=dataset_split)
            # self.dataset = self.dataset.map(_abs_label, batched=False, load_from_cache_file=False)
        else:
            self.dataset = Dataset.from_pandas(pd.DataFrame(data=pd.read_json(dataset_path)))

        if dataset_split is not None and isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset[dataset_split]

        if ds_size is not None:
            self.dataset = load_partial_dataset(self.dataset, size=ds_size)
