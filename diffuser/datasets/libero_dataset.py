from collections import namedtuple
import numpy as np
import torch
import pdb
import os
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.libero.benchmark import get_benchmark
from hydra.utils import to_absolute_path
from transformers import AutoModel, AutoTokenizer, logging


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

def get_task_embs(task_embedding_format, descriptions, task_embedding_one_hot_offset, data_max_word_len):
    if task_embedding_format == "one-hot":
        # offset defaults to 1, if we have pretrained another model, this offset
        # starts from the pretrained number of tasks + 1
        offset = task_embedding_one_hot_offset
        descriptions = [f"Task {i+offset}" for i in range(len(descriptions))]

    if task_embedding_format == "bert" or task_embedding_format == "one-hot":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=data_max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=data_max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=data_max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=data_max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    # cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    return task_embs

def pad_fn(input_tensor, min_height, min_width, fill_number=0):
    """
    对输入的 [B, C, H, W] 矩阵进行填充，使其高度和宽度至少为 min_height 和 min_width。
    填充会均匀分配到上下和左右两侧，以保持图片在中心。
    
    参数:
        input_tensor (torch.Tensor 或 np.ndarray): 输入的矩阵，形状为 [B, C, H, W]。
        min_height (int): 最小高度。
        min_width (int): 最小宽度。
        fill_number (int 或 float): 填充的值，默认为 0。
    
    返回:
        padded_tensor (torch.Tensor 或 np.ndarray): 填充后的矩阵。
    """
    # 检查输入类型
    is_torch = isinstance(input_tensor, torch.Tensor)
    is_numpy = isinstance(input_tensor, np.ndarray)
    
    if not (is_torch or is_numpy):
        raise TypeError("输入必须是 torch.Tensor 或 np.ndarray")
    
    # 获取输入的高度和宽度
    B, C, H, W = input_tensor.shape
    
    # 计算需要填充的高度和宽度
    pad_height = max(0, min_height - H)
    pad_width = max(0, min_width - W)
    
    # 将填充均匀分配到上下和左右两侧
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # 定义填充的边界
    # PyTorch 的填充顺序是 (左, 右, 上, 下)
    # NumPy 的填充顺序是 ((上, 下), (左, 右))
    if is_torch:
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        padded_tensor = torch.nn.functional.pad(input_tensor, padding, value=fill_number)
    else:
        padding = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        padded_tensor = np.pad(input_tensor, padding, mode='constant', constant_values=fill_number)
    
    return padded_tensor

class LiberoDataset(torch.utils.data.Dataset):
    def __init__(self, benchmark_name='libero-spatial', task_order_index=0, horizon=64,
        dataset_folder="", obs_modality="vision",
        task_embedding_format=None, task_embedding_one_hot_offset=None,
        data_max_word_len=None, **kwargs):

        # self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        task_order = task_order_index # can be from {0 .. 21}, default to 0, which is [task 0, 1, 2 ...]
        benchmark = get_benchmark(benchmark_name)(task_order)

        # prepare datasets from the benchmark
        datasets = []
        descriptions = []
        shape_meta = None
        n_tasks = benchmark.n_tasks

        for i in range(n_tasks):
            # currently we assume tasks from same benchmark have the same shape_meta
            task_i_dataset, shape_meta = get_dataset(
                    dataset_path=os.path.join(dataset_folder, benchmark.get_task_demonstration(i)),
                    obs_modality=obs_modality,
                    initialize_obs_utils=(i==0),
                    seq_len=horizon,
            )
            # add language to the vision dataset, hence we call vl_dataset
            descriptions.append(benchmark.get_task(i).language)
            datasets.append(task_i_dataset)

        task_embs = get_task_embs(task_embedding_format, descriptions, task_embedding_one_hot_offset, data_max_word_len)

        self.datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(datasets, task_embs)]
        self.n_demos = [data.n_demos for data in datasets]
        self.n_sequences = [data.total_num_sequences for data in datasets]
        self.shape_meta = shape_meta
        self.task_embs = task_embs
        self.task_id = 0
        self.observation_dim = None
        self.action_dim = shape_meta["ac_dim"]
        self.benchmark = benchmark

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.datasets[self.task_id])

    def __getitem__(self, idx):
        data = self.datasets[self.task_id][idx]
        return data

    @property
    def transition_dim(self):
        return self.observation_dim + self.action_dim
