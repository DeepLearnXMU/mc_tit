import os
import torch
import random
class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, split: str):
        self.split = split
        self.img_feat_path = []
        for i in range(len(os.listdir(feat_path))):
            self.img_feat_path.append(os.path.join(feat_path,split+str(i)+".pth"))
        self.size = len(self.img_feat_path)


    def __getitem__(self, idx):
        return torch.load(self.img_feat_path[idx])

    def __len__(self):
        return self.size


import gc
import os
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
from torch.utils.data.dataset import Dataset
import torch


DEFAULT_INPUT_FILE_NAME = "input.data"


class MMAPImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_iter: Iterable[np.ndarray],
        mmap_path: str = None,
        size: int = None,        
    ) -> None:
        super().__init__()

        self.mmap_inputs: np.ndarray = None

        if mmap_path is None:
            mmap_path = os.path.abspath(os.getcwd())
        self._mkdir(mmap_path)

        self.mmap_input_path = os.path.join(mmap_path, DEFAULT_INPUT_FILE_NAME)

        # If the total size is not known we load the dataset in memory first
        if size is None:
            input_iter = self._consume_iterable(input_iter)
            size = len(input_iter)

        self.length = size

        for idx, input in enumerate(input_iter):
            if self.mmap_inputs is None:
                self.mmap_inputs = self._init_mmap(
                    self.mmap_input_path, input.dtype, (self.length, *input.shape)
                )

            self.mmap_inputs[idx][:] = input[:]

        del input_iter
        gc.collect()


    def __getitem__(self, idx: int) -> torch.Tensor:
        return  torch.tensor(self.mmap_inputs[idx])


    def __len__(self) -> int:
        return self.length


    def _consume_iterable(self, input_iter: Iterable[np.ndarray]) -> Tuple[List[np.ndarray]]:
        inputs = []

        for idx, input in enumerate(input_iter):
            inputs.append(input)

        if not isinstance(inputs[0], np.ndarray):
            raise TypeError("Inputs and labels must be of type np.ndarray")

        return inputs


    def _mkdir(self, path: str) -> None:
        if os.path.exists(path):
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return
        except:
            raise ValueError(
                "Failed to create the path (check the user write permissions)."
            )


    def _init_mmap(self, path: str, dtype: np.dtype, shape: Tuple[int], remove_existing: bool = False) -> np.ndarray:
        open_mode = "r+"

        if remove_existing:
            open_mode = "w+"
        
        return np.memmap(
            path,
            dtype=dtype,
            mode=open_mode,
            shape=shape,
        )

def image_feature_iter(feat_path,split):

    img_feat_path = os.listdir(feat_path)
    for i in range(len(img_feat_path)):
        img_feat = np.load((os.path.join(feat_path,split+str(i)+".npy")))
        yield img_feat
