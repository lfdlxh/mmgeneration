# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class EncryptedTrafficPic(Dataset):
    """"
     Args:
        dataroot (str | :obj:`Path`): Path to the folder root of paired images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        testdir (str): Subfolder of dataroot which contain test images.
            Default: 'test'.
    """

    def __init__(self, dataroot, pipeline, test_mode=False, testdir='test'):
        super().__init__()
        self.img_tensor = torch.randn(3, self.size[0], self.size[1])

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return dict(real_img=self.img_tensor)
