
import os
import random
import nlp

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import LongformerTokenizerFast


from torch.utils.data import DataLoader



class SQuAD(Dataset):

    def __init__(self, root_dir, split):
        self.dataset = torch.load(os.path.join(root_dir, "{:s}_data.pt").format(split))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == '__main__':

    singledocvqa = SQuAD(split='val')

    for batch in singledocvqa:
        print(batch.keys())
        break
