
import os, re
import random
import torch

import numpy as np
from torch.utils.data import Dataset
from utils import correct_alignment
from transformers import LongformerTokenizerFast

from torch.utils.data import DataLoader


class SingleDocVQA(Dataset):

    def __init__(self, imbd_dir, split, tokenizer_weights):
        data = np.load(os.path.join(imbd_dir, "new_imdb_{:s}.npy".format(split)), allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]

        self.tokenizer = LongformerTokenizerFast.from_pretrained(tokenizer_weights)

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        context = ' '.join([word.lower() for word in record['ocr_tokens']])
        answers = list(set(answer.lower() for answer in record['answers']))

        encoding = self.tokenizer(question, context, return_tensors="pt", padding="max_length", truncation=True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)


        sample_info = {'input_ids': input_ids.squeeze(),
                       'attention_mask': attention_mask.squeeze(),
                       'start_positions': start_pos,
                       'end_positions': end_pos
                       }

        return sample_info

    def get_start_end_idx(self, encoding, context, answers):

        pos_idxs = []
        for answer in answers:
            start_idxs = [m.start() for m in re.finditer(re.escape(answer), context)]

            for start_idx in start_idxs:
                end_idx = start_idx + len(answer)
                start_idx, end_idx = correct_alignment(context, answer, start_idx, end_idx)

                if start_idx is not None:
                    pos_idxs.append([start_idx, end_idx])
                    break

        if len(pos_idxs) > 0:
            start_idx, end_idx = random.choice(pos_idxs)

            context_encodings = self.tokenizer.encode_plus(context, padding=True, truncation=True)
            start_positions_context = context_encodings.char_to_token(start_idx)
            end_positions_context = context_encodings.char_to_token(end_idx - 1)

            # here we will compute the start and end position of the answer in the whole example
            # as the example is encoded like this <s> question</s></s> context</s>
            # and we know the position of the answer in the context
            # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
            # this will give us the position of the answer span in whole example
            sep_idx = encoding['input_ids'][0].tolist().index(self.tokenizer.sep_token_id)

            if start_positions_context is not None and end_positions_context is not None:
                start_position = start_positions_context + sep_idx + 1
                end_position = end_positions_context + sep_idx + 1

            else:
                start_position, end_position = 0, 0

            pos_idx = [start_position, end_position]

        else:
            pos_idx = [0, 0]

        start_idxs = pos_idx[0]
        end_idxs = pos_idx[1]

        return start_idxs, end_idxs


def singledocvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch


if __name__ == '__main__':

    singledocvqa = SingleDocVQA("/SSD/Datasets/DocVQA/Task1/pythia_data/imdb/docvqa/", split='val')
