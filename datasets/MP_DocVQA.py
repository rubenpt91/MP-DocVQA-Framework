
import os
import random

import numpy as np
from torch.utils.data import Dataset
from transformers import LongformerTokenizerFast


from torch.utils.data import DataLoader



class MPDocVQA(Dataset):

    def __init__(self, imbd_dir, page_retrieval, split):
        data = np.load(os.path.join(imbd_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]

        self.page_retrieval = page_retrieval.lower()
        assert(self.page_retrieval in ['oracle', 'concat', 'logits'])

        self.max_answers = 2

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        answer_page_idx = record['answer_page_idx']

        if self.page_retrieval == 'oracle':
            context = ' '.join([word.lower() for word in record['ocr_tokens'][answer_page_idx]])

        elif self.page_retrieval == 'concat':
            context = ""
            # context_page_corresp = ""
            context_page_corresp = []
            for page_ix in range(record['imdb_doc_pages']):
                page_context = " ".join([word.lower() for word in record['ocr_tokens'][page_ix]])
                context += " " + page_context
                # context_page_corresp += " " + ''.join([str(page_ix)]*len(page_context))
                context_page_corresp.extend([-1] + [page_ix]*len(page_context))
                # context_page_corresp += " " + ' '.join([''.join([str(page_ix)]*len(word)) for word in record['ocr_tokens'][page_ix]])

            context = context.strip()
            context_page_corresp = context_page_corresp[1:]

        elif self.page_retrieval == 'logits':
            context = []
            for page_ix in range(record['imdb_doc_pages']):
                context.append(' '.join([word.lower() for word in record['ocr_tokens'][page_ix]]))

        answers = list(set(answer.lower() for answer in record['answers']))

        if self.page_retrieval == 'oracle' or self.page_retrieval == 'concat':
            start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        elif self.page_retrieval == 'logits':
            start_idxs, end_idxs = self._get_start_end_idx(context[record['answer_page_idx']], answers)

        sample_info = {'question_id': record['question_id'],
                       'questions': question,
                       'contexts': context,
                       'answers': answers,
                       'start_indxs': start_idxs,
                       'end_indxs': end_idxs,
                       'answer_page_idx': record['answer_page_idx']
                       }

        if self.page_retrieval == 'concat':
            sample_info['context_page_corresp'] = context_page_corresp

        return sample_info

    def _get_start_end_idx(self, context, answers):

        answer_positions = []
        for answer in answers:
            start_idx = context.find(answer)

            if start_idx != -1:
                end_idx = start_idx + len(answer)
                answer_positions.append([start_idx, end_idx])

        if len(answer_positions) > 0:
            start_idx, end_idx = random.choice(answer_positions)  # If both answers are in the context. Choose one randomly.
        else:
            start_idx, end_idx = 0, 0  # If the indices are out of the sequence length they are ignored. Therefore, we set them as a very big number.

        return start_idx, end_idx


def singledocvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch


if __name__ == '__main__':

    mp_docvqa = MPDocVQA("/SSD/Datasets/DocVQA/Task1/pythia_data/imdb/collection", split='val')
