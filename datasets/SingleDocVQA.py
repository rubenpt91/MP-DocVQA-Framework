
import os
import random

import numpy as np
from torch.utils.data import Dataset


class SingleDocVQA(Dataset):

    def __init__(self, imbd_dir, images_dir, split, kwargs):
        data = np.load(os.path.join(imbd_dir, "new_imdb_{:s}.npy".format(split)), allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]

        self.max_answers = 2
        self.images_dir = images_dir

        self.use_images = kwargs.get('use_images', False)
        self.get_raw_ocr_data = kwargs.get('get_raw_ocr_data', False)

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        context = ' '.join([word.lower() for word in record['ocr_tokens']])
        answers = list(set(answer.lower() for answer in record['answers']))

        if self.use_images:
            image_names = os.path.join(self.images_dir, "{:s}.png".format(record['image_name']))

        if self.get_raw_ocr_data:
            words = [word.lower() for word in record['ocr_tokens']]
            boxes = np.array([bbox for bbox in record['ocr_normalized_boxes']])

        start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        sample_info = {'question_id': record['question_id'],
                       'questions': question,
                       'contexts': context,
                       'answers': answers,
                       'start_indxs': start_idxs,
                       'end_indxs': end_idxs
                       }

        if self.use_images:
            sample_info['image_names'] = image_names

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes

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

    singledocvqa = SingleDocVQA("/SSD/Datasets/DocVQA/Task1/pythia_data/imdb/docvqa/", split='val')
