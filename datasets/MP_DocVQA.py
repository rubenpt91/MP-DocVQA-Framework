
import os
import random

import numpy as np
from torch.utils.data import Dataset


class MPDocVQA(Dataset):

    def __init__(self, imbd_dir, images_dir, page_retrieval, split, kwargs):
        data = np.load(os.path.join(imbd_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]

        self.page_retrieval = page_retrieval.lower()
        assert(self.page_retrieval in ['oracle', 'concat', 'logits', 'custom'])

        self.max_answers = 2
        self.images_dir = images_dir

        self.use_images = kwargs.get('use_images', False)
        self.get_raw_ocr_data = kwargs.get('get_raw_ocr_data', False)

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        answer_page_idx = record['answer_page_idx']
        num_pages = record['imdb_doc_pages']

        if self.page_retrieval == 'oracle':
            context = ' '.join([word.lower() for word in record['ocr_tokens'][answer_page_idx]])
            context_page_corresp = None

            if self.use_images:
                image_names = os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name'][answer_page_idx]))

            if self.get_raw_ocr_data:
                words = [word.lower() for word in record['ocr_tokens'][answer_page_idx]]
                boxes = np.array([bbox for bbox in record['ocr_normalized_boxes'][answer_page_idx]])

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

            if self.use_images:
                image_names = [os.path.join(self.images_dir, "{:s}.jpg".format(image_name)) for image_name in record['image_name']]

            if self.get_raw_ocr_data:
                # for p in range(num_pages):
                #     for bbox in record['ocr_normalized_boxes'][p]:
                #         assert (bbox[0] <= bbox[2] and bbox[1] <= bbox[3])

                words, boxes = [], []
                for p in range(num_pages):
                    words.extend([word.lower() for word in record['ocr_tokens'][p]])

                    mod_boxes = record['ocr_normalized_boxes'][p]
                    mod_boxes[:, 1] = mod_boxes[:, 1]/num_pages + p/num_pages
                    mod_boxes[:, 3] = mod_boxes[:, 3]/num_pages + p/num_pages

                    # for bbox in mod_boxes:
                    #     assert (bbox[0] <= bbox[2] and bbox[1] <= bbox[3])

                    # (record['ocr_normalized_boxes'][p] / num_pages) + p / num_pages  # (Wrong) - This would change left and right also.
                    boxes.extend(mod_boxes)  # bbox in l,t,r,b --> It is correct to move the whole bounding box. It will be similar as if the pages were displayed in diagonal.


                    # words.append([word.lower() for word in record['ocr_tokens'][p]])
                # boxes = record['ocr_normalized_boxes']
                boxes = np.array(boxes)

        elif self.page_retrieval in ['logits', 'custom']:
            context = []
            for page_ix in range(record['imdb_doc_pages']):
                context.append(' '.join([word.lower() for word in record['ocr_tokens'][page_ix]]))

            context_page_corresp = None

            if self.use_images:
                image_names = [os.path.join(self.images_dir, "{:s}.jpg".format(image_name)) for image_name in record['image_name']]

            if self.get_raw_ocr_data:
                words = []
                boxes = record['ocr_normalized_boxes']
                for p in range(num_pages):
                    words.append([word.lower() for word in record['ocr_tokens'][p]])

        answers = list(set(answer.lower() for answer in record['answers']))

        if self.page_retrieval == 'oracle' or self.page_retrieval == 'concat':
            start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        elif self.page_retrieval == 'logits':
            start_idxs, end_idxs = self._get_start_end_idx(context[record['answer_page_idx']], answers)

        sample_info = {'question_id': record['question_id'],
                       'questions': question,
                       'contexts': context,
                       'context_page_corresp': context_page_corresp,
                       'answers': answers,
                       # 'start_indxs': start_idxs,
                       # 'end_indxs': end_idxs,
                       'answer_page_idx': record['answer_page_idx']
                       }

        if self.use_images:
            sample_info['image_names'] = image_names

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes
            sample_info['num_pages'] = num_pages

        else:
            sample_info['start_indxs'] = start_idxs
            sample_info['end_indxs'] = end_idxs
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
