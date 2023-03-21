
import os
import random
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
import utils


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
        self.max_pages = kwargs.get('max_pages', 1)
        self.get_doc_id = False

    def __len__(self):
        return len(self.imdb)

    def sample(self, idx=None, question_id=None):

        if idx is not None:
            return self.__getitem__(idx)

        if question_id is not None:
            for idx in range(self.__len__()):
                record = self.imdb[idx]
                if record['question_id'] == question_id:
                    return self.__getitem__(idx)

            raise ValueError("Question ID {:d} not in dataset.".format(question_id))

        idx = random.randint(0, self.__len__())
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        record = self.imdb[idx]

        question = record['question']
        answers = list(set(answer.lower() for answer in record['answers']))
        answer_page_idx = record['answer_page_idx']
        num_pages = record['imdb_doc_pages']

        if self.page_retrieval == 'oracle':
            context = ' '.join([word.lower() for word in record['ocr_tokens'][answer_page_idx]])
            context_page_corresp = None
            num_pages = 1

            if self.use_images:
                image_names = os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name'][answer_page_idx]))
                images = Image.open(image_names).convert("RGB")

            if self.get_raw_ocr_data:
                words = [word.lower() for word in record['ocr_tokens'][answer_page_idx]]
                boxes = np.array([bbox for bbox in record['ocr_normalized_boxes'][answer_page_idx]])

        elif self.page_retrieval == 'concat':
            context = ""
            context_page_corresp = []
            for page_ix in range(record['imdb_doc_pages']):
                page_context = " ".join([word.lower() for word in record['ocr_tokens'][page_ix]])
                context += " " + page_context
                context_page_corresp.extend([-1] + [page_ix]*len(page_context))

            context = context.strip()
            context_page_corresp = context_page_corresp[1:]  # Remove the first character corresponding to the first space.

            if self.get_raw_ocr_data:
                words = []
                for p in range(num_pages):
                    words.extend([word.lower() for word in record['ocr_tokens'][p]])

                    """
                    mod_boxes = record['ocr_normalized_boxes'][p]
                    mod_boxes[:, 1] = mod_boxes[:, 1]/num_pages + p/num_pages
                    mod_boxes[:, 3] = mod_boxes[:, 3]/num_pages + p/num_pages

                    boxes.extend(mod_boxes)  # bbox in l,t,r,b
                    """
                # boxes = np.array(boxes)
                boxes = record['ocr_normalized_boxes']

            else:
                words, boxes = None, None

            if self.use_images:
                image_names = [os.path.join(self.images_dir, "{:s}.jpg".format(image_name)) for image_name in record['image_name']]
                images = [Image.open(img_path).convert("RGB") for img_path in image_names]
                images, boxes = utils.create_grid_image(images, boxes)

            else:
                boxes = np.array(boxes)

        elif self.page_retrieval == 'logits':
            context = []
            for page_ix in range(record['imdb_doc_pages']):
                context.append(' '.join([word.lower() for word in record['ocr_tokens'][page_ix]]))

            context_page_corresp = None

            if self.use_images:
                image_names = [os.path.join(self.images_dir, "{:s}.jpg".format(image_name)) for image_name in record['image_name']]
                images = [Image.open(img_path).convert("RGB") for img_path in image_names]

            if self.get_raw_ocr_data:
                words = []
                boxes = record['ocr_normalized_boxes']
                for p in range(num_pages):
                    words.append([word.lower() for word in record['ocr_tokens'][p]])

        elif self.page_retrieval == 'custom':
            first_page, last_page = self.get_pages(record)
            answer_page_idx = answer_page_idx - first_page
            num_pages = len(range(first_page, last_page))

            words = []
            boxes = []
            context = []
            image_names = []

            for page_ix in range(first_page, last_page):
                words.append([word.lower() for word in record['ocr_tokens'][page_ix]])
                boxes.append(np.array(record['ocr_normalized_boxes'][page_ix], dtype=np.float32))
                context.append(' '.join([word.lower() for word in record['ocr_tokens'][page_ix]]))
                image_names.append(os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name'][page_ix])))

            context_page_corresp = None

            if num_pages < self.max_pages:
                for _ in range(self.max_pages - num_pages):
                    words.append([''])
                    boxes.append(np.zeros([1, 4], dtype=np.float32))

            if self.use_images:
                images = [Image.open(img_path).convert("RGB") for img_path in image_names]
                images += [Image.new('RGB', (2, 2)) for i in range(self.max_pages - len(image_names))]  # Pad with 2x2 images.

        if self.page_retrieval in ['oracle', 'concat', 'none']:
            start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        elif self.page_retrieval == 'logits':
            start_idxs, end_idxs = self._get_start_end_idx(context[answer_page_idx], answers)

        else:
            start_idxs, end_idxs = None, None

        sample_info = {'question_id': record['question_id'],
                       'questions': question,
                       'contexts': context,
                       'context_page_corresp': context_page_corresp,
                       'answers': answers,
                       'answer_page_idx': answer_page_idx,
                       'num_pages': num_pages
                       }

        if self.use_images:
            sample_info['image_names'] = image_names
            sample_info['images'] = images

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes

        else:  # Information for extractive models
            sample_info['start_indxs'] = start_idxs
            sample_info['end_indxs'] = end_idxs

        if self.get_doc_id:
            sample_info['doc_id'] = [record['image_name'][page_ix] for page_ix in range(first_page, last_page)]

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

    def get_pages(self, sample_info):
        # TODO implement margins
        answer_page = sample_info['answer_page_idx']
        document_pages = sample_info['imdb_doc_pages']
        if document_pages <= self.max_pages:
            first_page, last_page = 0, document_pages

        else:
            first_page_lower_bound = max(0, answer_page-self.max_pages+1)
            first_page_upper_bound = answer_page
            first_page = random.randint(first_page_lower_bound, first_page_upper_bound)
            last_page = first_page + self.max_pages

            if last_page > document_pages:
                last_page = document_pages
                first_page = last_page-self.max_pages

            try:
                assert(answer_page in range(first_page, last_page))  # answer page is in selected range.
                assert(last_page-first_page == self.max_pages)  # length of selected range is correct.
            except:
                assert (answer_page in range(first_page, last_page))  # answer page is in selected range.
                assert (last_page - first_page == self.max_pages)  # length of selected range is correct.
        # print("[{:d} <= {:d} < {:d}][{:d} + {:d}]".format(first_page, answer_page, last_page, len(range(first_page, last_page)), padding_pages))
        assert(answer_page in range(first_page, last_page))
        assert(first_page >= 0)
        assert(last_page <= document_pages)

        return first_page, last_page


def mpdocvqa_collate_fn(batch):  # It's actually the same as in SP-DocVQA...
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch


if __name__ == '__main__':

    mp_docvqa = MPDocVQA("/SSD/Datasets/DocVQA/Task1/pythia_data/imdb/collection", split='val')
