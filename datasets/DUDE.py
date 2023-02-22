import os
import random
from PIL import Image

import numpy as np
from datasets.MP_DocVQA import MPDocVQA


class DUDE(MPDocVQA):
    def __init__(self, imbd_dir, images_dir, page_retrieval, split, data_kwargs, **kwargs):
        super(DUDE, self).__init__(imbd_dir, images_dir, page_retrieval, split, data_kwargs)

        if self.page_retrieval == "oracle":
            raise ValueError(
                "'Oracle' set-up is not valid for DUDE, since there is no GT for the answer page."
            )

        self.list_strategy = kwargs.get("list_strategy")
        self.none_strategy = kwargs.get("none_strategy", "none")
        self.qtype_learning = kwargs.get("qtype_learning", None)
        self.atype_learning = kwargs.get("atype_learning", None)
        # Defaults: {'none_strategy': 'token', 'list_strategy': 'separator',
        #'atype_learning': 'None', 'qtype_learning': 'None'}

    def __getitem__(self, idx):

        record = self.imdb[idx]

        question = record["question"]
        answers = record["answers"] 
        num_pages = record["num_pages"]
        answer_page_idx = random.choice(range(num_pages))  # random
        record["answer_page_idx"] = answer_page_idx  # putting it in here

        if self.page_retrieval == "oracle":
            raise ValueError(
                "'Oracle' set-up is not valid for DUDE, since there is no GT for the answer page."
            )
            # is there not in the case of extractive?
            """
            context = ' '.join([word.lower() for word in record['ocr_tokens'][answer_page_idx]])
            context_page_corresp = None

            if self.use_images:
                image_names = os.path.join(self.images_dir, "{:s}".format(record['image_name'][answer_page_idx]))
                images = Image.open(image_names).convert("RGB")

            if self.get_raw_ocr_data:
                words = [word.lower() for word in record['ocr_tokens'][answer_page_idx]]
                boxes = np.array([bbox for bbox in record['ocr_normalized_boxes'][answer_page_idx]])
            """

        elif self.page_retrieval == "concat":
            context = ""
            context_page_corresp = []
            for page_ix in range(record["num_pages"]):
                page_context = " ".join([word.lower() for word in record["ocr_tokens"][page_ix]])
                context += " " + page_context
                context_page_corresp.extend([-1] + [page_ix] * len(page_context))

            context = context.strip()
            context_page_corresp = context_page_corresp[1:]

            if self.use_images:
                image_names = [
                    os.path.join(self.images_dir, "{:s}".format(image_name))
                    for image_name in record["image_name"]
                ]
                images = [Image.open(img_path).convert("RGB") for img_path in image_names]

            if self.get_raw_ocr_data:
                words, boxes = [], []
                for p in range(num_pages):
                    if len(record["ocr_tokens"]):
                        continue

                    words.extend([word.lower() for word in record["ocr_tokens"][p]])

                    mod_boxes = record["ocr_normalized_boxes"][p]
                    mod_boxes[:, 1] = mod_boxes[:, 1] / num_pages + p / num_pages
                    mod_boxes[:, 3] = mod_boxes[:, 3] / num_pages + p / num_pages

                    boxes.extend(mod_boxes)  # bbox in l,t,r,b

                boxes = np.array(boxes)

        elif self.page_retrieval == "logits":
            context = []
            for page_ix in range(record["num_pages"]):
                context.append(" ".join([word.lower() for word in record["ocr_tokens"][page_ix]]))

            context_page_corresp = None

            if self.use_images:
                image_names = [
                    os.path.join(self.images_dir, "{:s}".format(image_name))
                    for image_name in record["image_name"]
                ]
                images = [Image.open(img_path).convert("RGB") for img_path in image_names]

            if self.get_raw_ocr_data:
                words = []
                boxes = record["ocr_normalized_boxes"]
                for p in range(num_pages):
                    words.append([word.lower() for word in record["ocr_tokens"][p]])

        elif self.page_retrieval == "custom":
            record["imdb_doc_pages"] = num_pages
            first_page, last_page = self.get_pages(record)
            answer_page_idx = answer_page_idx - first_page
            num_pages = len(range(first_page, last_page))

            words = []
            boxes = []
            context = []
            image_names = []

            for page_ix in range(first_page, last_page):
                words.append([word.lower() for word in record["ocr_tokens"][page_ix]])
                boxes.append(np.array(record["ocr_normalized_boxes"][page_ix], dtype=np.float32))
                context.append(" ".join([word.lower() for word in record["ocr_tokens"][page_ix]]))
                image_names.append(
                    os.path.join(self.images_dir, "{:s}".format(record["image_name"][page_ix]))
                )

            context_page_corresp = None

            if num_pages < self.max_pages:
                for _ in range(self.max_pages - num_pages):
                    words.append([""])
                    boxes.append(np.zeros([1, 4], dtype=np.float32))

            if self.use_images:
                images = [Image.open(img_path).convert("RGB") for img_path in image_names]
                images += [
                    Image.new("RGB", (0, 0)) for i in range(self.max_pages - len(image_names))
                ]  # Pad with 0x0 images.

        if self.page_retrieval == "oracle" or self.page_retrieval == "concat":
            start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        elif self.page_retrieval == "logits":
            start_idxs, end_idxs = self._get_start_end_idx(context[answer_page_idx], answers)

        if len(answers) == 0 and self.none_strategy == 'none':
            answers = ["none"]
            
        if len(answers) > 1 and 'list' in record['extra']['answer_type'] and self.list_strategy:
            if self.list_strategy == "separator":
                answers = " | ".join(answers)
                #print(answers)
        
        sample_info = {
            "question_id": record["question_id"],
            "questions": question,
            "contexts": context,
            "context_page_corresp": context_page_corresp,
            "answers": answers,
            "answer_page_idx": answer_page_idx,
            "imdb_doc_pages": num_pages,
        }

        if self.use_images:
            sample_info["image_names"] = image_names
            sample_info["images"] = images
            if not sample_info['images']:
                print(f"NO IMAGES: {sample_info['images']}")

        if self.get_raw_ocr_data:
            sample_info["words"] = words
            sample_info["boxes"] = boxes
            sample_info["num_pages"] = num_pages

        else:  # Information for extractive models
            # sample_info['context_page_corresp'] = context_page_corresp
            sample_info["start_indxs"] = start_idxs
            sample_info["end_indxs"] = end_idxs

        if self.get_doc_id:
            sample_info['doc_id'] = [record['image_name'][page_ix] for page_ix in range(first_page, last_page)]
        

        ## parse list and none strategies; adjust accordingly
        #if self.list_strategy: 
        #   do

        return sample_info

if __name__ == "__main__":
    #dude_dataset = DUDE("/SSD/Datasets/DUDE/imdb/", split="val")
    dude_dataset = DUDE("/cw/liir_data/NoCsBack/jordy/DUDE/", split="val")
    print(dude_dataset[1])
