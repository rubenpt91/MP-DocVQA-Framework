import os
import random
import json
from PIL import Image

import numpy as np
from datasets.MP_DocVQA import MPDocVQA


def empty_image(height=2, width=2):
    i = np.ones((height, width, 3), np.uint8) * 255  # whitepage
    return i


def open_precomputed(images_dir, split):
    print(f"Loading precomputed visual features from {images_dir}-{split}")
    pagename_idx = json.load(
        open(os.path.join(images_dir, f"{split}-visfeats.json"), "r")
    )
    page_visual_features = np.load(os.path.join(images_dir, f"{split}-visfeats.npz"))[
        "arr_0"
    ]
    page_visual_features = page_visual_features.reshape((-1, 197, 768))  # indexation
    return pagename_idx, page_visual_features


class DUDE(MPDocVQA):
    def __init__(
        self, imbd_dir, images_dir, page_retrieval, split, data_kwargs, **kwargs
    ):
        super(DUDE, self).__init__(
            imbd_dir, images_dir, page_retrieval, split, data_kwargs
        )

        if self.page_retrieval == "oracle":
            raise ValueError(
                "'Oracle' set-up is not valid for DUDE, since there is no GT for the answer page."
            )
        self.list_strategy = kwargs.get("list_strategy")
        self.none_strategy = (
            kwargs.get("none_strategy") if kwargs.get("none_strategy") else "none"
        )
        if self.none_strategy is None:
            self.none_strategy = "none"
        self.qtype_learning = kwargs.get("qtype_learning", None)
        self.atype_learning = kwargs.get("atype_learning", None)

        if self.qtype_learning == "MLP":
            self.QTYPES = [
                "abstractive",
                "extractive",
                "list/abstractive",
                "list/extractive",
                "not-answerable",
            ]
        if self.atype_learning == "MLP":
            self.q_id_to_ATYPE = json.load(
                open(
                    os.path.join(images_dir, f"trainval_QA-pairs_all_labelled.json"),
                    "r",
                )
            )
            self.ATYPES = sorted(
                set([v for v in self.q_id_to_ATYPE.values() if v != ""])
            )

        self.precomputed_visual_feats = bool(
            data_kwargs.get("precomputed_visual_feats", False)
            or kwargs.get("precomputed_visual_feats", False)
        )
        if self.precomputed_visual_feats:
            (
                self.pagename_idx,
                self.page_visual_features,
            ) = open_precomputed(images_dir, split)

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
        elif self.page_retrieval == "concat":
            context = ""
            context_page_corresp = []
            for page_ix in range(record["num_pages"]):
                page_context = " ".join(
                    [word.lower() for word in record["ocr_tokens"][page_ix]]
                )
                context += " " + page_context
                context_page_corresp.extend([-1] + [page_ix] * len(page_context))

            context = context.strip()
            context_page_corresp = context_page_corresp[1:]

            if self.use_images:
                image_names = [
                    os.path.join(self.images_dir, "{:s}".format(image_name))
                    for image_name in record["image_name"]
                ]
                images = [
                    Image.open(img_path).convert("RGB") for img_path in image_names
                ]

            if self.get_raw_ocr_data:
                words, boxes = [], []
                for p in range(num_pages):
                    if len(record["ocr_tokens"][p]) == 0:
                        continue

                    words.extend([word.lower() for word in record["ocr_tokens"][p]])

                    mod_boxes = np.array(record["ocr_normalized_boxes"][p])
                    mod_boxes[:, 1] = mod_boxes[:, 1] / num_pages + p / num_pages
                    mod_boxes[:, 3] = mod_boxes[:, 3] / num_pages + p / num_pages

                    boxes.extend(mod_boxes)  # bbox in l,t,r,b

                boxes = np.array(boxes)

        elif self.page_retrieval == "logits":
            context = []
            for page_ix in range(record["num_pages"]):
                context.append(
                    " ".join([word.lower() for word in record["ocr_tokens"][page_ix]])
                )

            context_page_corresp = None

            if self.use_images:
                image_names = [
                    os.path.join(self.images_dir, "{:s}".format(image_name))
                    for image_name in record["image_name"]
                ]
                images = [
                    Image.open(img_path).convert("RGB") for img_path in image_names
                ]

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
                boxes.append(
                    np.array(record["ocr_normalized_boxes"][page_ix], dtype=np.float32)
                )
                context.append(
                    " ".join([word.lower() for word in record["ocr_tokens"][page_ix]])
                )
                image_names.append(
                    os.path.join(
                        self.images_dir, "{:s}".format(record["image_name"][page_ix])
                    )
                )

            context_page_corresp = None

            # PADDING
            if num_pages < self.max_pages:
                for _ in range(self.max_pages - num_pages):
                    words.append([""])
                    boxes.append(np.zeros([1, 4], dtype=np.float32))

            if self.use_images:
                if self.precomputed_visual_feats:
                    images = self.retrieve_precomputed(image_names)
                else:
                    images = [
                        Image.open(img_path).convert("RGB") for img_path in image_names
                    ]
                    images += [
                        empty_image() for i in range(self.max_pages - len(image_names))
                    ]

        if self.page_retrieval == "oracle" or self.page_retrieval == "concat":
            start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        elif self.page_retrieval == "logits":
            start_idxs, end_idxs = self._get_start_end_idx(
                context[answer_page_idx], answers
            )

        # novel strategies
        if len(answers) == 0:
            if self.none_strategy == "none" or self.none_strategy == "token":
                answers = ["none"]
            elif self.none_strategy == "special_token":
                answers = ["NA"]

        if (
            len(answers) > 1
            and "list" in record["extra"]["answer_type"]
            and self.list_strategy
        ):
            if self.list_strategy == "separator":
                answers = " | ".join(answers)
            elif self.list_strategy == "special_token":
                answers = " [LSEP] ".join(answers)

        if self.qtype_learning == "special_token":
            answers = [a + f" & {record['extra']['answer_type']}" for a in answers]

        if self.atype_learning == "special_token":
            answers = [a + f" & {record['extra']['answer_data_type']}" for a in answers]

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
            if not sample_info["images"]:
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
            sample_info["doc_id"] = [
                record["image_name"][page_ix]
                for page_ix in range(first_page, last_page)
            ]
        if self.qtype_learning == "MLP":
#           record['qtype'] = record['extra']['answer_type']
            record['qtype_idx'] = self.QTYPES.index(record['extra']['answer_type'])
        if self.atype_learning == "MLP":
#           record['atype'] = self.q_id_to_ATYPE[record["question_id"]]
            record['atype_idx'] = self.ATYPES.index(self.q_id_to_ATYPE[record["question_id"]])

        return sample_info

    def retrieve_precomputed(self, image_names):
        images = []
        for img_path in image_names:
            index = self.pagename_idx.get(
                img_path.replace(self.images_dir + "/", ""), self.pagename_idx["PAD"]
            )
            images.append(self.page_visual_features[index])
        images += [
            self.page_visual_features[self.pagename_idx["PAD"]]
            for i in range(self.max_pages - len(image_names))
        ]
        return images


if __name__ == "__main__":
    # dude_dataset = DUDE("/SSD/Datasets/DUDE/imdb/", split="val")
    dude_dataset = DUDE("/cw/liir_data/NoCsBack/jordy/DUDE/", split="val")
    print(dude_dataset[1])
