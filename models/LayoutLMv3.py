import random
import numpy as np

import torch
import torch.nn as nn
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
from PIL import Image
import models._model_utils as model_utils
import utils


# from utils import correct_alignment
# from utils import create_grid_image

# from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Model  # TODO Remove
# from transformers.models.layoutlmv3.processing_layoutlmv3 import LayoutLMv3Processor    # TODO Remove


class LayoutLMv3:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.processor = LayoutLMv3Processor.from_pretrained(config['model_weights'], apply_ocr=False)  # Check that this do not fuck up the code.
        self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.ignore_index = 9999  # 0

        # img = Image.open('/SSD2/MP-DocVQA/images/nkkh0227_p2.jpg')
        # self.processor(img, 'question', ['words'], boxes=[[1, 2, 3, 4]])

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def forward(self, batch, return_pred_answer=False):

        bs = len(batch['question_id'])
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']
        images = batch['images']

        if self.page_retrieval == 'logits':
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            answ_confidence = []
            for batch_idx in range(bs):
                images = [Image.open(img_path).convert("RGB") for img_path in batch['image_names'][batch_idx]]
                boxes = [(bbox * 1000).astype(int) for bbox in batch['boxes'][batch_idx]]
                document_encoding = self.processor(images, [question[batch_idx]] * len(images), batch["words"][batch_idx], boxes=boxes, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                answer_page = None
                document_outputs = None
                for page_idx in range(len(document_encoding['input_ids'])):
                    # input_ids = document_encoding["input_ids"][page_idx].to(self.model.device)
                    # attention_mask = document_encoding["attention_mask"][page_idx].to(self.model.device)

                    page_inputs = {k: v[page_idx].unsqueeze(dim=0) for k, v in document_encoding.items()}
                    # Retrieval with logits is available only during inference and hence, the start and end indices are not used.
                    # start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
                    # end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

                    page_outputs = self.model(**page_inputs)
                    pred_answer, answer_conf = self.get_answer_from_model_output(page_inputs["input_ids"].unsqueeze(dim=0), page_outputs)

                    """
                    start_logits_cnf = [page_outputs.start_logits[batch_ix, max_start_logits_idx.item()].item() for batch_ix, max_start_logits_idx in enumerate(page_outputs.start_logits.argmax(-1))][0]
                    end_logits_cnf = [page_outputs.end_logits[batch_ix, max_end_logits_idx.item()].item() for batch_ix, max_end_logits_idx in enumerate(page_outputs.end_logits.argmax(-1))][0]
                    page_logits = np.mean([start_logits_cnf, end_logits_cnf])
                    """

                    if answer_conf[0] > max_logits:
                        answer_page = page_idx
                        document_outputs = page_outputs
                        max_logits = answer_conf[0]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.extend(self.get_answer_from_model_output([document_encoding["input_ids"][answer_page]], document_outputs)[0] if return_pred_answer else None)
                pred_answer_pages.append(answer_page)
                answ_confidence.append(max_logits)

        else:

            boxes = [(bbox * 1000).astype(int) for bbox in batch['boxes']]  # Scale boxes 0->1 to 0-->1000.
            encoding = self.processor(images, question, batch["words"], boxes=boxes, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

            start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)
            outputs = self.model(**encoding, start_positions=start_pos, end_positions=end_pos)
            pred_answers, answ_confidence = self.get_answer_from_model_output(encoding.input_ids, outputs) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = [batch['context_page_corresp'][batch_idx][pred_start_idx] if len(batch['context_page_corresp'][batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]

            elif self.page_retrieval == 'none':
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, answ_confidence

    def get_concat_v_multi_resize(self, im_list, resample=Image.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample) for im in im_list]

        # Fix equal height for all images (breaks the aspect ratio).
        heights = [im.height for im in im_list]
        im_list_resize = [im.resize((im.height, max(heights)), resample=resample) for im in im_list_resize]

        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new('RGB', (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    def get_start_end_idx(self, encoding, context, answers):
        pos_idx = []
        for batch_idx in range(len(encoding.input_ids)):
            answer_pos = []
            for answer in answers[batch_idx]:
                encoded_answer = [token for token in self.processor.tokenizer.encode([answer], boxes=[0, 0, 0, 0]) if token not in self.processor.tokenizer.all_special_ids]
                answer_tokens_length = len(encoded_answer)

                for token_pos in range(len(encoding.input_ids[batch_idx])):
                    if encoding.input_ids[batch_idx][token_pos: token_pos+answer_tokens_length].tolist() == encoded_answer:
                        answer_pos.append([token_pos, token_pos + answer_tokens_length-1])

            if len(answer_pos) == 0:
                pos_idx.append([self.ignore_index, self.ignore_index])

            else:
                answer_pos = random.choice(answer_pos)  # To add variability, pick a random correct span.
                pos_idx.append(answer_pos)

        start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
        end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)

        return start_idxs, end_idxs

    def get_answer_from_model_output(self, input_tokens, outputs):
        start_idxs = torch.argmax(outputs.start_logits, axis=1)
        end_idxs = torch.argmax(outputs.end_logits, axis=1)

        answers = [self.processor.tokenizer.decode(input_tokens[batch_idx][start_idxs[batch_idx]: end_idxs[batch_idx]+1], skip_special_tokens=True).strip() for batch_idx in range(len(input_tokens))]
        # answers_conf = ((outputs.start_logits.max(dim=1).values + outputs.end_logits.max(dim=1).values) / 2).tolist()

        start_logits = outputs.start_logits.softmax(dim=1).detach().cpu()
        end_logits = outputs.end_logits.softmax(dim=1).detach().cpu()
        answ_confidence = []
        for batch_idx in range(len(input_tokens)):
            conf_mat = np.matmul(np.expand_dims(start_logits[batch_idx].unsqueeze(dim=0), -1),
                                 np.expand_dims(end_logits[batch_idx].unsqueeze(dim=0), 1)).squeeze(axis=0)

            answ_confidence.append(
                conf_mat[start_idxs[batch_idx], end_idxs[batch_idx]].item()
            )

        aansw_confidence = model_utils.get_extractive_confidence(outputs)

        return answers, answ_confidence
