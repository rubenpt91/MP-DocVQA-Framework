import random

import torch
import torch.nn as nn
from transformers import LayoutLMv2Processor, LayoutLMv2ForQuestionAnswering
from PIL import Image
import cv2
import models._model_utils as model_utils

# from transformers.models.layoutlmv2.modeling_layoutlmv2 import LayoutLMv2Model    # TODO Remove
# from transformers.models.layoutlmv2.processing_layoutlmv2 import LayoutLMv2Processor    # TODO Remove


# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb
class LayoutLMv2:

    def __init__(self, config):
        self.batch_size = config.batch_size
        # self.processor = LayoutLMv2Processor.from_pretrained(config['model_weights'])  # Check that this do not fuck up the code.
        self.processor = LayoutLMv2Processor.from_pretrained(config.model_weights, apply_ocr=False)  # Check that this do not fuck up the code.
        self.model = LayoutLMv2ForQuestionAnswering.from_pretrained(config.model_weights)
        self.page_retrieval = config.page_retrieval.lower()
        self.ignore_index = 9999  # 0


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
                boxes = [(bbox * 1000).astype(int) for bbox in batch['boxes'][batch_idx]]
                document_encoding = self.processor(images[batch_idx], [question[batch_idx]] * len(images[batch_idx]), batch["words"][batch_idx], boxes=boxes, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                pred_answer_page = None
                for page_idx in range(len(document_encoding['input_ids'])):
                    # input_ids = document_encoding["input_ids"][page_idx].to(self.model.device)
                    # attention_mask = document_encoding["attention_mask"][page_idx].to(self.model.device)

                    page_inputs = {k: v[page_idx].unsqueeze(dim=0) for k, v in document_encoding.items()}
                    # Retrieval with logits is available only during inference and hence, the start and end indices are not used.
                    # start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
                    # end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

                    page_outputs = self.model(**page_inputs)
                    pred_answer, answer_conf = self.get_answer_from_model_output(page_inputs["input_ids"].unsqueeze(dim=0), page_outputs)

                    if answer_conf[0] > max_logits:
                        final_answer = pred_answer
                        pred_answer_page = page_idx
                        max_logits = answer_conf[0]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(final_answer if return_pred_answer else None)
                pred_answer_pages.append(pred_answer_page)

        else:
            boxes = [(bbox * 1000).astype(int) for bbox in batch['boxes']]
            encoding = self.processor(images, question, batch["words"], boxes=boxes, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            # encoding = self.processor(images, question, return_tensors="pt", padding=True, truncation=True).to(self.model.device)  # Apply OCR

            start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)
            outputs = self.model(**encoding, start_positions=start_pos, end_positions=end_pos)
            pred_answers, answ_confidence = self.get_answer_from_model_output(encoding.input_ids, outputs) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = [batch['context_page_corresp'][batch_idx][pred_start_idx] if len(batch['context_page_corresp'][batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]

            elif self.page_retrieval == 'none':
                pred_answer_pages = None

        if random.randint(0, 1000) == 0:
            for question_id, gt_answer, pred_answer in zip(batch['question_id'], answers, pred_answers):
                print("ID: {:5d}  GT: {:}  -  Pred: {:s}".format(question_id, gt_answer, pred_answer))

        return outputs, pred_answers, pred_answer_pages, answ_confidence


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

        answers = [self.processor.tokenizer.decode(input_tokens[batch_idx][start_idxs[batch_idx]: end_idxs[batch_idx]+1]).strip() for batch_idx in range(len(input_tokens))]
        # answ_confidence = np.mean([outputs.start_logits.softmax(dim=1).detach().cpu(), outputs.end_logits.softmax(dim=1).detach().cpu()])

        answ_confidence = model_utils.get_extractive_confidence(outputs)
        return answers, answ_confidence
