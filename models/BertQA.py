import re, random
import torch
import torch.nn as nn

import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import models._model_utils as model_utils
from utils import correct_alignment


class BertQA:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.model = AutoModelForQuestionAnswering.from_pretrained(config['model_weights'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None

        self.max_sequence_length = config.get('max_sequence_length', 512)
        self.ignore_index = 9999  # 0

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def prepare_inputs_for_vqa(self, question, context, context_page_corresp, answers=None):
        encoding = self.tokenizer(question, context, padding=True, truncation=True, max_length=self.max_sequence_length, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        context_encoding = self.tokenizer.batch_encode_plus(context, padding=True, truncation=True, max_length=self.max_sequence_length)

        if answers is not None:
            start_pos, end_pos, context_page_token_correspondent = model_utils.get_start_end_idx('BertQA', encoding, context, context_encoding, answers, context_page_corresp, self.page_retrieval, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id, self.ignore_index, self.model.device)
        else:
            start_pos, end_pos, context_page_token_correspondent = None, None, None

        return input_ids, attention_mask, context_encoding, start_pos, end_pos, context_page_token_correspondent

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        if self.page_retrieval == 'logits':
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            answ_confidence = []
            for batch_idx in range(len(context)):
                document_encoding = self.tokenizer([question[batch_idx]]*len(context[batch_idx]), context[batch_idx], padding=True, truncation=True, max_length=self.max_sequence_length, return_tensors="pt",)

                max_logits = -999999
                answer_page = None
                document_outputs = None
                for page_idx in range(len(document_encoding['input_ids'])):
                    input_ids = document_encoding["input_ids"][page_idx].to(self.model.device)
                    attention_mask = document_encoding["attention_mask"][page_idx].to(self.model.device)
                    page_outputs = self.model(input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0))
                    pred_answer, answer_conf = self.get_answer_from_model_output(input_ids.unsqueeze(dim=0), page_outputs)

                    if answer_conf[0] > max_logits:
                        answer_page = page_idx
                        document_outputs = page_outputs
                        max_logits = answer_conf[0]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.extend(self.get_answer_from_model_output([document_encoding["input_ids"][answer_page]], document_outputs)[0] if return_pred_answer else None)
                pred_answer_pages.append(answer_page)
                answ_confidence.append(max_logits)

        else:
            input_ids, attention_mask, context_encoding, start_pos, end_pos, context_page_token_correspondent = self.prepare_inputs_for_vqa(question, context, batch['context_page_corresp'], answers)
            outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
            pred_answers, answ_confidence = self.get_answer_from_model_output(input_ids, outputs) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = [context_page_token_correspondent[batch_idx][pred_start_idx].item() if len(context_page_token_correspondent[batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]

            elif self.page_retrieval == 'none':
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, answ_confidence

    def get_answer_from_model_output(self, input_tokens, outputs):
        start_idxs = torch.argmax(outputs.start_logits, axis=1)
        end_idxs = torch.argmax(outputs.end_logits, axis=1)

        answers = []
        for batch_idx in range(len(input_tokens)):
            context_tokens = self.tokenizer.convert_ids_to_tokens(input_tokens[batch_idx].tolist())

            answer_tokens = context_tokens[start_idxs[batch_idx]: end_idxs[batch_idx] + 1]
            answer = self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(answer_tokens)
            )

            answer = answer.strip()  # remove space prepending space token
            answers.append(answer)

        answ_confidence = model_utils.get_extractive_confidence(outputs)

        return answers, answ_confidence
