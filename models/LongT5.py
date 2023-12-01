import re, random
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import models._model_utils as model_utils


class LongT5:

    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_weights)
        self.model = LongT5ForConditionalGeneration.from_pretrained(config.model_weights)
        self.page_retrieval = config.page_retrieval.lower()

        self.max_sequence_length = getattr(config, 'max_sequence_length', 4096)
        self.ignore_index = 9999  # 0

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def prepare_inputs_for_vqa(self, question, context, answers=None):
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, padding=True, truncation=True, max_length=self.max_sequence_length, return_tensors='pt').to(self.model.device)

        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None

        return tokens.input_ids, tokens.attention_mask, labels

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []
            for batch_idx in range(len(context)):
                input_ids, attention_mask, _ = self.prepare_inputs_for_vqa([question[batch_idx]] * num_pages[batch_idx], context[batch_idx])
                pred_answer, logits = self.get_answer_from_model_output(input_ids, attention_mask) if return_pred_answer else None

                max_logits = -999999
                answer_page = None
                best_answer = None
                for p_ix in range(len(input_ids)):
                    if logits[p_ix] > max_logits:
                        max_logits = logits[p_ix]
                        answer_page = p_ix
                        best_answer = pred_answer[p_ix]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)
                pred_answers_conf.append(max_logits)

        else:
            input_ids, attention_mask, labels = self.prepare_inputs_for_vqa(question, context, answers)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) if labels is not None else None
            pred_answers, logits = self.get_answer_from_model_output(input_ids, attention_mask) if return_pred_answer else None
            pred_answer_pages = batch['answer_page_idx'] if self.page_retrieval == 'oracle' else None

        return outputs, pred_answers, pred_answer_pages, logits

    def get_answer_from_model_output(self, input_tokens, attention_mask):
        output = self.model.generate(input_tokens, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)

        return pred_answers, pred_answers_conf

