import os
import torch
import numpy as np
from numpy.distutils.system_info import blas_armpl_info
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
# from simpletransformers.question_answering import QuestionAnsweringModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class BertQA:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.model = AutoModelForQuestionAnswering.from_pretrained(config['model_weights'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower()

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        start_idxs = batch.get('start_idxs', None)
        end_idxs = batch.get('end_idxs', None)

        if self.page_retrieval == 'logits':
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            for batch_idx in range(len(context)):
                document_encoding = self.tokenizer([question[batch_idx]]*len(context[batch_idx]), context[batch_idx], return_tensors="pt", padding=True, truncation=True)

                max_logits = -999999
                answer_page = None
                document_outputs = None
                for page_idx in range(len(document_encoding['input_ids'])):
                    input_ids = document_encoding["input_ids"][page_idx].to(self.model.device)
                    attention_mask = document_encoding["attention_mask"][page_idx].to(self.model.device)

                    # Retrieval with logits is available only during inference and hence, the start and end indices are not used.
                    # start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
                    # end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

                    page_outputs = self.model(input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0))

                    start_logits_cnf = [page_outputs.start_logits[batch_ix, max_start_logits_idx.item()].item() for batch_ix, max_start_logits_idx in enumerate(page_outputs.start_logits.argmax(-1))][0]
                    end_logits_cnf = [page_outputs.end_logits[batch_ix, max_end_logits_idx.item()].item() for batch_ix, max_end_logits_idx in enumerate(page_outputs.end_logits.argmax(-1))][0]
                    page_logits = np.mean([start_logits_cnf, end_logits_cnf])

                    if page_logits > max_logits:
                        answer_page = page_idx
                        document_outputs = page_outputs
                        max_logits = page_logits

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(self.get_answer_from_model_output([document_encoding["input_ids"][answer_page]], document_outputs)[0] if return_pred_answer else None)
                pred_answer_pages.append(answer_page)

        else:
            encoding = self.tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)

            start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
            end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

            outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
            pred_answers = self.get_answer_from_model_output(input_ids, outputs) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = [batch['context_page_corresp'][batch_idx][pred_start_idx] if len(batch['context_page_corresp'][batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]

                # pred_answer_pages = []
                # for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1)):
                    # context_page_corresp = batch['context_page_corresp'][batch_idx]
                    # pred_answer_pages.append(context_page_corresp[pred_start_idx] if len(context_page_corresp) > 0 else 0)



        # start_logits_cnf = [outputs.start_logits[batch_ix, max_start_logits_idx.item()].item() for batch_ix, max_start_logits_idx in enumerate(outputs.start_logits.argmax(-1))]
        # end_logits_cnf = [outputs.end_logits[batch_ix, max_end_logits_idx.item()].item() for batch_ix, max_end_logits_idx in enumerate(outputs.end_logits.argmax(-1))]

        return outputs, pred_answers, pred_answer_pages

    def get_start_end_idx(self, encoding, context, answers):
        if False:
            pass
        else:
            start_pos, end_pos = -1, -1
        return start_pos, end_pos

    def get_answer_from_model_output(self, input_tokens, outputs):
        start_idxs = torch.argmax(outputs.start_logits, axis=1)
        end_idxs = torch.argmax(outputs.end_logits, axis=1)

        answers = []
        for elm_idx in range(len(input_tokens)):
            context_tokens = self.tokenizer.convert_ids_to_tokens(input_tokens[elm_idx].tolist())

            answer_tokens = context_tokens[start_idxs[elm_idx]: end_idxs[elm_idx] + 1]
            answer = self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(answer_tokens)
            )

            answer = answer.strip()  # remove space prepending space token
            answers.append(answer)

        return answers
