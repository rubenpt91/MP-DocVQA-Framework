import random
import torch
import numpy as np
from numpy.ma.core import outer
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower()

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        if self.page_retrieval == 'logits':
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)

            outputs = []
            pred_answers = []
            pred_answer_pages = []
            for batch_idx in range(len(context)):
                input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip([question[batch_idx]]*len(context[batch_idx]), context[batch_idx])]
                tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                answer_page = None
                best_answer = None
                pred_answer, logits = self.get_answer_from_model_output(tokens)
                for p_ix in range(len(input_text)):
                    if logits[p_ix] > max_logits:
                        max_logits = logits[p_ix]
                        answer_page = p_ix
                        best_answer = pred_answer[p_ix]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                # pred_answers.append(self.get_answer_from_model_output(document_outputs)[0] if return_pred_answer else None)
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)

        else:
            input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
            tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)

            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)

            outputs = self.model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, labels=labels)
            # pred_answers = self.get_answer_from_model_output(outputs) if return_pred_answer else None
            pred_answers, logits = self.get_answer_from_model_output(tokens) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            else:
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages

    """
    def get_answer_from_model_output(self, output):
        pred_answers = []
        batched_pred_tokens = output.logits.argmax(dim=-1)
        for pred_tokens in batched_pred_tokens:
            pred_answer = self.tokenizer.decode(pred_tokens)
            pred_answers.append(pred_answer.replace(self.tokenizer.eos_token, '').strip())

        return pred_answers
    """

    def get_answer_from_model_output(self, input_tokens):
        bs = input_tokens.input_ids.shape[0]
        output = self.model.generate(**input_tokens, output_scores=True, return_dict_in_generate=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)

        """
        logits = np.zeros(len(output['scores'][0]))
        for seq_ix in range(len(output['scores'])):
            seq_logits = output['scores'][seq_ix].max(dim=-1)
            for batch_ix, token_id in enumerate(seq_logits.indices):
                logits[batch_ix] += seq_logits.values[batch_ix] if token_id not in [self.tokenizer.pad_token_id] else 0
        """

        all_logits = torch.stack(output.scores)
        best_logits = np.zeros(len(output['scores'][0]))
        for seq_ix in range(len(output['scores'])):
            for batch_ix in range(bs):
                token_id = output.sequences[batch_ix, seq_ix+1]
                best_logits[batch_ix] += all_logits[seq_ix, batch_ix, token_id] if token_id not in self.tokenizer.all_special_ids else 0

        return pred_answers, best_logits

