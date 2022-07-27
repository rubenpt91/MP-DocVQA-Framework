import random
import torch
import numpy as np
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
            outputs = []
            pred_answers = []
            pred_answer_pages = []
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
                document_outputs = None
                page_logits = []
                for page_idx in range(len(tokens.input_ids)):
                    page_outputs = self.model(input_ids=tokens.input_ids[page_idx].unsqueeze(dim=0), attention_mask=tokens.attention_mask[page_idx].unsqueeze(dim=0), labels=labels[batch_idx].unsqueeze(dim=0))

                    max_logits_idx = page_outputs.logits.argmax(dim=-1).squeeze(dim=0)

                    sequence_logits = []
                    for seq_idx, vocab_idx in enumerate(max_logits_idx):
                        if vocab_idx == self.tokenizer.eos_token_id:
                            break

                        #  elif vocab_idx not in [self.tokenizer.pad_token_id, 3]:   Avoid this weird _ token?
                        sequence_logits.append(page_outputs.logits.squeeze(dim=0)[seq_idx, vocab_idx.item()].item())

                    #  print("Sequence: {:d}".format(len(sequence_logits)))

                    if np.mean(sequence_logits) > max_logits:
                        answer_page = page_idx
                        document_outputs = page_outputs
                        max_logits = np.mean(sequence_logits)

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(self.get_answer_from_model_output(document_outputs)[0] if return_pred_answer else None)
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
            pred_answers = self.get_answer_from_model_output(tokens) if return_pred_answer else None

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
        output = self.model.generate(**input_tokens)
        pred_answers = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return pred_answers

