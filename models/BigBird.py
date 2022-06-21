import re, random

import torch
from transformers import BigBirdTokenizer, BigBirdTokenizerFast, BigBirdForQuestionAnswering


class BigBird:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = BigBirdTokenizerFast.from_pretrained(config['model_weights'])
        self.model = BigBirdForQuestionAnswering.from_pretrained(config['model_weights'])

    def get_start_end_idx(self, question, context, answers):

        # encodings = self.tokenizer.encode_plus([question, context], padding=True)

        pos_idx = []
        for batch_idx in range(self.batch_size):
            batch_pos_idxs = []
            for answer in answers[batch_idx]:
                # start_idx = context[batch_idx].find(answer)
                start_idxs = [m.start() for m in re.finditer(re.escape(answer), context[batch_idx])]

                for start_idx in start_idxs:
                    end_idx = start_idx + len(answer)

                    encodings = self.tokenizer.encode_plus([question[batch_idx], context[batch_idx]], padding=True,  truncation=True)

                    context_encodings = self.tokenizer.encode_plus(context[batch_idx])
                    start_positions_context = context_encodings.char_to_token(start_idx)
                    end_positions_context = context_encodings.char_to_token(end_idx - 1)

                    sep_idx = encodings['input_ids'].index(self.tokenizer.sep_token_id)
                    start_positions = start_positions_context + sep_idx + 1
                    end_positions = end_positions_context + sep_idx + 2

                    if self.tokenizer.decode(encodings['input_ids'][start_positions:end_positions]).strip() == answer:
                        batch_pos_idxs.append([start_positions, end_positions])
                        break

            if len(batch_pos_idxs) > 0:
                pos_idx.append(random.choice(batch_pos_idxs))
            else:
                pos_idx.append([0, 0])

        start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
        end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)
        return start_idxs, end_idxs

    def forward(self, question, context, answers, start_idxs=None, end_idxs=None, return_pred_answer=False):
        encoding = self.tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        start_pos, end_pos = self.get_start_end_idx(question, context, answers)
        # start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
        # end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

        outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
        # pred_start_idxs = torch.argmax(outputs.start_logits, axis=1)
        # pred_end_idxs = torch.argmax(outputs.end_logits, axis=1)
        pred_answers = self.get_answer_from_model_output(input_ids, outputs) if return_pred_answer else None

        return outputs, pred_answers

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

        return answers
