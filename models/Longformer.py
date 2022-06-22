import re, random

import torch
from transformers import LongformerTokenizer, LongformerTokenizerFast, LongformerForQuestionAnswering
from utils import correct_alignment

""" From https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/longformer_qa_training.ipynb#scrollTo=ON0le-uD4yiK
Longformer uses sliding-window local attention which scales linearly with sequence length. This is what allows longformer to handle longer sequences. For more details on how the sliding window attention works, please refer to the paper. Along with local attention longformer also allows you to use global attention for certain tokens. For QA task, all question tokens should have global attention.

The attention is configured using the attention_mask paramter of the forward method of LongformerForQuestionAnswering. Mask values are selected in [0, 1, 2]: 0 for no attention (padding tokens), 1 for local attention (a sliding window attention), 2 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).

As stated above all question tokens should be given gloabl attention. The LongformerForQuestionAnswering model handles this automatically for you. To allow it to do that

    The input sequence must have three sep tokens, i.e the sequence should be encoded like this <s> question</s></s> context</s>. If you encode the question and answer as a input pair, then the tokenizer already takes care of that, you shouldn't worry about it.
    input_ids should always be a batch of examples.
"""
class Longformer:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = LongformerTokenizerFast.from_pretrained(config['model_weights'])
        self.model = LongformerForQuestionAnswering.from_pretrained(config['model_weights'])

    """
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

                    encodings = self.tokenizer.encode_plus([question[batch_idx], context[batch_idx]], padding=True)

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
    """

    def get_start_end_idx(self, encoding, context, answers):

        # encodings = self.tokenizer.encode_plus([question, context], padding=True)

        pos_idx = []
        for batch_idx in range(self.batch_size):
            batch_pos_idxs = []
            for answer in answers[batch_idx]:
                start_idxs = [m.start() for m in re.finditer(re.escape(answer), context[batch_idx])]

                for start_idx in start_idxs:
                    end_idx = start_idx + len(answer)
                    start_idx, end_idx = correct_alignment(context[batch_idx], answer, start_idx, end_idx)

                    if start_idx is not None:
                        batch_pos_idxs.append([start_idx, end_idx])
                        break

            if len(batch_pos_idxs) > 0:
                start_idx, end_idx = random.choice(batch_pos_idxs)

                context_encodings = self.tokenizer.encode_plus(context[batch_idx], padding=True, truncation=True)
                start_positions_context = context_encodings.char_to_token(start_idx)
                end_positions_context = context_encodings.char_to_token(end_idx - 1)

                # here we will compute the start and end position of the answer in the whole example
                # as the example is encoded like this <s> question</s></s> context</s>
                # and we know the position of the answer in the context
                # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
                # this will give us the position of the answer span in whole example
                sep_idx = encoding['input_ids'][batch_idx].tolist().index(self.tokenizer.sep_token_id)
                start_position = start_positions_context + sep_idx + 1
                end_position = end_positions_context + sep_idx + 1

                if end_position > 512:
                    start_position, end_position = 0, 0

                pos_idx.append([start_position, end_position])

            else:
                pos_idx.append([0, 0])

        start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
        end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)

        return start_idxs, end_idxs

    def forward(self, question, context, answers, start_idxs=None, end_idxs=None, return_pred_answer=False):
        encoding = self.tokenizer(question, context, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)
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
