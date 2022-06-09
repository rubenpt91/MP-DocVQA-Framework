import os
import torch
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
# from simpletransformers.question_answering import QuestionAnsweringModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class BertQA:

    def __init__(self, config):
        self.batch_size = config['training_parameters']['batch_size']
        self.model = AutoModelForQuestionAnswering.from_pretrained(config['Model_weights'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['Model_weights'])

    def forward(self, input_ids, attention_mask, start_pos, end_pos, return_pred_answer=False):
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        start_pos = start_pos.to(self.model.device)
        end_pos = end_pos.to(self.model.device)

        outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
        pred_answers = self.get_answer_from_model_output(input_ids, outputs) if return_pred_answer else None

        return outputs, pred_answers

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
