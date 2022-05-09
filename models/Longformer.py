
import torch
from transformers import LongformerTokenizer, LongformerForQuestionAnswering


class Longformer:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = LongformerTokenizer.from_pretrained(config['Model_weights'])
        self.model = LongformerForQuestionAnswering.from_pretrained(config['Model_weights'])

    def forward(self, question, context, start_idxs=None, end_idxs=None, return_pred_answer=False):
        encoding = self.tokenizer(question, context, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
        end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

        outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
        answers = self.get_answer_from_model_output(input_ids, outputs) if return_pred_answer else None

        return outputs, answers

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
