import re, random
import numpy as np

import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForQuestionAnswering
from PIL import Image
from utils import correct_alignment

from transformers.models.layoutlmv2.modeling_layoutlmv2 import LayoutLMv2Model
from transformers.models.layoutlmv2.processing_layoutlmv2 import LayoutLMv2Processor

class LayoutLMv2:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.processor = LayoutLMv2Processor.from_pretrained(config['model_weights'])
        self.model = LayoutLMv2ForQuestionAnswering.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower()

    def forward(self, batch, return_pred_answer=False):

        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        if self.page_retrieval == 'logits':
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            for batch_idx in range(len(context)):
                images = [Image.open(img_path).convert("RGB") for img_path in batch['image_names'][batch_idx]]
                document_encoding = self.processor(images, [question[batch_idx]] * len(context[batch_idx]), return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                answer_page = None
                document_outputs = None
                for page_idx in range(len(document_encoding['input_ids'])):
                    input_ids = document_encoding["input_ids"][page_idx].to(self.model.device)
                    attention_mask = document_encoding["attention_mask"][page_idx].to(self.model.device)

                    page_inputs = {k: v[page_idx].unsqueeze(dim=0) for k, v in document_encoding.items()}
                    # Retrieval with logits is available only during inference and hence, the start and end indices are not used.
                    # start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
                    # end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

                    page_outputs = self.model(**page_inputs)

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
            # encoding = self.tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)
            # input_ids = encoding["input_ids"].to(self.model.device)
            # attention_mask = encoding["attention_mask"].to(self.model.device)

            if self.page_retrieval == 'oracle':
                images = [Image.open(img_path).convert("RGB") for img_path in batch['image_names']]

            elif self.page_retrieval == 'concat':
                raise NotImplementedError
                images = [Image.open(img_path).convert("RGB") for img_path in batch['image_names']]

            encoding = self.processor(images, question, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

            # start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)
            outputs = self.model(**encoding, start_positions=None, end_positions=None)
            pred_answers = self.get_answer_from_model_output(encoding.input_ids, outputs) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = [batch['context_page_corresp'][batch_idx][pred_start_idx] if len(batch['context_page_corresp'][batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]

        if random.randint(0, 1000) == 0:
            print(batch['question_id'])
            for gt_answer, pred_answer in zip(answers, pred_answers):
                print(gt_answer, pred_answer)

            for start_p, end_p, pred_start_p, pred_end_p in zip(start_pos, end_pos, outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1)):
                print("GT: {:d}-{:d} \t Pred: {:d}-{:d}".format(start_p.item(), end_p.item(), pred_start_p, pred_end_p))

        return outputs, pred_answers, pred_answer_pages

    def get_start_end_idx(self, encoding, context, answers):
        pos_idx = []
        for batch_idx in range(len(context)):
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

                context_encodings = self.processor.encode_plus(context[batch_idx], padding=True, truncation=True)
                start_positions_context = context_encodings.char_to_token(start_idx)
                end_positions_context = context_encodings.char_to_token(end_idx - 1)

                # here we will compute the start and end position of the answer in the whole example
                # as the example is encoded like this <s> question</s></s> context</s>
                # and we know the position of the answer in the context
                # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
                # this will give us the position of the answer span in whole example
                sep_idx = encoding['input_ids'][batch_idx].tolist().index(self.processor.sep_token_id)

                if start_positions_context is not None and end_positions_context is not None:
                    start_position = start_positions_context + sep_idx + 1
                    end_position = end_positions_context + sep_idx + 1

                    if end_position > 512:
                        start_position, end_position = 0, 0

                else:
                    start_position, end_position = 0, 0

                pos_idx.append([start_position, end_position])

            else:
                pos_idx.append([0, 0])

        start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
        end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)

        return start_idxs, end_idxs

    def get_answer_from_model_output(self, input_tokens, outputs):
        start_idxs = torch.argmax(outputs.start_logits, axis=1)
        end_idxs = torch.argmax(outputs.end_logits, axis=1)

        answers = [self.processor.tokenizer.decode(input_tokens[batch_idx][start_idxs[batch_idx]: end_idxs[batch_idx]]).strip() for batch_idx in range(len(input_tokens))]

        return answers
