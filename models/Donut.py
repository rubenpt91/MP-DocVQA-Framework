import random

import torch
import torch.nn as nn
from transformers import DonutProcessor, VisionEncoderDecoderModel
import models._model_utils as model_utils
import transformers.models.donut.modeling_donut_swin
import transformers.models.donut.processing_donut


class Donut:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.processor = DonutProcessor.from_pretrained(config['model_weights'])
        self.model = VisionEncoderDecoderModel.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.ignore_id = -100  # 0

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def prepare_inputs_for_vqa(self, question, images, answers=None):

        pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(self.model.device)

        if answers is not None:
            task_prompt_with_labels = ["<s_docvqa><s_question>{:s}</s_question><s_answer>{:s}</s_answer>".format(q, random.choice(a)) for q, a in zip(question, answers)]
            decoder_encoding = self.processor.tokenizer(task_prompt_with_labels, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True)
            labels = decoder_encoding.input_ids
            labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # Model doesn't need to predict pad token

            # Model doesn't need to predict prompt (for VQA)
            prompt_end_token_id = torch.LongTensor(self.processor.tokenizer.encode("</s_question>", add_special_tokens=False))
            end_prompt_idx = torch.nonzero(labels == torch.LongTensor(prompt_end_token_id))[:, -1]
            for batch_idx in range(len(labels)):
                labels[batch_idx, :end_prompt_idx[batch_idx] + 1] = self.ignore_id

            # decoder_encoding.input_ids = decoder_encoding  # TODO: Check and probably remove...
            # decoder_encoding.input_ids = decoder_encoding.input_ids.to(self.model.device)
            decoder_encoding.to(self.model.device)

        else:
            task_prompt = ["<s_docvqa><s_question>{:s}</s_question><s_answer>".format(q) for q in question]
            decoder_encoding = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            # decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=512)

            labels = None

        return decoder_encoding.input_ids, decoder_encoding.attention_mask, pixel_values, labels

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        images = batch['images']
        answers = batch['answers']

        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []
            for batch_idx in range(len(question)):
                input_ids, attention_mask, image_pixel_values, _ = self.prepare_inputs_for_vqa([question[batch_idx]]*num_pages[batch_idx], images[batch_idx])
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
            input_ids, attention_mask, image_pixel_values, labels = self.prepare_inputs_for_vqa(question, images, answers)
            # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            outputs = self.model(image_pixel_values, decoder_input_ids=input_ids[:, :-1], labels=labels[:, 1:])

            input_ids, attention_mask, image_pixel_values, labels = self.prepare_inputs_for_vqa(question, images)
            pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_ids, image_pixel_values) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            else:
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    def get_answer_from_model_output(self, input_ids, image_pixels):
        outputs = self.model.generate(
            image_pixels,
            decoder_input_ids=input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)
        sequence = [seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "") for seq in sequence]
        pred_answers = [self.processor.token2json(seq)['answer'] for seq in sequence]
        pred_answers_conf = [None for _ in range(len(input_ids))]

        return pred_answers, pred_answers_conf
