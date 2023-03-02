import random
import torch
import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration

# from pdb import set_trace

from tokenization_utils import update_tokenizer


def generative_confidence(output):
    batch_logits = torch.stack(output.scores, dim=1)[:, :-1, :]  # b x s x V and dropping EOS token
    decoder_output_confs = torch.amax(batch_logits.softmax(-1), 2).cpu().numpy()
    confidences = decoder_output_confs.prod(1)  # b
    return confidences


class T5:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.tokenizer = T5Tokenizer.from_pretrained(config["model_weights"])
        if 'use_2d' in config and config['use_2d']:
            from benchmarker.model.t5 import T52dForConditionalGeneration as t5_chosen_implementation
        else:
            t5_chosen_implementation = T5ForConditionalGeneration
        self.model = t5_chosen_implementation.from_pretrained(config["model_weights"])
        # assert self.model.encoder.block._modules['0'].layer._modules['0'].SelfAttention.relative_attention_bias
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = config.get(
                "generation_max_tokens", 20
            )  # fix for too short answers
        self.page_retrieval = (
            config["page_retrieval"].lower() if "page_retrieval" in config else None
        )
        self.tokenizer, self.model = update_tokenizer(self.tokenizer, self.model, config)

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def forward(self, batch):
        question = batch["questions"]
        context = batch["contexts"]
        answers = batch["answers"]

        if self.page_retrieval == "logits":
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            for batch_idx in range(len(context)):
                input_text = [
                    "question: {:s}  context: {:s}".format(q, c)
                    for q, c in zip(
                        [question[batch_idx]] * len(context[batch_idx]),
                        context[batch_idx],
                    )
                ]
                tokens = self.tokenizer(
                    input_text, return_tensors="pt", padding=True, truncation=True
                ).to(self.model.device)

                max_logits = -999999
                answer_page = None
                best_answer = None
                pred_answer, logits = self.get_answer_from_model_output(tokens)
                for p_ix in range(len(input_text)):
                    if logits[p_ix] > max_logits:
                        max_logits = logits[p_ix]
                        answer_page = p_ix
                        best_answer = pred_answer[p_ix]

                outputs.append(
                    None
                )  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)

        else:
            attention_mask, input_ids, labels, seg_data = self.prepare_data_for_forward_2d(answers, batch, context,
                                                                                           question)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                masked_lm_labels=labels,
                seg_data=seg_data
            )

        return outputs

    def prepare_data_for_forward_2d(self, answers, batch, context, question):
        input_text = [
            "question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)
        ]
        if 'boxes' in batch:
            bboxes = []
            tokens = []
            for batch_i in range(len(batch['questions'])):
                btokenized = self.tokenizer(
                    batch['words'][batch_i], return_tensors="pt", padding=True, truncation=True
                ).to(self.model.device)
                token_flat = btokenized.data['input_ids'].flatten()
                masked_flat = torch.masked_select(token_flat, token_flat > 1)
                boxes_flat = torch.Tensor(batch['boxes'][batch_i], device=self.model.device).repeat_interleave(
                    (btokenized.data['input_ids'] > 1).sum(1), dim=0)

                q_tokenized = self.tokenizer(
                    batch['questions'][batch_i], return_tensors="pt", padding=True, truncation=True
                ).to(self.model.device)
                q_tokenized = q_tokenized.data['input_ids'][0, :-1]
                q_len = q_tokenized.shape[0]
                q_bboxes = torch.stack(
                    [torch.arange(0, q_len,device=self.model.device) * 0.05,
                     torch.ones(q_len, device=self.model.device) * -0.05,
                     torch.arange(0, q_len, device=self.model.device) * 0.05,
                     torch.ones(q_len, device=self.model.device) * -0.05], dim=1)
                bboxes.append(torch.cat([q_bboxes, boxes_flat]))
                tokens.append(torch.cat([q_tokenized, masked_flat]))
            pass
            # TODO: stack and shorten the input
            max_input_len = -1
            if max_input_len > 0:
                tokens = [t[:max_input_len] for t in tokens]
                bboxes = [b[:max_input_len] for b in bboxes]
            input_ids = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
            bboxes = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True)
            attention_mask = (input_ids > 0).to(int)
            seg_data = {"tokens": {"bboxes": bboxes}}
            # seg_data[self.level]["token_map"]
            # seg_data[self.level]["bboxes"]

        else:
            tokens = self.tokenizer(
                input_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
        answers = [random.choice(answer) for answer in answers]
        labels = self.tokenizer(answers, return_tensors="pt", padding=True)
        labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
        labels = labels.input_ids.to(self.model.device)
        return attention_mask, input_ids, labels, seg_data

    def get_answer_from_model_output(self, input_ids, seg_data, return_confidence=False):
        bs = input_ids.shape[0]
        # output = self.model.generate(**input_tokens, output_scores=True, return_dict_in_generate=True)
        output = self.model.generate(  # without labels
            input_ids=input_ids, seg_data=seg_data, output_scores=True, return_dict_in_generate=True, output_attentions=True
        )
        pred_answers = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)

        all_logits = torch.stack(output.scores)
        best_logits = all_logits[1:, :, :].max(2)[0].sum(0)
        if return_confidence:
            return pred_answers, generative_confidence(output)

        return pred_answers, best_logits
