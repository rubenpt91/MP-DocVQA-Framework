import random
import torch
import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers.models.t5.modeling_t5

# from pdb import set_trace

from tokenization_utils import update_tokenizer


class T5:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.tokenizer = T5Tokenizer.from_pretrained(config["model_weights"])
        self.model = T5ForConditionalGeneration.from_pretrained(config["model_weights"])
        self.model.generation_config.max_length = config.get(
            "generation_max_tokens", 20
        )  # fix for short answers
        self.page_retrieval = (
            config["page_retrieval"].lower() if "page_retrieval" in config else None
        )
        self.tokenizer, self.model = update_tokenizer(
            self.tokenizer, self.model, config
        )

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def forward(self, batch, return_pred_answer=False, return_confidence=False):
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
            input_text = [
                "question: {:s}  context: {:s}".format(q, c)
                for q, c in zip(question, context)
            ]
            tokens = self.tokenizer(
                input_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors="pt", padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)

            outputs = self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                labels=labels,
            )
            pred_answers, logits = (
                self.get_answer_from_model_output(tokens)
                if return_pred_answer
                else None
            )
            if self.page_retrieval == "oracle":
                pred_answer_pages = batch["answer_page_idx"]

            else:
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages

    def get_answer_from_model_output(self, input_tokens, return_confidence=False):
        bs = input_tokens.input_ids.shape[0]
        # output = self.model.generate(**input_tokens, output_scores=True, return_dict_in_generate=True)
        output = self.model.generate(
            **input_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=True
        )
        pred_answers = self.tokenizer.batch_decode(
            output["sequences"], skip_special_tokens=True
        )

        all_logits = torch.stack(output.scores)
        best_logits = np.zeros(len(output["scores"][0]))
        for seq_ix in range(len(output["scores"])):
            for batch_ix in range(bs):
                token_id = output.sequences[batch_ix, seq_ix + 1]
                best_logits[batch_ix] += (
                    all_logits[seq_ix, batch_ix, token_id]
                    if token_id not in self.tokenizer.all_special_ids
                    else 0
                )

        return pred_answers, best_logits
