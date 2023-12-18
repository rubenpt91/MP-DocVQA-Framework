import random
import numpy as np

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PreTrainedModel
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
import models._model_utils as model_utils
import transformers.models.t5.modeling_t5


class HF_VT5(PreTrainedModel):

    def __init__(self, t5_config):
        super().__init__(t5_config)

        self.language_backbone = T5ForConditionalGeneration(t5_config)
        self.spatial_embedding = SpatialEmbeddings(t5_config) if t5_config.use_spatial_features else None
        self.visual_embedding = VisualEmbeddings(t5_config) if t5_config.use_visual_features else None


class VT5:
    def __init__(self, config):
        self.page_retrieval = config.page_retrieval.lower()
        self.max_source_length = getattr(config, 'max_source_length', 512)

        self.use_spatial_features = getattr(config, 'use_spatial_features', True)
        self.use_visual_features = getattr(config, 'use_visual_features', True)
        t5_config = CustomT5Config.from_pretrained(config.model_weights)
        t5_config.visual_module_config = config.visual_module
        t5_config.use_spatial_features = self.use_spatial_features
        t5_config.use_visual_features = self.use_visual_features

        self.tokenizer = T5Tokenizer.from_pretrained(config.model_weights)
        self.model = HF_VT5.from_pretrained(config.model_weights, config=t5_config)

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None):
        bs = len(words)
        # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, words)]
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0  # To become [0, 0, 0, 0] array.

        # Get input_ids, attention_mask and boxes.
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # Tokenize the word and ignore eos_token
                input_ids.extend(tokenized_word)
                input_boxes.extend([box]*len(tokenized_word))  # Repeat the box for each token corresponding to the word.

            batch_input_ids.append(input_ids[:self.max_source_length-1] + [self.tokenizer.eos_token_id])  # Append the eos_token at the end.
            batch_input_boxes.append(np.concatenate([input_boxes[:self.max_source_length-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
            longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_source_length)

        # Convert to tensors and pad. Actually, a pad tensor is created and it's filled with corresponding values.
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx][:len(batch_input_boxes[batch_idx])])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1

        # Send everything to GPU
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # Get semantic and spatial embeddings
        semantic_embedding = self.model.language_backbone.shared(tensor_input_ids)

        if self.use_visual_features:
            visual_embedding, visual_emb_mask = self.model.visual_embedding(images)

        if self.use_spatial_features:
            spatial_embedding = self.model.spatial_embedding(tensor_boxes) if self.use_spatial_features else None

            if self.use_visual_features:
                vis_boxes = self.model.visual_embedding.get_visual_boxes(num_pages=len(visual_embedding), scale=1000).to(semantic_embedding.device)  # Get visual boxes.
                vis_boxes_emb = self.model.spatial_embedding(vis_boxes.long())  # Get the spatial embeddings from the boxes.
                visual_embedding = visual_embedding + vis_boxes_emb  # Sum both visual-spatial representation.

        # Add spatial and semantic embeddings.
        input_embeds = torch.add(semantic_embedding, spatial_embedding) if self.use_spatial_features else semantic_embedding

        # Concatenate semantic + visual embeddings
        input_embeds = torch.cat([input_embeds, visual_embedding], dim=1) if self.use_visual_features else input_embeds
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1) if self.use_visual_features else tensor_attention_mask

        # Tokenize answers
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None

        return input_embeds, tensor_attention_mask, labels

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']
        bs = len(question)

        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []

            for batch_idx in range(bs):
                input_embeds, attention_mask, _ = self.prepare_inputs_for_vqa([question[batch_idx]]*num_pages[batch_idx], words[batch_idx], boxes[batch_idx], images[batch_idx])  # Answers are not considered. Logits set-up is made only for inference.
                pred_answer, logits = self.get_answer_from_model_output(input_embeds, attention_mask)
                # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip([question[batch_idx]]*len(context[batch_idx]), context[batch_idx])]
                # tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                answer_page = None
                best_answer = None
                for page_ix in range(len(input_embeds)):
                    if logits[page_ix] > max_logits:
                        max_logits = logits[page_ix]
                        answer_page = page_ix
                        best_answer = pred_answer[page_ix]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)
                pred_answers_conf.append(max_logits)

        else:
            input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(question, words, boxes, images, answers)

            outputs = self.model.language_backbone(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels) if labels is not None else None
            pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask) if return_pred_answer else None
            pred_answer_pages = batch['answer_page_idx'] if self.page_retrieval else None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.model.language_backbone.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)

        return pred_answers, pred_answers_conf
