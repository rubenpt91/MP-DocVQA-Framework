import random
import numpy as np

import torch
import torch.nn as nn
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering, LayoutLMv3Model
# from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering, LayoutLMv3Model
from transformers.modeling_outputs import BaseModelOutput, QuestionAnsweringModelOutput
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3ClassificationHead
from utils import correct_alignment
from torch.nn import CrossEntropyLoss


from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Model  # TODO Remove
# from transformers.models.layoutlmv3.processing_layoutlmv3 import LayoutLMv3Processor    # TODO Remove


class HiLayoutLMv3Model(LayoutLMv3Model):

    def __init__(self, config):
        super().__init__(config)

        self.page_token_embeddings = nn.Embedding(config.page_tokens, self.config.hidden_size)  # self.page_token_embeddings(torch.tensor([0])).shape

    # def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
    #     """
    #     Create the bounding boxes for the visual (patch) tokens.
    #     """
    #     visual_bbox_x = torch.div(torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc")
    #     visual_bbox_y = torch.div(torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc")
    #     visual_bbox = torch.stack(
    #         [
    #             visual_bbox_x[:-1].repeat(image_size[0], 1),
    #             visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
    #             visual_bbox_x[1:].repeat(image_size[0], 1),
    #             visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
    #         ],
    #         dim=-1,
    #     ).view(-1, 4)
    #
    #     cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])  # TODO - For each [PAGE] token.
    #     self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)
    #
    # def calculate_visual_bbox(self, device, dtype, batch_size):  # TODO - Maybe no need to included. When self.visual_bbox is changed. Do I need to change something here also?
    #     visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
    #     visual_bbox = visual_bbox.to(device).type(dtype)
    #     return visual_bbox
    #
    # def forward_image(self, pixel_values):
    #     embeddings = self.patch_embed(pixel_values)
    #
    #     # add [CLS] token
    #     batch_size, seq_len, _ = embeddings.size()
    #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    #     embeddings = torch.cat((cls_tokens, embeddings), dim=1)
    #
    #     # add position embeddings
    #     if self.pos_embed is not None:
    #         embeddings = embeddings + self.pos_embed  # TODO - Need to change positional embeddings.
    #
    #     embeddings = self.pos_drop(embeddings)
    #     embeddings = self.norm(embeddings)
    #
    #     return embeddings

    # @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            bbox=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            pixel_values=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            page_tokens_embeddings = self.page_token_embeddings(torch.arange(self.config.page_tokens).to(device)).unsqueeze(0).expand([batch_size, -1, -1])
            page_tokens_bbox = torch.tensor([[0 + 1, 0 + 1, 1000 - 1, 1000 - 1]]).repeat(batch_size, 10, 1).to(device)
            page_tokens_mask = torch.ones([batch_size, 10]).to(device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if pixel_values is not None:
            patch_height, patch_width = int(pixel_values.shape[2] / self.config.patch_size), int(pixel_values.shape[3] / self.config.patch_size)
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones((batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device)
            if attention_mask is not None:
                # attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
                attention_mask = torch.cat([page_tokens_mask, attention_mask, visual_attention_mask], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
            else:
                # attention_mask = visual_attention_mask
                attention_mask = torch.cat([page_tokens_mask, visual_attention_mask], dim=1)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(device, dtype=torch.long, batch_size=batch_size)
                    if bbox is not None:
                        # final_bbox = torch.cat([bbox, visual_bbox], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
                        final_bbox = torch.cat([page_tokens_bbox, bbox, visual_bbox], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
                    else:
                        # final_bbox = visual_bbox
                        final_bbox = torch.cat([page_tokens_bbox, visual_bbox], dim=1)

                page_position_ids = torch.arange(0, 10).repeat(batch_size, 1).to(device)
                visual_position_ids = torch.arange(0, visual_embeddings.shape[1], dtype=torch.long, device=device).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    # final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
                    final_position_ids = torch.cat([page_position_ids, position_ids, visual_position_ids], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
                else:
                    # final_position_ids = visual_position_ids
                    final_position_ids = torch.cat([page_position_ids, visual_position_ids], dim=1)

            if input_ids is not None or inputs_embeds is not None:
                # embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
                embedding_output = torch.cat([page_tokens_embeddings, embedding_output, visual_embeddings], dim=1)  # TODO - CAT HERE. Include the [PAGE] tokens also.
            else:
                # embedding_output = visual_embeddings
                embedding_output = torch.cat([page_tokens_embeddings, visual_embeddings], dim=1)

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        # TODO 354+197 = 551 + 10 561
        # TODO Why embedding output: [2, 561, 768], encoder_output: [2, 561, 768]

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class HiLayoutLMv3ClassificationHead(LayoutLMv3ClassificationHead):

    def __init__(self, config, pool_feature=False):
        super().__init__(config, pool_feature)


class HiLayoutLMv3(LayoutLMv3ForQuestionAnswering):

    def __init__(self, config, page_tokens):
        # config.vocab_size = 50270
        # config.vocab_size = 50300
        super().__init__(config)

        self.num_labels = config.num_labels
        self.page_tokens = page_tokens
        config.page_tokens = page_tokens

        self.layoutlmv3 = HiLayoutLMv3Model(config)
        self.qa_outputs = HiLayoutLMv3ClassificationHead(config, pool_feature=False)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        bbox=None,
        pixel_values=None,
        num_pages=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        page_embeddings = []
        page_embeddings_x = []
        for page_idx in range(max(num_pages)):
            outputs = self.layoutlmv3(
                input_ids[:, page_idx],
                attention_mask=attention_mask[:, page_idx],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                bbox=bbox[:, page_idx],
                pixel_values=pixel_values[:, page_idx],
            )

            sequence_output = outputs[0]
            page_embeddings.append(sequence_output[:, :self.page_tokens])
            page_embeddings_x.append(sequence_output)

        """
        for page_idx in range(max(num_pages)):
            hidden_states = encoder_outputs[0]
            page_embeddings.append(hidden_states[:, :self.page_tokens])

        document_embeddings = torch.cat(page_embeddings, dim=1)
        """
        # document_embeddings = torch.cat(page_embeddings, dim=1)
        document_embeddings_x = torch.cat(page_embeddings_x, dim=1)
        # logits = self.qa_outputs(document_embeddings_x)
        logits = self.qa_outputs(document_embeddings_x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Proxy_HiLayoutLMv3:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.processor = LayoutLMv3Processor.from_pretrained(config['model_weights'], apply_ocr=False)
        self.processor.tokenizer.add_tokens("[PAGE]")

        try:
            self.model = HiLayoutLMv3.from_pretrained(config['model_weights'], page_tokens=config['page_tokens'])
        except RuntimeError:
            self.model = HiLayoutLMv3.from_pretrained(config['model_weights'], page_tokens=config['page_tokens'], ignore_mismatched_sizes=True)

            x = torch.load('weights/layoutlmv3_oracle_mp-docvqa.ckpt/pytorch_model.bin')
            previous_embedding_size = x['layoutlmv3.embeddings.word_embeddings.weight'].shape[0]
            with torch.no_grad():
                self.model.layoutlmv3.embeddings.word_embeddings.weight[:previous_embedding_size] = x['layoutlmv3.embeddings.word_embeddings.weight']

        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.ignore_index = 9999  # 0

        self.page_tokens = config.get('page_tokens', 10)
        self.max_doc_pages = config.get('max_pages', 1)

        if config.get('freeze_encoder', False):
            for n, p in self.model.named_parameters():
                if not n.startswith('qa_outputs'):
                    p.requires_grad = False

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def forward(self, batch, return_pred_answer=False):

        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']
        num_pages = batch['num_pages']
        answer_pages_idx = batch['answer_page_idx']

        bs = len(batch['question_id'])

        if self.page_retrieval in ['oracle', None]:
            # images = batch['images']
            raise ValueError("Oracle set-up not available for Hi-LT5. Instead, specify 'max_pages: 1' in dataset config with 'page_retrieval: custom'.")

        elif self.page_retrieval in ['logits', 'concat']:
            raise ValueError("{:s} set-up not available for Hi-LayoutLMv3".format(self.page_retrieval.capitalize()))

        else:
            images = batch['images']

        boxes = []
        for doc_boxes in batch['boxes']:
            try:
                boxes.append([(box * 1000).astype(int) for box in doc_boxes])
            except AttributeError:
                boxes.append([(box * 1000).astype(int) for box in doc_boxes])


        longest_sequence = 0
        input_ids, attention_mask, bboxes, pixel_values = [], [], [], []
        documents_num_tokens = []
        for batch_idx in range(bs):
            """
            text = [["[PAGE]"] * self.page_tokens + page_words for page_words in batch["words"][batch_idx]]
            doc_boxes = [np.concatenate((np.tile(np.array([0, 0, 1000, 1000]), [self.page_tokens, 1]), page_boxes)) for page_boxes in boxes[batch_idx]]
            encoding = self.processor(images[batch_idx], [question[batch_idx]]*self.max_doc_pages, text, boxes=doc_boxes, return_tensors="pt", padding=True, truncation=True)
            """

            encoding = self.processor(images[batch_idx], [question[batch_idx]]*self.max_doc_pages, batch["words"][batch_idx], boxes=boxes[batch_idx], return_tensors="pt", padding=True, truncation=True)
            input_ids.append(encoding.input_ids)
            attention_mask.append(encoding.attention_mask)
            bboxes.append(encoding.bbox)
            pixel_values.append(encoding.pixel_values)
            documents_num_tokens.append(encoding.attention_mask.sum(dim=1))
            longest_sequence = max(longest_sequence, encoding.input_ids.shape[-1])

        all_input_ids = torch.zeros([bs, max(num_pages), longest_sequence], dtype=torch.long)
        all_attention_masks = torch.zeros([bs, max(num_pages), longest_sequence], dtype=torch.long)
        all_boxes = torch.zeros([bs, max(num_pages), longest_sequence, 4], dtype=torch.long)
        all_pixel_values = torch.zeros([bs, max(num_pages), 3, self.processor.feature_extractor.size, self.processor.feature_extractor.size], dtype=torch.float32)

        for batch_idx in range(bs):
            all_input_ids[batch_idx, :num_pages[batch_idx], :input_ids[batch_idx].shape[-1]] = input_ids[batch_idx][:num_pages[batch_idx]]
            all_attention_masks[batch_idx, :num_pages[batch_idx], :attention_mask[batch_idx].shape[-1]] = attention_mask[batch_idx][:num_pages[batch_idx]]
            all_boxes[batch_idx, :num_pages[batch_idx], :input_ids[batch_idx].shape[-1]] = bboxes[batch_idx][:num_pages[batch_idx]]
            all_pixel_values[batch_idx, :num_pages[batch_idx]] = pixel_values[batch_idx][:num_pages[batch_idx]]

        all_input_ids = all_input_ids.to(self.model.device)
        all_attention_masks = all_attention_masks.to(self.model.device)
        all_boxes = all_boxes.to(self.model.device)
        all_pixel_values = all_pixel_values.to(self.model.device)

        # Get answer start and end token position.
        # answer_page_words = [batch["words"][batch_idx][answer_pages_idx[batch_idx]] for batch_idx in range(bs)]
        # answer_page_boxes = [boxes[batch_idx][answer_pages_idx[batch_idx]] for batch_idx in range(bs)]
        # answer_page_imgs = [images[batch_idx][answer_pages_idx[batch_idx]] for batch_idx in range(bs)]
        # answer_page_encoding = self.processor(answer_page_imgs, question, answer_page_words, boxes=answer_page_boxes, return_tensors="pt", padding=True, truncation=True)

        padded_encoding, context_page_corresp = self.create_padded_encoding(input_ids, num_pages)
        start_pos, end_pos = self.get_start_end_idx(padded_encoding, answers)  # TODO --> Adapt to multipage
        # start_pos, end_pos = self.get_start_end_idx(answer_page_encoding, answers, documents_num_tokens, answer_pages_idx)  # TODO --> Adapt to multipage

        # outputs = self.model(**encoding, start_positions=start_pos, end_positions=end_pos)
        outputs = self.model(input_ids=all_input_ids,
                             attention_mask=all_attention_masks,
                             bbox=all_boxes,
                             pixel_values=all_pixel_values,
                             num_pages=num_pages,
                             start_positions=start_pos,
                             end_positions=end_pos)

        pred_answers, pred_answer_pages = self.get_answer_from_model_output(padded_encoding, outputs, context_page_corresp) if return_pred_answer else None

        # if self.page_retrieval == 'oracle':
        #     pred_answer_pages = batch['answer_page_idx']
        #
        # elif self.page_retrieval == 'concat':
        #     pred_answer_pages = [batch['context_page_corresp'][batch_idx][pred_start_idx] if len(batch['context_page_corresp'][batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]
        #
        # elif self.page_retrieval is None:
        #     pred_answer_pages = [-1 for _ in range(bs)]

        if random.randint(0, 1000) == 0:
            for question_id, gt_answer, pred_answer in zip(batch['question_id'], answers, pred_answers):
                print("ID: {:5d}  GT: {:}  -  Pred: {:s}".format(question_id, gt_answer, pred_answer))
        #
        #     for start_p, end_p, pred_start_p, pred_end_p in zip(start_pos, end_pos, outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1)):
        #         print("GT: {:d}-{:d} \t Pred: {:d}-{:d}".format(start_p.item(), end_p.item(), pred_start_p, pred_end_p))

        return outputs, pred_answers, pred_answer_pages

    def get_start_end_idx(self, input_ids, answers):
        pos_idx = []
        for batch_idx in range(len(input_ids)):
            answer_pos = []
            for answer in answers[batch_idx]:
                encoded_answer = [token for token in self.processor.tokenizer.encode([answer], boxes=[0, 0, 0, 0]) if token not in self.processor.tokenizer.all_special_ids]
                answer_tokens_length = len(encoded_answer)

                for token_pos in range(len(input_ids[batch_idx])):
                    if input_ids[batch_idx][token_pos: token_pos+answer_tokens_length].tolist() == encoded_answer:
                        answer_pos.append([token_pos, token_pos + answer_tokens_length-1])

            if len(answer_pos) == 0:
                pos_idx.append([self.ignore_index, self.ignore_index])

            else:
                answer_pos = random.choice(answer_pos)  # To add variability, pick a random correct span.
                pos_idx.append(answer_pos)

        start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
        end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)

        return start_idxs, end_idxs

    def create_padded_encoding(self, input_ids, num_pages):
        # Encoding that follows the internal LayoutLMv3 shape structure ([PAGE] (10) + Text (N) + Visual (197)) to encode the input tokens for the answer encoding and prediction.
        bs = len(input_ids)
        padded_encoding = torch.full([bs, max(num_pages) * 1024], self.processor.tokenizer.pad_token_id, dtype=torch.long)
        context_page_corresp = torch.full([bs, max(num_pages) * 1024], -1, dtype=torch.long)
        for batch_idx in range(bs):
            num_page_tokens = self.page_tokens
            for page_idx in range(num_pages[batch_idx]):
                page_tokens = input_ids[batch_idx][page_idx]
                padded_encoding[batch_idx, num_page_tokens: num_page_tokens + len(page_tokens)] = page_tokens
                context_page_corresp[batch_idx, num_page_tokens: num_page_tokens + len(page_tokens)] = page_idx
                num_page_tokens += len(page_tokens) + 197 + self.page_tokens  # TODO - CHECK if the <pad> inside the page_tokens itself worngly shifts the answer tokens.

        return padded_encoding, context_page_corresp

    # def get_start_end_idx(self, encoding, answers, documents_num_tokens, answer_pages_idx):
    #     pos_idx = []
    #     for batch_idx in range(len(answers)):
    #         answer_pos = []
    #         for answer in answers[batch_idx]:
    #             encoded_answer = [token for token in self.processor.tokenizer.encode([answer], boxes=[0, 0, 0, 0]) if token not in self.processor.tokenizer.all_special_ids]
    #             answer_tokens_length = len(encoded_answer)
    #
    #             for token_pos in range(len(encoding.input_ids[batch_idx])):
    #                 if encoding.input_ids[batch_idx][token_pos: token_pos+answer_tokens_length].tolist() == encoded_answer:
    #                     answer_pos.append([token_pos, token_pos + answer_tokens_length-1])
    #
    #         if len(answer_pos) == 0:
    #             pos_idx.append([self.ignore_index, self.ignore_index])
    #
    #         else:
    #             # Shift the answer position to match the corresponding page when taking into account all the pages and the visual embeddings (197).
    #             trailing_tokens = documents_num_tokens[batch_idx][:answer_pages_idx[batch_idx]].sum().item()
    #             trailing_tokens += (197 + 10) * answer_pages_idx[batch_idx]
    #
    #             answer_pos = random.choice(answer_pos)  # To add variability, pick a random correct span.
    #             # pos_idx.append(answer_pos)
    #             pos_idx.append([pos+trailing_tokens for pos in answer_pos])
    #
    #     start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
    #     end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)
    #
    #     return start_idxs, end_idxs

    def get_answer_from_model_output(self, input_tokens, outputs, context_page_corresp):
        bs = len(input_tokens)
        start_idxs = torch.argmax(outputs.start_logits, axis=1)
        end_idxs = torch.argmax(outputs.end_logits, axis=1)

        answers = [self.processor.tokenizer.decode(input_tokens[batch_idx][start_idxs[batch_idx]: end_idxs[batch_idx]+1]).strip() for batch_idx in range(bs)]
        answer_pages = [context_page_corresp[batch_idx][start_idxs[batch_idx]].item() for batch_idx in range(bs)]
        return answers, answer_pages
