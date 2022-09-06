"""


import random

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm as BertLayerNorm
import torch.nn.functional as F


# from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from typing import Any, Dict

from pythia.models.m4c import M4C
from pythia.models.t5 import ModelBase
from pythia.common.registry import registry

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.file_utils import ModelOutput

# import transformers.models.t5.modeling_t5


class LayoutT5Config(T5Config):
    def __init__(self, max_2d_position_embeddings=1024,  **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        # following: T5 doesn't have this. copy from bertConfig
        self.type_vocab_size = 2  # x,y
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.model_type = 't5'
        self.initializer_range = 0.02  # same value as in BERT
        if kwargs.get('pm_max_length') is not None:
            self.pm_max_length = kwargs.get('pm_max_length')



class SpatialEmbeddings(nn.Module):
    """
"""
    Spatial embedding by summing x, y, w, h projected by nn.Embedding to hidden size.
    """
"""
    def __init__(self, config):
        super(SpatialEmbeddings, self).__init__()

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def forward(self, bbox):
        left_position_embeddings =  self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        embeddings = (
            left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MLP(nn.Module):
    """
""" Very simple multi-layer perceptron (also called FFN)"""
"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LayoutT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = SpatialEmbeddings(config)
        # embeddings.*.weights are populated via model_or_path2 and model_type2 (see argparse options)
        self.config = config
        #if self.embeddings.config.hidden_size != self.config.hidden_size:
        self.emb_matcher = MLP(self.embeddings.config.hidden_size, 0, self.config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            # inputs_embeds = self.shared(input_ids)
            token_embeds_t5 = self.shared(input_ids)  # read from default T5

            inputs_embeds_lm = self.embeddings(bbox)
            #if self.embeddings.config.hidden_size != self.config.hidden_size:
            inputs_embeds_lm = self.emb_matcher(inputs_embeds_lm)

            inputs_embeds = token_embeds_t5 + inputs_embeds_lm   # <=================== Use of bounding boxes.

            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
"""
        modified from: https://github.com/huggingface/transformers/blob/3ed5e97ba04ce9b24b4a7161ea74572598a4c480/src/transformers/generation_utils.py#L368-L373
        original implemention seems to forget to pass on other input args.
        the input_ids here is actually the decoder input id.
        """
"""
        if kwargs.get('encoder_outputs') is not None:
            return {'attention_mask': kwargs.get('attention_mask'),
                    'encoder_outputs': kwargs.get('encoder_outputs'),
                    'decoder_input_ids': input_ids,
                    }
        else:
            raise ValueError("Make sure that encoder_outputs is already computed when preapring inputs for generation. --y.x.")

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            token_embeds_t5 = self.shared(input_ids) # read from default T5
            inputs_embeds_lm = self.embeddings(model_kwargs['bbox'])
            inputs_embeds_lm = self.emb_matcher(inputs_embeds_lm)
            inputs_embeds = token_embeds_t5 + inputs_embeds_lm
            # load visua emb here for terminatorFCN
            # inputs_embeds_vis = self.visual_embeddings(model_kwargs['img'], model_kwargs['img_normed_bbox'])
            # inputs_embeds += inputs_embeds_vis

            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() \
                         if not argument.startswith("decoder_") \
                         and argument not in ['input_ids','attention_mask','inputs_embeds'] \
                         and argument not in ['bbox', 'labels'] \
            }  # ^^^ arguments specified explicitly below and arguments already used in self.embeddings.forward.
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids=None, attention_mask=model_kwargs['attention_mask'],
                                                                    inputs_embeds=inputs_embeds, **encoder_kwargs)
        return model_kwargs


class FUCKBase(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self._device = registry.get("current_device")
        self.pretraining = config.pretraining
        self.use_layout = config.use_layout

        self.max_ocr_tokens = config.classifier.ocr_max_num
        self.doc_pages = registry.get("config").dataset_attributes[registry.get("config").datasets].get('pages_per_document', None)  # Necessary if I want to use also LayoutT5 for doccvqa
        assert(self.doc_pages is None or self.doc_pages == 1), 'LayoutT5 not suitable for collections. You must set num pages to 1 '

        if self.doc_pages is not None:
            self.doc_pages += -1  # There is one page which its index is 0.

    # def build(self):
    #     # modules requiring custom learning rates (usually for finetuning)
    #     self.finetune_modules = []
    #
    #     # split model building into several components
    #     self._build_txt_encoding()
    #     self._build_obj_encoding()
    #     self._build_ocr_encoding()
    #     self._build_mmt()
    #     self._build_output()

    def _prepare_inputs(self, all_tokens, all_bbox, fwd_results, extra_key=''):
        assert extra_key == '' or extra_key == 'masked_'
        tokens = self.tokenizer.pad(all_tokens, return_tensors='pt')
        padded_boxes = torch.zeros((*tokens.input_ids.shape, 4), dtype=torch.long).to(self._device)
        for ix, elm in enumerate(all_bbox):
            padded_boxes[ix, :elm.shape[0], :] = elm

        fwd_results[extra_key+'input_ids'] = tokens.input_ids.to(self._device)
        fwd_results[extra_key+'attention_mask'] = tokens.attention_mask.to(self._device)
        fwd_results[extra_key+'bbox_coordinates'] = padded_boxes

    def _prepare_labels(self, all_labels, fwd_results, extra_key=''):
        assert extra_key == '' or extra_key == 'masked_'
        lm_labels = self.tokenizer.pad(all_labels, return_tensors='pt')
        lm_labels.input_ids[lm_labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
        fwd_results[extra_key+'labels'] = lm_labels.input_ids.to(self._device)

    def _build_output(self):
        pass


    def _forward_txt_encoding(self, sample_list, fwd_results):
        # pretraining = True

        bs = len(sample_list.image_id)
        all_bbox, all_tokens, all_labels, all_str_labels = [], [], [], []
        end_token_box = torch.LongTensor([0, 0, 0, 0]).to(self._device)
        q_bbox = torch.LongTensor([0, 0, 1000, 1000]).to(self._device)
        for batch_ix in range(bs):
            # Keep the same bounding box for each token that belongs to the same word.
            if not self.pretraining:
                q_tokens = self.tokenizer("question: {}  context: ".format(sample_list.question[batch_ix]))
                q_tokens_bbox = [[q_bbox] * len(q_tokens.input_ids[:-1])][0]

            if self.doc_pages is not None:
                ocr_tokens = [i for i in sample_list.context_tokens[batch_ix][self.doc_pages] if i != '<pad>']
            else:
                if len(sample_list.context_tokens[batch_ix]) == 1:
                    ocr_tokens = [i for i in sample_list.context_tokens[batch_ix][0] if i != '<pad>']
                else:
                    ocr_tokens = [i for i in sample_list.context_tokens[batch_ix] if i != '<pad>']

            ocr_tokens_batch = []
            ocr_tokens_bbox_batch = []
            for ocr_ix in range(len(ocr_tokens)):
                ocr_tok = self.tokenizer(ocr_tokens[ocr_ix])
                ocr_tokens_batch.extend(ocr_tok.input_ids[:-1])

                if self.doc_pages is not None:
                    ocr_tokens_bbox_batch.extend([sample_list.raw_ocr_coordinates[batch_ix, self.doc_pages, ocr_ix]] * len(ocr_tok.input_ids[:-1]))
                else:
                    if len(sample_list.raw_ocr_coordinates[batch_ix]) == 1:
                        ocr_tokens_bbox_batch.extend([sample_list.raw_ocr_coordinates[batch_ix, 0, ocr_ix]] * len(ocr_tok.input_ids[:-1]))
                    else:
                        ocr_tokens_bbox_batch.extend([sample_list.raw_ocr_coordinates[batch_ix, ocr_ix]] * len(ocr_tok.input_ids[:-1]))

            # Capping at MaxLength
            ocr_tokens_batch = ocr_tokens_batch[:self.max_ocr_tokens]
            ocr_tokens_bbox_batch = ocr_tokens_bbox_batch[:self.max_ocr_tokens]

            input_ids = q_tokens.input_ids[:-1] + ocr_tokens_batch + [self.tokenizer.eos_token_id]
            bbox = torch.stack((q_tokens_bbox + ocr_tokens_bbox_batch + [end_token_box]))

            if 'unique_answer_scores' in sample_list:
                idx = torch.multinomial(sample_list.unique_answer_scores[batch_ix], 1)
                answer = sample_list.unique_answer[batch_ix][idx.cpu()]
                labels = self.tokenizer(answer).input_ids
            else:
                labels = [0]

            all_labels.append({'input_ids': labels})
            all_tokens.append({'input_ids': input_ids})
            all_bbox.append(bbox)

        self._prepare_inputs(all_tokens, all_bbox, fwd_results)
        # self._prepare_inputs(masked_all_tokens, masked_all_bbox, fwd_results, extra_key='masked_')
        self._prepare_labels(all_labels, fwd_results)
        # self._prepare_labels(masked_all_labels, fwd_results, extra_key='masked_')

        if self.pretraining:
            fwd_results['str_labels'] = all_str_labels

    def _forward_mmt(self, sample_list, fwd_results):
        out = self.mmt.forward(input_ids=fwd_results['input_ids'],
                               bbox=fwd_results['bbox_coordinates'],
                               attention_mask=fwd_results['attention_mask'],
                               labels=fwd_results['labels'])

        fwd_results["losses"] = out[0]
        fwd_results["scores"] = out.logits

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        # self._forward_obj_encoding(sample_list, fwd_results)
        # self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_mmt(sample_list, fwd_results)

        # only keep scores in the forward pass results
        results = {"loss": fwd_results["losses"],
                   "scores": fwd_results["scores"],
                   "input_ids": fwd_results["input_ids"],
                   "bbox": fwd_results["bbox_coordinates"],
                   "attention_mask": fwd_results["attention_mask"],
                   "pred": (self.mmt, self.tokenizer)}

        if self.pretraining:
            results["str_labels"] = fwd_results["str_labels"]

        # results = {"loss": fwd_results["losses"],
        #            "scores": fwd_results["scores"],
        #            "input_ids": fwd_results["input_ids"],
        #            "bbox": fwd_results["bbox_coordinates"],
        #            "attention_mask": fwd_results["attention_mask"]}
        #
        # if not self.pretraining:
        #     results["pred"] = (self.mmt, self.tokenizer)
        # else:
        #     results["str_labels"] = fwd_results["str_labels"]

        return results


@registry.register_model('layout_t5_small')
class LayoutT5Small(FUCKBase):
    def __init__(self, config):
        super().__init__(config)
        self.layoutT5_config = LayoutT5Config.from_pretrained("t5-small")

    def _build_mmt(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.mmt = LayoutT5Model.from_pretrained("t5-small", config=self.layoutT5_config)


@registry.register_model('layout_t5_base')
class LayoutT5Base(FUCKBase):
    def __init__(self, config):
        super().__init__(config)
        self.layoutT5_config = LayoutT5Config.from_pretrained("t5-base")

    def _build_mmt(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.mmt = LayoutT5Model.from_pretrained("t5-base", config=self.layoutT5_config)
"""


"""
class LT5:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = LayoutT5.from_pretrained(config['model_weights'])

    def forward(self, question, context, answers, return_pred_answer=False):

        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)

        answers = [random.choice(answer) for answer in answers]
        labels = self.tokenizer(answers, return_tensors='pt', padding=True)
        labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
        labels = labels.input_ids.to(self.model.device)

        outputs = self.model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, labels=labels)
        pred_answers = self.get_answer_from_model_output(outputs) if return_pred_answer else None

        return outputs, pred_answers

    def get_answer_from_model_output(self, output):
        pred_answers = []
        batched_pred_tokens = output.logits.argmax(dim=-1)
        for pred_tokens in batched_pred_tokens:
            pred_answer = self.tokenizer.decode(pred_tokens)
            pred_answers.append(pred_answer.replace(self.tokenizer.eos_token, '').strip())

        return pred_answers
"""




import random, warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm as BertLayerNorm

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


import transformers.models.t5.modeling_t5


class LayoutT5Config(T5Config):
    def __init__(self, max_2d_position_embeddings=1024,  **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12


class SpatialEmbeddings(nn.Module):
    """
    Spatial embedding by summing x, y, w, h projected by nn.Embedding to hidden size.
    """

    def __init__(self, config):
        super(SpatialEmbeddings, self).__init__()

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def forward(self, bbox):
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])  # TODO Remove width and height to test how much important are they.
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])  # TODO Remove width and height to test how much important are they.

        embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
                + h_position_embeddings
                + w_position_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LT5(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.spatial_embeddings = SpatialEmbeddings(config)
        self.emb_matcher = MLP(self.spatial_embeddings.config.hidden_size, 0, self.config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            semantic_emb = self.shared(input_ids)  # read from default T5
            spatial_emb = self.emb_matcher(self.spatial_embeddings(bbox))
            inputs_embeds = semantic_emb + spatial_emb
            encoder_outputs = self.encoder(
                input_ids=None,  # Input IDs must be None because input embeds is provided.
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Proxy_LT5:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        config_x = LayoutT5Config.from_pretrained(config['model_weights'])
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = LT5.from_pretrained(config['model_weights'], config=config_x)
        self.page_retrieval = config['page_retrieval'].lower()

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        question_box = [0, 0, 1000, 1000]
        padding_box = [0, 0, 0, 0]
        eos_box = [0, 0, 0, 0]

        if self.page_retrieval == 'logits':
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)

            outputs = []
            pred_answers = []
            pred_answer_pages = []
            for batch_idx in range(len(context)):
                input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip([question[batch_idx]]*len(context[batch_idx]), context[batch_idx])]
                tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                answer_page = None
                best_answer = None
                pred_answer, logits = self.get_answer_from_model_output(tokens)
                for p_ix in range(len(input_text)):
                    if logits[p_ix] > max_logits:
                        max_logits = logits[p_ix]
                        answer_page = p_ix
                        best_answer = pred_answer[p_ix]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                # pred_answers.append(self.get_answer_from_model_output(document_outputs)[0] if return_pred_answer else None)
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)

        else:
            input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
            tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
            boxes = torch.zeros([len(context), tokens.input_ids.shape[-1], 4], dtype=torch.long)
            boxes[:] = torch.tensor(padding_box)
            for batch_idx in range(len(context)):
                question_boxes = torch.tensor([question_box] * len(self.tokenizer("question: {:s}  context: ".format(question[batch_idx])).input_ids[:-1]))

                if len(batch['words'][batch_idx]) >= 1:
                    context_boxes = torch.tensor(np.array([box for word, word_box in zip(batch['words'][batch_idx], batch['boxes'][batch_idx]) for box in [word_box]*len(self.tokenizer(word).input_ids[:-1])]))
                    context_boxes = context_boxes[:self.tokenizer.model_max_length - len(question_boxes) - 1]  # Remove boxes out of model max length.
                else:
                    context_boxes = torch.tensor(padding_box)

                boxes[batch_idx, :len(question_boxes)] = question_boxes

                try:
                    boxes[batch_idx, len(question_boxes): len(question_boxes)+len(context_boxes)] = context_boxes*1000
                except:
                    boxes[batch_idx, len(question_boxes): len(question_boxes)+len(context_boxes)] = context_boxes*1000

                boxes[batch_idx, len(question_boxes) + len(context_boxes)] = torch.tensor(eos_box)

            boxes = boxes.to(self.model.device)
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)

            outputs = self.model(input_ids=tokens.input_ids, bbox=boxes, attention_mask=tokens.attention_mask, labels=labels)
            # pred_answers = self.get_answer_from_model_output(outputs) if return_pred_answer else None
            pred_answers, logits = self.get_answer_from_model_output(tokens) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            else:
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages

    """
    def get_answer_from_model_output(self, output):
        pred_answers = []
        batched_pred_tokens = output.logits.argmax(dim=-1)
        for pred_tokens in batched_pred_tokens:
            pred_answer = self.tokenizer.decode(pred_tokens)
            pred_answers.append(pred_answer.replace(self.tokenizer.eos_token, '').strip())

        return pred_answers
    """

    def get_answer_from_model_output(self, input_tokens):
        bs = input_tokens.input_ids.shape[0]
        output = self.model.generate(**input_tokens, output_scores=True, return_dict_in_generate=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)

        """
        logits = np.zeros(len(output['scores'][0]))
        for seq_ix in range(len(output['scores'])):
            seq_logits = output['scores'][seq_ix].max(dim=-1)
            for batch_ix, token_id in enumerate(seq_logits.indices):
                logits[batch_ix] += seq_logits.values[batch_ix] if token_id not in [self.tokenizer.pad_token_id] else 0
        """

        all_logits = torch.stack(output.scores)
        best_logits = np.zeros(len(output['scores'][0]))
        for seq_ix in range(len(output['scores'])):
            for batch_ix in range(bs):
                token_id = output.sequences[batch_ix, seq_ix+1]
                best_logits[batch_ix] += all_logits[seq_ix, batch_ix, token_id] if token_id not in self.tokenizer.all_special_ids else 0

        return pred_answers, best_logits
