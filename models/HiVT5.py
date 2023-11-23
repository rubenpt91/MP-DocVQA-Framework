import random, warnings
import numpy as np
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput, BaseModelOutput
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings, RetrievalModule

import transformers.models.t5.modeling_t5
""" START - FOR GREEDY SEARCH """
# import torch.distributed as dist
# from typing import Union
# from transformers.generation_utils import GreedySearchOutput, GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
# from transformers.generation_logits_process import LogitsProcessorList
# from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
""" END - FOR GREEDY SEARCH """

""" START - FOR GREEDY SEARCH """
import torch.distributed as dist
from typing import Union
from transformers.generation.utils import GreedySearchOutput, GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
""" END - FOR GREEDY SEARCH """


class HiVT5(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.spatial_embeddings = SpatialEmbeddings(config)
        self.visual_embeddings = VisualEmbeddings(config)

        self.retrieval_module = RetrievalModule(config)

        self.use_spatial_features = config.use_spatial_features
        self.use_visual_features = config.use_visual_features
        self.page_tokens = config.page_tokens
        self.max_doc_pages = config.max_doc_pages

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        extra_kwargs_to_be_removed = ['bbox', 'images', 'attention_mask', 'num_pages']
        encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any(argument.startswith(p) for p in irrelevant_prefix + extra_kwargs_to_be_removed)}

        # 2.2 replace input ids by the hierarchical layout-aware input embeddings
        bbox = model_kwargs['bbox']
        images = model_kwargs['images']
        num_pages = model_kwargs['num_pages']
        attention_mask = model_kwargs['attention_mask']

        page_embeddings = []
        page_encoder_attentions = []
        for p_idx in range(max(model_kwargs['num_pages'])):

            if self.use_spatial_features:
                semantic_emb = self.shared(inputs_tensor[:, p_idx])  # read from default T5
                spatial_emb = self.spatial_embeddings(bbox[:, p_idx])
                text_embeds = semantic_emb + spatial_emb
            else:
                text_embeds = self.shared(inputs_tensor[:, p_idx])  # read from default T5

            page_idx_mask = [batch_idx for batch_idx in range(len(num_pages)) if num_pages[batch_idx] >= p_idx + 1]

            if self.use_visual_features:
                visual_emb, vis_mask = self.visual_embeddings([doc_images[p_idx] for doc_images in images], page_idx_mask=page_idx_mask)
                inputs_embeds = torch.cat((text_embeds, visual_emb), dim=1)
                inputs_mask = torch.cat((attention_mask[:, p_idx], vis_mask), dim=1)

            else:
                inputs_embeds = text_embeds
                inputs_mask = attention_mask[:, p_idx]

            encoder_outputs = encoder(
                input_ids=None,
                attention_mask=inputs_mask,
                inputs_embeds=inputs_embeds,
                **encoder_kwargs
            )

            hidden_states = encoder_outputs[0]
            page_embeddings.append(hidden_states[:, :self.page_tokens])

            if model_kwargs['output_attentions']:
                page_encoder_attentions.append(encoder_outputs.attentions)

        document_embeddings = torch.cat(page_embeddings, dim=1)

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = None
        # model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        output_model_kwargs = {'last_hidden_state': document_embeddings}

        if model_kwargs['output_attentions']:
            output_model_kwargs['attentions'] = page_encoder_attentions

        model_kwargs["encoder_outputs"]: ModelOutput = ModelOutput(output_model_kwargs)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "num_pages": kwargs.get('num_pages'),
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.GreedySearchDecoderOnlyOutput`], [`~generation_utils.GreedySearchEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                greedy_search_output = GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )

                greedy_search_output.ret_logits = outputs.ret_logits
                return greedy_search_output
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


    def forward(
        self,
        input_ids=None,
        bbox=None,
        images=None,
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
        num_pages=None,
        answer_page_idx=None,
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
            page_embeddings = []
            page_encoder_attentions = []
            # for page_idx in range(self.max_doc_pages):
            for page_idx in range(max(num_pages)):
                semantic_emb = self.shared(input_ids[:, page_idx])  # read from default T5
                # spatial_emb = self.spatial_emb_matcher(self.spatial_embeddings(bbox[:, page_idx]))
                spatial_emb = self.spatial_embeddings(bbox[:, page_idx])
                text_embeds = semantic_emb + spatial_emb

                page_idx_mask = [batch_idx for batch_idx in range(len(num_pages)) if num_pages[batch_idx] >= page_idx+1]
                visual_emb, vis_mask = self.visual_embeddings([doc_images[page_idx] for doc_images in images], page_idx_mask=page_idx_mask)

                # TODO: Try with / without.
                vis_boxes = self.visual_embeddings.get_visual_boxes(num_pages=len(visual_emb), scale=1000)  # Get visual boxes.
                vis_boxes_emb = self.spatial_embeddings(vis_boxes.long().to(self.device))  # Get the spatial embeddings from the boxes.
                visual_emb = visual_emb + vis_boxes_emb  # Sum both visual-spatial representation.

                inputs_embeds = torch.cat((text_embeds, visual_emb), dim=1)
                inputs_mask = torch.cat((attention_mask[:, page_idx], vis_mask), dim=1)
                encoder_outputs = self.encoder(
                    input_ids=None,  # Input IDs must be None because input embeds is provided.
                    attention_mask=inputs_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                # Keep only [PAGE] token representation.
                hidden_states = encoder_outputs[0]
                page_embeddings.append(hidden_states[:, :self.page_tokens])

                if output_attentions:
                    page_encoder_attentions.append(encoder_outputs.attentions)

            document_embeddings = torch.cat(page_embeddings, dim=1)

            # attention_mask = torch.zeros([hidden_states.shape[0], self.num_doc_cls_tokens * self.doc_pages]).to(document_embeddings.device)  # Pages, hidden size. Make use of all information of the document embedding
            attention_mask = torch.zeros([hidden_states.shape[0], self.page_tokens * max(num_pages)]).to(document_embeddings.device)  # Pages, hidden size. Make use of all information of the document embedding
            for bs_idx in range(len(hidden_states)):
                attention_mask[bs_idx, :min(num_pages[bs_idx], self.max_doc_pages) * self.page_tokens] = 1

            attention_mask = attention_mask.to(document_embeddings.device)

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):  # EncoderOutputs is True when comes from _prepare_encoder_decoder_kwargs_for_generation, during .generation function.
            page_encoder_attentions = encoder_outputs['attentions'] if output_attentions else None
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

            hidden_states = encoder_outputs[0]  # TODO - This should be replaced by document embeddings
            document_embeddings = hidden_states

            attention_mask = torch.zeros([hidden_states.shape[0], self.page_tokens * max(num_pages)])
            for bs_idx in range(len(hidden_states)):
                attention_mask[bs_idx, : min(num_pages[bs_idx], max(num_pages)) * self.page_tokens] = 1

            attention_mask = attention_mask.to(document_embeddings.device)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens should be given as an input
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
            # encoder_hidden_states=hidden_states,
            encoder_hidden_states=document_embeddings,  # Previous 'hidden states' in original T5
            encoder_attention_mask=attention_mask,  # Multi-page attention mask.
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

        loss = None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        ret_loss, ret_logits = self.retrieval_module(document_embeddings, answer_page_idx)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        model_output = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=page_encoder_attentions if output_attentions else None,
            # encoder_attentions=encoder_outputs.attentions,
        )

        model_output.ret_logits = ret_logits
        model_output.ret_loss = ret_loss

        return model_output


class Proxy_HiVT5:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.page_tokens = config.get('page_tokens', 10)
        self.max_doc_pages = config.get('max_pages', 1)

        config_x = CustomT5Config.from_pretrained(config['model_weights'])
        config_x.page_tokens = self.page_tokens
        config_x.max_doc_pages = self.max_doc_pages
        config_x.use_spatial_features = config.get('use_spatial_features', True)
        config_x.page_retrieval_config = config['retrieval_module']
        config_x.use_visual_features = config.get('use_visual_features', True)
        config_x.visual_module_config = config['visual_module']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        # self.tokenizer.add_tokens("[PAGE]")  # Single representation
        [self.tokenizer.add_tokens(f"[PAGE_{i}]") for i in range(self.page_tokens)]  # Multiple representation
        self.max_text_tokens = config.get('max_text_tokens', self.tokenizer.model_max_length)
        self.decoding_max_length = config_x.max_length
        # [self.tokenizer.add_tokens("[PAGE_{:d}]".format(p)) for p in range(self.num_doc_cls_tokens)]  # Different representation

        # Whenever the number of [PAGE] tokens or Max pages per document changes, the architecture also changes and therefore, it needs to be fine-tuned.
        self.model = HiVT5.from_pretrained(config['model_weights'], config=config_x, ignore_mismatched_sizes=True)

        if config.get('freeze_encoder', False):
            for n, p in self.model.named_parameters():
                if not (n.startswith('decoder') or n.startswith('retrieval_module')):
                    p.requires_grad = False

        self.device = config['device']

    def parallelize(self):
        self.model = nn.DataParallel(self.model)
        # self.model = nn.parallel.DistributedDataParallel(self.model)  # TODO: Apparently faster, but needs some specific handling...?

    def prepare_vqa_input_ids(self, batch):
        bs = len(batch['question_id'])
        num_pages = batch['num_pages']

        question = batch['questions']
        context = batch['contexts']
        page_token_box = [0, 0, 1000, 1000]
        question_box = [0, 0, 1000, 1000]
        padding_box = [0, 0, 0, 0]
        eos_box = [0, 0, 0, 0]

        longest_sequence = 0
        all_input_ids = torch.zeros([bs, max(num_pages), self.max_text_tokens], dtype=torch.long)
        all_attention_masks = torch.zeros([bs, max(num_pages), self.max_text_tokens], dtype=torch.long)
        all_boxes = torch.zeros([bs, max(num_pages), self.max_text_tokens, 4], dtype=torch.long)

        for batch_idx in range(bs):

            # Do directly the three loops in once. Then, trim the tensors to the: 1 longest sequence or max_seq_length.
            page_tokens = ''.join([f"[PAGE_{i}]" for i in range(self.page_tokens)])  # Multiple representation
            input_text = [f"{page_tokens}: question: {question[batch_idx]}  context: {c}" for c in context[batch_idx]]
            tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
            all_input_ids[batch_idx, :num_pages[batch_idx], :tokens.input_ids.shape[-1]] = tokens.input_ids
            all_attention_masks[batch_idx, :num_pages[batch_idx], :tokens.attention_mask.shape[-1]] = tokens.attention_mask

            longest_sequence = max(longest_sequence, tokens.input_ids.shape[-1])

            length_pretext = len(self.tokenizer("question: {:s}  context: ".format(question[batch_idx])).input_ids[:-1])
            pretext_boxes = torch.tensor([question_box] * length_pretext)
            for page_idx in range(num_pages[batch_idx]):
                if len(batch['boxes'][batch_idx][page_idx]) >= 1:
                    context_boxes = torch.tensor(np.array([box for word, word_box in zip(batch['words'][batch_idx][page_idx], batch['boxes'][batch_idx][page_idx]) for box in [word_box] * len(self.tokenizer(word).input_ids[:-1])]))
                    context_boxes = context_boxes[:self.tokenizer.model_max_length - self.page_tokens - len(pretext_boxes) - 1]  # Remove boxes out of model max length.

                else:
                    context_boxes = torch.tensor(padding_box)

                all_boxes[batch_idx, page_idx, :self.page_tokens] = torch.tensor(page_token_box)
                all_boxes[batch_idx, page_idx, self.page_tokens: self.page_tokens + len(pretext_boxes)] = pretext_boxes
                # all_boxes[batch_idx, page_idx, self.page_tokens + length_pretext: self.page_tokens + length_pretext + len(context_boxes)] = context_boxes * 1000
                all_boxes[batch_idx, page_idx, self.page_tokens + length_pretext: self.page_tokens + length_pretext + len(context_boxes)] = context_boxes * 1000

                all_boxes[batch_idx, page_idx, self.page_tokens + length_pretext + len(context_boxes)] = torch.tensor(eos_box)

        max_seq = min(longest_sequence, self.max_text_tokens)
        all_input_ids = all_input_ids[:, :, :max_seq]
        all_boxes = all_boxes[:, :, :max_seq]
        all_attention_masks = all_attention_masks[:, :, :max_seq]

        all_input_ids = all_input_ids.to(self.device)
        all_boxes = all_boxes.to(self.device)
        all_attention_masks = all_attention_masks.to(self.device)

        return all_input_ids, all_boxes, all_attention_masks

    def forward(self, batch, output_attentions=False, return_pred_answer=False):
        question_id = batch['question_id']
        answers = batch['answers']
        num_pages = batch['num_pages']
        answer_page_idx = torch.LongTensor(batch['answer_page_idx']).to(self.device)

        bs = len(question_id)
        if self.page_retrieval == 'oracle':
            input_ids, input_boxes, attention_mask = self.prepare_vqa_input_ids(batch)

            raise ValueError("Oracle set-up not available for Hi-VT5. Instead, specify 'max_pages: 1' in dataset config with 'page_retrieval: custom'.")

        elif self.page_retrieval in ['logits', 'concat']:
            raise ValueError("{:s} set-up not available for Hi-LT5".format(self.page_retrieval.capitalize()))

        else:
            input_ids, input_boxes, attention_mask = self.prepare_vqa_input_ids(batch)

            if self.model.training or output_attentions:  # TODO: Output attentions should be for inference (generate). But I can't output encoder attentions..
            # if self.model.training:  # TODO: Output attentions should be for inference (generate). But I can't output encoder attentions..
                answers = [random.choice(answer) for answer in answers]
                labels = self.tokenizer(answers, return_tensors='pt', padding=True)
                labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
                labels = labels.input_ids.to(self.device)

                outputs = self.model(input_ids=input_ids, bbox=input_boxes, images=batch['images'], attention_mask=attention_mask, labels=labels, num_pages=num_pages, answer_page_idx=answer_page_idx, output_attentions=output_attentions)
                _, pred_answers, pred_answer_pages = self.get_answer_from_model_output(input_ids, input_boxes, batch['images'], attention_mask, num_pages) if return_pred_answer else None

            else:
                outputs, pred_answers, pred_answer_pages = self.get_answer_from_model_output(input_ids, input_boxes, batch['images'], attention_mask, num_pages)

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

        pred_answer_conf = [-1 for _ in range(len(pred_answers))]
        return outputs, pred_answers, pred_answer_pages, pred_answer_conf

    def get_answer_from_model_output(self, input_ids, boxes, images, attention_mask, num_pages):
        output = self.model.generate(input_ids=input_ids, bbox=boxes, images=images, attention_mask=attention_mask, num_pages=num_pages, max_length=self.decoding_max_length, output_attentions=True, return_dict_in_generate=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answer_pages = output.ret_logits.argmax(dim=-1).tolist()
        return output, pred_answers, pred_answer_pages

    def forward_record_and_retrieve_attention_dict(self, record):

        with torch.no_grad():
            batched_record = {k: [v] for k, v in record.items()}  # fake batch
            outputs, pred_answer, pred_answer_page = self.forward(batched_record, output_attentions=True, return_pred_answer=True)

        num_pages = record['num_pages']

        page_tokens = [f"[PAGE_{i}]" for i in range(self.page_tokens)]
        pretext = f"{page_tokens}: question: {record['questions']}  context:"  # Multiple representation
        pretext_tokens = self.tokenizer(pretext).input_ids[:-1]
        pretext_boxes = [np.array([0, 0, 1, 1], dtype=np.float32) for _ in range(len(pretext_tokens))]

        document_text = []
        document_boxes = []
        for page_idx in range(num_pages):
            page_tokens = pretext_tokens[:]  # Deepcopy
            page_boxes = pretext_boxes[:]  # Deepcopy
            # page_tokens = [token for _ in range(self.page_tokens) for token in self.tokenizer("[PAGE]").input_ids[:-1]]
            # page_boxes = [np.array([0, 0, 1, 1], dtype=np.float32) for _ in range(self.page_tokens)]
            tokenized_words = [self.tokenizer(page_words).input_ids[:-1] for page_words in record['words'][page_idx]]
            for tokenized_word, token_boxes in zip(tokenized_words, record['boxes'][page_idx]):
                page_tokens.extend(tokenized_word)
                page_boxes.extend([token_boxes]*len(tokenized_word))

            if len(page_tokens) > self.max_text_tokens:
                page_tokens = page_tokens[:self.max_text_tokens-1] + [self.tokenizer.eos_token_id]
                page_boxes = page_boxes[:self.max_text_tokens-1] + [np.array([0, 0, 0, 0], dtype=np.float32)]

            else:
                padding_size = self.max_text_tokens - len(page_boxes)
                page_tokens_pad = [self.tokenizer.pad_token_id] * padding_size
                page_boxes_pad = [box for box in np.zeros([padding_size, 4], dtype=np.float32)]

                page_tokens += page_tokens_pad
                page_boxes += page_boxes_pad

            page_tokens = self.tokenizer.convert_ids_to_tokens(page_tokens)

            page_tokens += ["[VIS_CLS]"] + ["[VIS_{:d}]".format(i) for i in range(0, 196)]
            page_boxes += self.model.visual_embeddings.get_visual_boxes(num_pages=1, scale=1).tolist()

            document_text.append(page_tokens)
            document_boxes.append(page_boxes)

        # answer_text = self.tokenizer.convert_ids_to_tokens(labels.input_ids[0])  # Only works if the prediction is the GT.
        answer_text = self.tokenizer.convert_ids_to_tokens(self.tokenizer(pred_answer[0]).input_ids)
        decoder_input_text = ["[PAGE_{:d},{:d}]".format(page_idx, token_idx) for page_idx in range(num_pages) for token_idx in range(self.page_tokens)]

        # Convert tensors to CPU
        encoder_att = []
        for page_idx in range(len(outputs.encoder_attentions)):
            encoder_att.append([att.data.cpu() for att in outputs.encoder_attentions[page_idx]])

        decoder_att = [att.data.cpu() for att in outputs.decoder_attentions]
        cross_att = [att.data.cpu() for att in outputs.cross_attentions]
        # decoder_att = [att.data.cpu() for att in outputs.decoder_attentions[0]]
        # cross_att = [att.data.cpu() for att in outputs.cross_attentions[0]]

        att_dict = {
            "encoder_att": encoder_att,
            "decoder_att": decoder_att,
            "cross_att": cross_att,
            "encoder_text": document_text,
            "encoder_boxes": document_boxes,
            "answer_text": answer_text,
            "decoder_input_text": decoder_input_text,
        }

        return outputs, pred_answer, pred_answer_page, att_dict
