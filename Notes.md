# Longformer
* arXiv [paper](https://arxiv.org/pdf/2004.05150.pdf)
* Huggingface [documentation](https://huggingface.co/docs/transformers/model_doc/longformer)
* Medium [blog](https://medium.com/dair-ai/longformer-what-bert-should-have-been-78f4cd595be9#:~:text=The%2030%20layer%20model%2C%20when,(102M%20vs%20277M%20parameters)). Check graphics of memory?


config.attention_window
global_attention_mask
* 0: the token attends “locally”,
* 1: the token attends “globally”.


Problem of Longformer: Since it's based on window. It might miss elements far from the current token. This might be crucial in tables, for example, where the column name might be many tokens (sequencially ordered) before. Hence, loosing the ability to attend freely to any other token in the document, might 'suponer' loss of information and decrease of performance.
Also, it allows up to 4,096 tokens, where we should allow much more...

Longformer: 
 * Everything global: Full context
 * Only questions global: Window of w

Since it's BERT-like it is an extractive question ansering method.
(Comes from RoBERTa)


**How the global attention mask is used?**

Code in [forward](https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/longformer/modeling_longformer.py#L1964)
if global_attention_mask is None:
    global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)


Up to 4,096 tokens?
config.attention_window.

len(model.config.attention_window)  # len(attention_window) == num_hidden_layers - [Documentation](https://huggingface.co/docs/transformers/model_doc/longformer#transformers.LongformerConfig.attention_window)



Global attention mask: [Documentation](https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/longformer#transformers.LongformerModel.forward.global_attention_mask)
