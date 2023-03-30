import random, re
import torch
import numpy as np
from utils import correct_alignment


def get_start_end_idx(model, encoding, context, context_encoding, answers, context_page_char_correspondent, page_retrieval, sep_token_id, pad_token_id, ignore_id, device):
    pos_idx = []
    context_page_token_correspondent = []
    for batch_idx in range(len(context)):
        batch_pos_idxs = []
        for answer in answers[batch_idx]:
            start_idxs = [m.start() for m in re.finditer(re.escape(answer), context[batch_idx])]

            for start_idx in start_idxs:
                end_idx = start_idx + len(answer)
                start_idx, end_idx = correct_alignment(context[batch_idx], answer, start_idx, end_idx)

                if start_idx is not None and end_idx != 0:
                    batch_pos_idxs.append([start_idx, end_idx])
                    break

        if len(batch_pos_idxs) == 0:
            # Answer not in context
            pos_idx.append([ignore_id, ignore_id])

        else:
            # Select one of the possible correct answers
            start_idx, end_idx = random.choice(batch_pos_idxs)

            start_positions_context = context_encoding[batch_idx].char_to_token(start_idx)
            end_positions_context = context_encoding[batch_idx].char_to_token(end_idx - 1)

            # here we will compute the start and end position of the answer in the whole example
            # as the example is encoded like this <s> question</s></s> context</s>
            # and we know the position of the answer in the context
            # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
            # this will give us the position of the answer span in whole example
            sep_idx = encoding['input_ids'][batch_idx].tolist().index(sep_token_id)

            if start_positions_context is not None and end_positions_context is not None:

                if model in ['BertQA', 'BigBird']:
                    start_position = start_positions_context + sep_idx
                    end_position = end_positions_context + sep_idx

                elif model in ['Longformer']:  # Two <sep> tokens together. Then, extra position must be considered.
                    start_position = start_positions_context + sep_idx + 1
                    end_position = end_positions_context + sep_idx + 1

            else:
                # The answer is in the context but out of the input sequence due to max sequence limitation (512 / 4096)
                start_position, end_position = ignore_id, ignore_id

            pos_idx.append([start_position, end_position])

        # Page correspondence for concat:
        if page_retrieval == 'concat':
            # context_encodings = tokenizer.encode_plus(context[batch_idx], padding=True, truncation=True)
            # context_encoding
            page_change_idxs = [0] + [i + 1 for i, x in enumerate(context_page_char_correspondent[batch_idx]) if x == -1]
            page_change_idxs_tokens = [context_encoding[batch_idx].char_to_token(idx) for idx in page_change_idxs]

            page_tok_corr = np.full(len(context_encoding.input_ids[batch_idx]), fill_value=-1)
            for page_idx in range(len(page_change_idxs_tokens)):
                if page_change_idxs_tokens[page_idx] is None:
                    break

                start_page_idx = page_change_idxs_tokens[page_idx]
                if page_idx + 1 < len(page_change_idxs_tokens) and page_change_idxs_tokens[page_idx + 1] is not None:
                    end_page_idx = page_change_idxs_tokens[page_idx + 1]
                else:
                    end_page_idx = context_encoding.input_ids[batch_idx].index(pad_token_id) if pad_token_id in context_encoding.input_ids[batch_idx] else None

                page_tok_corr[start_page_idx:end_page_idx] = page_idx

            context_page_token_correspondent.append(page_tok_corr)

    start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(device)
    end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(device)

    return start_idxs, end_idxs, context_page_token_correspondent


def get_extractive_confidence(outputs):
    bs = len(outputs['start_logits'])
    start_idxs = torch.argmax(outputs.start_logits, axis=1)
    end_idxs = torch.argmax(outputs.end_logits, axis=1)

    answ_confidence = []
    for batch_idx in range(bs):
        conf_mat = np.matmul(np.expand_dims(outputs.start_logits.softmax(dim=1)[batch_idx].unsqueeze(dim=0).detach().cpu(), -1),
                             np.expand_dims(outputs.end_logits.softmax(dim=1)[batch_idx].unsqueeze(dim=0).detach().cpu(), 1)).squeeze(axis=0)

        answ_confidence.append(
            conf_mat[start_idxs[batch_idx], end_idxs[batch_idx]].item()
        )

    return answ_confidence


def get_generative_confidence(output):
    batch_logits = torch.stack(output.scores, dim=1)[:, :-1, :]  # b x s x V and dropping EOS token
    decoder_output_confs = torch.amax(batch_logits.softmax(-1), 2)
    confidences = decoder_output_confs.prod(1)  # b
    return confidences.tolist()
