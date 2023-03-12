import torch


def get_generative_confidence(output):
    batch_logits = torch.stack(output.scores, dim=1)[:, :-1, :]  # b x s x V and dropping EOS token
    decoder_output_confs = torch.amax(batch_logits.softmax(-1), 2)
    confidences = decoder_output_confs.prod(1)  # b
    return confidences.tolist()
