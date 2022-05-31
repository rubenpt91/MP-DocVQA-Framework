"""
Data preprocessing script from: https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/longformer_qa_training.ipynb#scrollTo=UYOvGLdVgoxt
"""
import nlp
import torch

from transformers import LongformerTokenizerFast


def get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx  # When the gold label position is good
    elif context[start_idx - 1:end_idx - 1] == gold_text:
        return start_idx - 1, end_idx - 1  # When the gold label is off by one character
    elif context[start_idx - 2:end_idx - 2] == gold_text:
        return start_idx - 2, end_idx - 2  # When the gold label is off by two character
    else:
        raise ValueError()


# Tokenize our training dataset
def convert_to_features(example):
    tokenizer = LongformerTokenizerFast.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

    # Tokenize contexts and questions (as pairs of inputs)
    input_pairs = [example['question'], example['context']]
    encodings = tokenizer.encode_plus(input_pairs, padding='max_length', truncation=True, max_length=512)
    context_encodings = tokenizer.encode_plus(example['context'])

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    # this will give us the position of answer span in the context text
    start_idx, end_idx = get_correct_alignement(example['context'], example['answers'])
    start_positions_context = context_encodings.char_to_token(start_idx)
    end_positions_context = context_encodings.char_to_token(end_idx - 1)

    # here we will compute the start and end position of the answer in the whole example
    # as the example is encoded like this <s> question</s></s> context</s>
    # and we know the postion of the answer in the context
    # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
    # this will give us the position of the answer span in whole example
    sep_idx = encodings['input_ids'].index(tokenizer.sep_token_id)
    start_positions = start_positions_context + sep_idx + 1
    end_positions = end_positions_context + sep_idx + 1

    if end_positions > 512:
        start_positions, end_positions = 0, 0

    encodings.update({'start_positions': start_positions,
                      'end_positions': end_positions,
                      'attention_mask': encodings['attention_mask']})
    return encodings


if __name__ == '__main__':
    # Load train and validation split of squad
    # train_dataset = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
    valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

    # train_dataset = train_dataset.map(convert_to_features)
    valid_dataset = valid_dataset.map(convert_to_features, load_from_cache_file=False)

    # Set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    # train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    # Cache the dataset, so we can load it directly for training
    # torch.save(train_dataset, '/SSD2/DocCVQA_Baselines_data/SQuAD/train_data.pt')
    torch.save(valid_dataset, '/SSD2/DocCVQA_Baselines_data/SQuAD/val_data.pt')

