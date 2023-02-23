# tokenization utils

QTYPES = ["extractive", "abstractive", "list/abstractive", "list/extractive", "not-answerable"]


def initialize_tokens_by_averaging(tokenizer, model, sorted_tokens):
    with torch.no_grad():
        for idx in range(0, len(sorted_tokens)):

            tokens = tokenizer.tokenize(sorted_tokens[idx])

            tokenized_ids = tokenizer.convert_tokens_to_ids(tokens)

            tokenizer.add_tokens(sorted_tokens[idx])

            model.resize_token_embeddings(len(tokenizer))

            ##model.bert.embeddings.word_embeddings.weight
            #specific to T5

def initialize_tokens_randomly(tokenizer, model, sorted_tokens):

    for idx in range(0, len(sorted_tokens)):
        tokenizer.add_tokens(sorted_tokens[idx])

    # resize embedding layers
    model.resize_token_embeddings(len(tokenizer))
    

