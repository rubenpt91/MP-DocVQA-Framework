# tokenization utils

QTYPES = [
    "extractive",
    "abstractive",
    "list/abstractive",
    "list/extractive",
    "not-answerable",
]


def initialize_tokens_by_averaging(tokenizer, model, sorted_tokens):

    for idx in range(0, len(sorted_tokens)):

        tokens = tokenizer.tokenize(sorted_tokens[idx])

        tokenized_ids = tokenizer.convert_tokens_to_ids(tokens)

        tokenizer.add_tokens(sorted_tokens[idx])

        model.resize_token_embeddings(len(tokenizer))

        ##model.bert.embeddings.word_embeddings.weight
        # specific to T5
        model.shared.weight[-1, :] = model.shared.weight[tokenized_ids].mean(axis=0)


def initialize_tokens_randomly(tokenizer, model, sorted_tokens):
    tokenizer.add_tokens(sorted_tokens)
    # resize embedding layers
    model.resize_token_embeddings(len(tokenizer))
