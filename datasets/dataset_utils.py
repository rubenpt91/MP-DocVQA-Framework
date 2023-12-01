
def parse_values(v):
    if any(item is not None for item in v):
        return v

    else:
        return None


def docvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    batch = {k: parse_values(v) for k, v in batch.items()}  # If there is a list of None, replace it with single None value.

    return batch
