import math
from PIL import Image
import numpy as np


def parse_values(v):
    if any(item is not None for item in v):
        return v

    else:
        return None


def docvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    batch = {k: parse_values(v) for k, v in batch.items()}  # If there is a list of None, replace it with single None value.

    return batch


def compute_grid(num_pages):
    rows = cols = math.ceil(math.sqrt(num_pages))

    if rows * (cols-1) >= num_pages:
        cols = cols-1

    return rows, cols


def get_page_position_in_grid(page, cols):
    page_row = math.floor(page/cols)
    page_col = page % cols

    return page_row, page_col


def create_grid_image(images, boxes=None):
    rows, cols = compute_grid(len(images))

    # rescaling to min width [height padding]
    min_width = min(im.width for im in images)
    images = [
        im.resize((min_width, int(im.height * min_width / im.width)), resample=Image.BICUBIC) for im in images
    ]

    w, h = max([img.size[0] for img in images]), max([img.size[1] for img in images])
    assert w == min_width
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    # Squeeze bounding boxes to the dimension of a single grid.
    for page_ix in range(len(boxes)):

        if len(boxes[page_ix]) == 0:
            boxes[page_ix] = np.empty((0,4))

        else:
            page_row, page_col = get_page_position_in_grid(page_ix, cols)
            boxes[page_ix][:, [0, 2]] = boxes[page_ix][:, [0, 2]] / cols * (page_col+1)  # Resize width
            boxes[page_ix][:, [1, 3]] = boxes[page_ix][:, [1, 3]] / rows * (page_row+1)  # Resize height

    boxes = np.concatenate(boxes)
    return grid, boxes
