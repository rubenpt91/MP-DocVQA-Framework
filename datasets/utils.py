import ast, math, random
from PIL import Image

import os, yaml, json
import argparse

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='MP-DocVQA framework')

    # Required
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to yml file with model configuration.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to yml file with dataset configuration.')

    # Optional
    parser.add_argument('--eval-start', action='store_true', default=True, help='Whether to evaluate the model before training or not.')
    parser.add_argument('--no-eval-start', dest='eval_start', action='store_false')

    # Overwrite config parameters
    parser.add_argument('-p', '--page-retrieval', type=str, help='Page retrieval set-up.')
    parser.add_argument('-bs', '--batch-size', type=int, help='DataLoader batch size.')
    parser.add_argument('-msl', '--max-sequence-length', type=int, help='Max input sequence length of the model.')
    parser.add_argument('--seed', type=int, help='Seed to allow reproducibility.')
    parser.add_argument('--save-dir', type=str, help='Seed to allow reproducibility.')

    parser.add_argument('--data-parallel', action='store_true', help='Boolean to overwrite data-parallel arg in config parallelize the execution.')
    parser.add_argument('--no-data-parallel', action='store_false', dest='data_parallel', help='Boolean to overwrite data-parallel arg in config to indicate to parallelize the execution.')
    return parser.parse_args()


def parse_multitype2list_arg(argument):
    if argument is None:
        return argument

    if '-' in argument and '[' in argument and ']' in argument:
        first, last = argument.strip('[]').split('-')
        argument = list(range(int(first), int(last)))
        return argument

    argument = ast.literal_eval(argument)

    if isinstance(argument, int):
        argument = [argument]

    elif isinstance(argument, list):
        argument = argument

    return argument


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def save_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f)


"""
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
"""


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def check_config(config):
    model_name = config['model_name'].lower()

    if 'page_retrieval' not in config:
        config['page_retrieval'] = 'none'

    page_retrieval = config['page_retrieval'].lower()
    if model_name not in ['hi-layoutlmv3', 'hi-lt5', 'hi-vt5'] and page_retrieval == 'custom':
        raise ValueError("'Custom' retrieval is not allowed for {:}".format(model_name))

    elif model_name in ['hi-layoutlmv3, hilt5', 'hi-lt5', 'hivt5', 'hi-vt5'] and page_retrieval in ['concat', 'logits']:
        raise ValueError("Hierarchical model {:} can't run on {:} retrieval type. Only 'oracle' and 'custom' are allowed.".format(model_name, page_retrieval))

    if page_retrieval == 'custom' and model_name not in ['hi-layoutlmv3', 'hi-lt5', 'hi-vt5']:
        raise ValueError("'Custom' page retrieval only allowed for Heirarchical methods ('hi-layoutlmv3', 'hi-lt5', 'hi-vt5').")

    elif page_retrieval in ['concat', 'logits'] and config.get('max_pages') is not None:
        print("WARNING - Max pages ({:}) value is ignored for {:} page-retrieval setting.".format(config.get('max_pages'), page_retrieval))

    elif page_retrieval == 'none' and config['dataset_name'] not in ['SP-DocVQA']:
        print("Page retrieval can't be none for dataset '{:s}'. This is intended only for single page datasets. Please specify in the method config file the 'page_retrieval' setup to one of the following: [oracle, concat, logits, custom] ".format(config['dataset_name']))

    if 'save_dir' in config:
        if not config['save_dir'].endswith('/'):
            config['save_dir'] = config['save_dir'] + '/'

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

    return True


def load_config(args):
    model_config_path = "configs/models/{:}.yml".format(args.model)
    dataset_config_path = "configs/datasets/{:}.yml".format(args.dataset)
    model_config = parse_config(yaml.safe_load(open(model_config_path, "r")), args)
    dataset_config = parse_config(yaml.safe_load(open(dataset_config_path, "r")), args)
    training_config = model_config.pop('training_parameters')

    # Append and overwrite config values from arguments.
    # config = {'dataset_params': dataset_config, 'model_params': model_config, 'training_params': training_config}
    config = {**dataset_config, **model_config, **training_config}

    config = config | {k: v for k, v in args._get_kwargs() if v is not None}
    config.pop('model')
    config.pop('dataset')

    # Set default seed
    if 'seed' not in config:
        print("Seed not specified. Setting default seed to '{:d}'".format(42))
        config['seed'] = 42

    check_config(config)

    return config


def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get('includes', []):
        config = load_config(included_config_path, args) | config

    return config


def correct_alignment(context, answer, start_idx, end_idx):

    if context[start_idx: end_idx] == answer:
        return [start_idx, end_idx]

    elif context[start_idx - 1: end_idx] == answer:
        return [start_idx - 1, end_idx]

    elif context[start_idx: end_idx + 1] == answer:
        return [start_idx, end_idx + 1]

    else:
        print(context[start_idx: end_idx], answer)
        return None


def time_stamp_to_hhmmss(timestamp, string=True):
    hh = int(timestamp/3600)
    mm = int((timestamp-hh*3600)/60)
    ss = int(timestamp - hh*3600 - mm*60)

    time = "{:02d}:{:02d}:{:02d}".format(hh, mm, ss) if string else [hh, mm, ss]

    return time


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
            continue

        page_row, page_col = get_page_position_in_grid(page_ix, cols)
        boxes[page_ix][:, [0, 2]] = boxes[page_ix][:, [0, 2]] / cols * (page_col+1)  # Resize width
        boxes[page_ix][:, [1, 3]] = boxes[page_ix][:, [1, 3]] / rows * (page_row+1)  # Resize height

    boxes = np.concatenate(boxes)
    return grid, boxes
