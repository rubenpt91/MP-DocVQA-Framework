import random

import os, yaml, json
import argparse

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Baselines for MP-DocVQA')

    # Required
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to yml file with model configuration.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to yml file with dataset configuration.')

    # Optional
    parser.add_argument('--eval-start', action='store_true', default=True, help='Whether to evaluate the model before training or not.')
    parser.add_argument('--no-eval-start', dest='eval_start', action='store_false')

    # Overwrite config parameters
    parser.add_argument('-bs', '--batch-size', type=int, help='DataLoader batch size.')
    parser.add_argument('--seed', type=int, help='Seed to allow reproducibility.')

    parser.add_argument('--data-parallel', action='store_true', help='Boolean to indicate to parallelize the execution.')
    parser.add_argument('--no-data-parallel', action='store_false', dest='data_parallel', help='Boolean to indicate to parallelize the execution.')
    return parser.parse_args()


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


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
    page_retrieval = config.get('page_retrieval', '').lower()
    if model_name not in ['hi-layoutlmv3', 'hi-lt5', 'hi-vlt5'] and page_retrieval == 'custom':
        raise ValueError("'Custom' retrieval is not allowed for {:}".format(model_name))

    elif model_name in ['hi-layoutlmv3, hilt5', 'hi-lt5', 'hi-lt5'] and page_retrieval in ['concat', 'logits']:
        raise ValueError("Hierarchical model {:} can't run on {:} retrieval type. Only 'oracle' and 'custom' are allowed.".format(model_name, page_retrieval))

    if page_retrieval in ['concat', 'logits'] and config.get('max_pages') is not None:
        print("WARNING - Max pages ({:}) value is ignored for {:} retrieval setting.".format(config.get('max_pages'), page_retrieval))

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
