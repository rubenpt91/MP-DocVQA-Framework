import os, ast, yaml, json, random
import argparse

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument('-sf', '--use_spatial_features', type=str2bool, help='Specify whether or not use spatial features.')
    parser.add_argument('-vf', '--use_visual_features', type=str2bool, help='Specify whether or not use vsiual features.')
    parser.add_argument('--seed', type=int, help='Seed to allow reproducibility.')
    parser.add_argument('--save-dir', type=str, help='Seed to allow reproducibility.')

    # Parallelism
    parser.add_argument('-dp', '--data-parallel', type=str2bool, help='Boolean to overwrite data-parallel arg in config parallelize the execution.')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of available nodes/hosts.')
    parser.add_argument('--node_id',   type=int, default=0, help='Unique ID too identify the current node/host.')
    parser.add_argument('--num_gpus',  type=int, default=1, help='Number of GPUs in each node.')
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
    model_name = config.model_name.lower()
    dataset_name = config.dataset_name.lower()

    if 'page_retrieval' not in config:
        config.page_retrieval = 'none'

    if 'use_spatial_features' in config and model_name not in ['vt5', 'hivt5']:
        print("WARNING - The ability to enable/desable spatial features is implemented only for VT5 and Hi-VT5. This will be ignored for {:}, and run with default configuration".format(config.model_name))

    if 'use_visual_features' in config and model_name not in ['vt5', 'hivt5']:
        print("WARNING - The ability to enable/desable visual features is implemented only for VT5 and Hi-VT5. This will be ignored for {:}, and run with default configuration.".format(config.model_name))

    page_retrieval = config.page_retrieval.lower()
    max_pages = getattr(config, 'max_pages', None)
    if model_name not in ['hi-vt5'] and page_retrieval == 'custom':
        raise ValueError("'Custom' retrieval is not allowed for {:}".format(model_name))

    elif model_name in ['hi-vt5'] and page_retrieval in ['concat', 'logits']:
        raise ValueError("Hierarchical model {:} can't run on {:} retrieval type. Only 'oracle' and 'custom' are allowed.".format(model_name, page_retrieval))

    if page_retrieval == 'oracle' and dataset_name == 'dude':
        raise ValueError("'Oracle' set-up is not valid for DUDE, since there is no GT for the answer page.")

    elif page_retrieval == 'custom' and model_name not in ['hi-vt5']:
        raise ValueError("'Custom' page retrieval only allowed for Heirarchical methods ('hi-vt5').")

    elif page_retrieval in ['concat', 'logits'] and max_pages is not None:
        print("WARNING - Max pages ({:}) value is ignored for {:} page-retrieval setting.".format(max_pages, page_retrieval))

    elif page_retrieval == 'none' and config.multipage_dataset:
        print("Page retrieval can't be 'none' for dataset '{:s}'. This is intended only for single page datasets. Please specify in the method config file the 'page_retrieval' setup to one of the following: [oracle, concat, logits, custom] ".format(config.dataset_name))

    config.world_size = config.num_gpus * config.num_nodes
    config.distributed = True if config.world_size > 1 else False

    if config.num_nodes > 1:
        print("WARNING - This framework has been never tested using more than 1 node. Please, check that everything works as expected before blindly relying on this.")

    if 'save_dir' in config:
        if not config.save_dir.endswith('/'):
            config.save_dir = config.save_dir + '/'

        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

    return True


def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get('includes', []):
        config = load_config(included_config_path, args) | config

    return config


def config_to_Namespace(config):
    config = argparse.Namespace(**config)
    # config = argparse.Namespace(**{k: config_to_Namespace(v) if isinstance(v, dict) else v for k,v in config.items()})
    return config


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

    config = config_to_Namespace(config)
    check_config(config)
    return config


def time_stamp_to_hhmmss(timestamp, string=True):
    hh = int(timestamp/3600)
    mm = int((timestamp-hh*3600)/60)
    ss = int(timestamp - hh*3600 - mm*60)

    time = "{:02d}:{:02d}:{:02d}".format(hh, mm, ss) if string else [hh, mm, ss]

    return time