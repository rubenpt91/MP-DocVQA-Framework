import ast, random

import os, yaml, json
import argparse

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Baselines for MP-DocVQA")

    # Required
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to yml file with model configuration.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to yml file with dataset configuration.",
    )

    # Optional

    parser.add_argument(
        "--eval-start",
        action="store_true",
        default=True,
        help="Whether to evaluate the model before training or not.",
    )
    parser.add_argument("--no-eval-start", dest="eval_start", action="store_false")

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=False,
        default="",
        help="Path to checkpoint with configuration.",
    )
    
    parser.add_argument(
        "--cf",
        dest="return_confidence",
        action="store_true",
        default=False,
        help="Add confidence to sample scoring",
    )
    # Overwrite config parameters
    parser.add_argument("-bs", "--batch-size", type=int, help="DataLoader batch size.")
    parser.add_argument("--seed", type=int, help="Seed to allow reproducibility.")

    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Boolean to overwrite data-parallel arg in config parallelize the execution.",
    )
    parser.add_argument(
        "--no-data-parallel",
        action="store_false",
        dest="data_parallel",
        help="Boolean to overwrite data-parallel arg in config to indicate to parallelize the execution.",
    )
    return parser.parse_args()


def parse_multitype2list_arg(argument):
    if argument is None:
        return argument

    if "-" in argument and "[" in argument and "]" in argument:
        first, last = argument.strip("[]").split("-")
        argument = list(range(int(first), int(last)))
        return argument

    argument = ast.literal_eval(argument)

    if isinstance(argument, int):
        argument = [argument]

    elif isinstance(argument, list):
        argument = argument

    return argument


def save_json(path, data):
    with open(path, "w+") as f:
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def check_config(config):
    model_name = config["model_name"].lower()

    if "page_retrieval" not in config:
        config["page_retrieval"] = "none"

    page_retrieval = config["page_retrieval"]
    if (
        model_name not in ["hi-layoutlmv3", "hi-lt5", "hi-vt5"]
        and page_retrieval == "custom"
    ):
        raise ValueError("'Custom' retrieval is not allowed for {:}".format(model_name))

    elif model_name in [
        "hi-layoutlmv3, hilt5",
        "hi-lt5",
        "hivt5",
        "hi-vt5",
    ] and page_retrieval in ["concat", "logits"]:
        raise ValueError(
            "Hierarchical model {:} can't run on {:} retrieval type. Only 'oracle' and 'custom' are allowed.".format(
                model_name, page_retrieval
            )
        )

    if page_retrieval in ["concat", "logits"] and config.get("max_pages") is not None:
        print(
            "WARNING - Max pages ({:}) value is ignored for {:} page-retrieval setting.".format(
                config.get("max_pages"), page_retrieval
            )
        )

    elif page_retrieval == "none" and config["dataset_name"] not in ["SP-DocVQA"]:
        print(
            "Page retrieval can't be none for dataset '{:s}'. This is intended only for single page datasets. Please specify in the method config file the 'page_retrieval' setup to one of the following: [oracle, concat, logits, custom] ".format(
                config["dataset_name"]
            )
        )

    return True


def load_config(args):
    if args.checkpoint:
        checkpoint_config = os.path.join(args.checkpoint, "config.yml")
        if os.path.exists(checkpoint_config):
            print("Loading model from checkpoint")
            config = parse_config(yaml.safe_load(open(checkpoint_config, "r")), args)
            print(config)
        else:
            raise FileNotFoundError(f"{checkpoint_config}")
    else:
        model_config_path = "configs/models/{:}.yml".format(args.model)
        model_config = parse_config(yaml.safe_load(open(model_config_path, "r")), args)
        dataset_config_path = "configs/datasets/{:}.yml".format(args.dataset)
        dataset_config = parse_config(
            yaml.safe_load(open(dataset_config_path, "r")), args
        )
        training_config = model_config.pop("training_parameters")

        # Append and overwrite config values from arguments.
        # config = {'dataset_params': dataset_config, 'model_params': model_config, 'training_params': training_config}
        config = {**dataset_config, **model_config, **training_config}

    config.update({k: v for k, v in args._get_kwargs() if v is not None})
    config.pop("model")
    config.pop("dataset")

    # Set default seed
    if "seed" not in config:
        print("Seed not specified. Setting default seed to '{:d}'".format(42))
        config["seed"] = 42

    check_config(config)

    return config


def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get("includes", []):
        config.update(load_config(included_config_path, args))

    return config


def correct_alignment(context, answer, start_idx, end_idx):

    if context[start_idx:end_idx] == answer:
        return [start_idx, end_idx]

    elif context[start_idx - 1 : end_idx] == answer:
        return [start_idx - 1, end_idx]

    elif context[start_idx : end_idx + 1] == answer:
        return [start_idx, end_idx + 1]

    else:
        print(context[start_idx:end_idx], answer)
        return None


def time_stamp_to_hhmmss(timestamp, string=True):
    hh = int(timestamp / 3600)
    mm = int((timestamp - hh * 3600) / 60)
    ss = int(timestamp - hh * 3600 - mm * 60)

    time = "{:02d}:{:02d}:{:02d}".format(hh, mm, ss) if string else [hh, mm, ss]

    return time
