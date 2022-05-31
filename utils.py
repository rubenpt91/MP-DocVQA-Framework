import yaml, json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Baselines for DocCVQAv2')

    parser.add_argument('-c', '--config', type=str, required=True, help='Path to yml file with experiment configuration.')

    return parser.parse_args()


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def load_config(config_path, args):
    return parse_config(yaml.safe_load(open(config_path, "r")), args)


def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get('includes', []):
        config = load_config(included_config_path, args) | config

    # Append and overwrite config values from argumments.
    config = {k: v for k, v in args._get_kwargs()} | config
    return config

