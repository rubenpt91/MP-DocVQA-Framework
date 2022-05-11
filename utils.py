import yaml, json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Baselines for DocCVQAv2')

    parser.add_argument('-c', '--config', type=str, required=True, help='Path to yml file with experiment configuration.')

    return parser.parse_args()


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def load_config(config_path):
    return yaml.safe_load(open(config_path, "r"))

