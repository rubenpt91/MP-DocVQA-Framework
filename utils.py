import yaml, json


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def load_config(config_path):
    return yaml.safe_load(open(config_path, "r"))

