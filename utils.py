import yaml, json

from models.Longformer import Longformer
from models.BertQA import BertQA


def build_model(config):

    if config['Model'].lower() == 'bert' or config['Model'].lower() == 'bertqa':
        model = BertQA(config)

    elif config['Model'].lower() == 'longformer':
        model = Longformer(config)

    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose BertQA or Longformer".format(config['Model']))

    model.model.to(config['device'])
    return model


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def load_config(config_path):
    return yaml.safe_load(open(config_path, "r"))

