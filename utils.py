import json


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


