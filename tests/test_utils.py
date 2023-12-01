import argparse


def set_test_config(config):
    config.save_dir = 'test/' + config.save_dir
    config.data_parallel = False
    config.batch_size = 2
    config.train_epochs = 1

    return config


def get_mock_args():
    args = argparse.Namespace()
    args.seed = 42
    return args


# python -m unittest discover tests/

    """ TODO:
     - Test forward.
     - Test predict / generate. 
     - Test data_parallel.
     - Ensure that we can get the answers from public test set.
    """
