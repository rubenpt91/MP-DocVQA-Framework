import argparse


def set_test_config(config):
    config.save_dir = 'test/' + config.save_dir
    config.batch_size = 2
    config.train_epochs = 1

    return config


def get_mock_args():
    args = argparse.Namespace()
    args.seed = 42
    return args


def print_answers_func(gt_answers, pred_answers):
    if gt_answers is not None:
        print("".join(["\t\tGT: {:} \tPred: {:}\n".format(gt_answer, pred_answer) for gt_answer, pred_answer in zip(gt_answers, pred_answers)]))
    else:
        print("".join(["\t\tGT: {:} \tPred: {:}\n".format(None, pred_answer) for pred_answer in pred_answers]))


# python -m unittest discover tests/

    """ TODO:
     - Test forward.
     - Test predict / generate. 
     - Test data_parallel.
     - Ensure that we can get the answers from public test set.
    """
