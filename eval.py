import time, datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.SingleDocVQA import SingleDocVQA, singledocvqa_collate_fn
from logger import Logger
from metrics import Evaluator
from utils import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset


def evaluate(data_loader, model, evaluator, **kwargs):

    return_scores_by_sample = kwargs['return_scores_by_sample'] if 'return_scores_by_sample' in kwargs else False
    return_answers = kwargs['return_answers'] if 'return_answers' in kwargs else False

    if return_scores_by_sample:
        total_accuracies = []
        total_anls = []
        total_ret_prec = []

    else:
        total_accuracies = 0
        total_anls = 0
        total_ret_prec = 0

    all_pred_answers = []
    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            outputs, pred_answers, pred_answer_page = model.forward(batch, return_pred_answer=True)  # Longformer
            # outputs, pred_answers, answer_page = model.forward(questions, contexts, gt_answers, return_pred_answer=True)  # Longformer
            # outputs, pred_answers = model.forward(questions, contexts, start_pos=start_pos, end_pos=end_pos, return_pred_answer=True)  # Longformer SQuAD
            # print(pred_answers)

        metric = evaluator.get_metrics(batch['answers'], pred_answers)
        ret_metric = evaluator.get_retrieval_metric(batch['answer_page_idx'], pred_answer_page)

        if return_scores_by_sample:
            total_accuracies.extend(metric['accuracy'])
            total_anls.extend(metric['anls'])
            total_ret_prec.extend(ret_metric)

        else:
            total_accuracies += sum(metric['accuracy'])
            total_anls += sum(metric['anls'])
            total_ret_prec += sum(ret_metric)

        if return_answers:
            all_pred_answers.extend(pred_answers)

    if not return_scores_by_sample:
        total_accuracies = total_accuracies/len(data_loader.dataset)
        total_anls = total_anls/len(data_loader.dataset)

    return total_accuracies, total_anls, total_ret_prec, all_pred_answers


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args)
    start_time = time.time()

    dataset = build_dataset(config, 'val')
    val_data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=singledocvqa_collate_fn)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    accuracy_list, anls_list, ret_prec_list, pred_answers = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=True, return_answers=True)
    accuracy, anls, ret_prec = np.mean(accuracy_list), np.mean(anls_list), np.mean(ret_prec_list)

    inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
    logger.log_val_metrics(accuracy, anls, ret_prec, update_best=False)

    save_data = {
        "Model": config["model_name"],
        "Model_weights": config["model_weights"],
        "Dataset": config["dataset_name"],
        "Page retrieval": config.get('page_retrieval', '-').capitalize(),
        "Inference time": inf_time,
        "Mean accuracy": accuracy,
        "Mean ANLS": anls,
        "Mean Retrieval precision": ret_prec,
        "Sample_accuracy": accuracy_list,
        "Sample_anls": anls_list,
        "Sample_ret_prec": ret_prec_list,
        "Answers": pred_answers,
    }

    experiment_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_json("{:s}/results/{:}_results__{:}.json".format(config['save_dir'], config['model_name'], experiment_date), save_data)
