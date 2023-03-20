import os, time, datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.SP_DocVQA import singlepage_docvqa_collate_fn
from logger import Logger
from metrics import Evaluator
from utils import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset


def evaluate(data_loader, model, evaluator, **kwargs):

    return_scores_by_sample = kwargs.get('return_scores_by_sample', False)
    return_answers = kwargs.get('return_answers', False)

    if return_scores_by_sample:
        scores_by_samples = {}
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
        bs = len(batch['question_id'])
        with torch.no_grad():
            outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)
            # print(pred_answers)

        metric = evaluator.get_metrics(batch['answers'], pred_answers, batch.get('answer_type', None))

        if 'answer_page_idx' in batch and pred_answer_page is not None:
            ret_metric = evaluator.get_retrieval_metric(batch['answer_page_idx'], pred_answer_page)
        else:
            ret_metric = [0 for _ in range(bs)]

        if return_scores_by_sample:
            for batch_idx in range(bs):
                scores_by_samples[batch['question_id'][batch_idx]] = {
                    'accuracy': metric['accuracy'][batch_idx],
                    'anls': metric['anls'][batch_idx],
                    'ret_prec': ret_metric[batch_idx],
                    'pred_answer': pred_answers[batch_idx],
                    'pred_answer_conf': answer_conf[batch_idx],
                    'pred_answer_page': pred_answer_page[batch_idx] if pred_answer_page is not None else None
                }

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
        total_ret_prec = total_ret_prec/len(data_loader.dataset)
        scores_by_samples = []

    return total_accuracies, total_anls, total_ret_prec, all_pred_answers, scores_by_samples


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args)
    start_time = time.time()

    dataset = build_dataset(config, 'test')
    val_data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=singlepage_docvqa_collate_fn)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    accuracy_list, anls_list, answer_page_pred_acc_list, pred_answers, scores_by_samples = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=True, return_answers=True)
    accuracy, anls, answ_page_pred_acc = np.mean(accuracy_list), np.mean(anls_list), np.mean(answer_page_pred_acc_list)

    inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
    logger.log_val_metrics(accuracy, anls, answ_page_pred_acc, update_best=False)

    save_data = {
        "Model": config["model_name"],
        "Model_weights": config["model_weights"],
        "Dataset": config["dataset_name"],
        "Page retrieval": config.get('page_retrieval', '-').capitalize(),
        "Inference time": inf_time,
        "Mean accuracy": accuracy,
        "Mean ANLS": anls,
        "Mean Retrieval precision": answ_page_pred_acc,
        "Scores by samples": scores_by_samples,
    }

    experiment_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(config['save_dir'], 'results', "{:}_{:}_{:}__{:}.json".format(config['model_name'], config['dataset_name'], config.get('page_retrieval', '').lower(), experiment_date))
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))


