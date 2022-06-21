
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.SingleDocVQA import SingleDocVQA, singledocvqa_collate_fn
from logger import Logger
from metrics import Evaluator
from utils import parse_args, load_config, save_json
from build_utils import build_model


def evaluate(data_loader, model, evaluator, **kwargs):

    return_scores_by_sample = kwargs['return_scores_by_sample'] if 'return_scores_by_sample' in kwargs else False
    return_answers = kwargs['return_answers'] if 'return_answers' in kwargs else False

    if return_scores_by_sample:
        total_accuracies = []
        total_anls = []

    else:
        total_accuracies = 0
        total_anls = 0

    all_pred_answers = []
    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        questions = batch['questions']
        contexts = batch['contexts']
        gt_answers = batch['answers']
        # start_pos = torch.LongTensor(batch['start_indxs'])
        # end_pos = torch.LongTensor(batch['end_indxs'])

        with torch.no_grad():
            outputs, pred_answers = model.forward(questions, contexts, gt_answers, return_pred_answer=True)  # Longformer
            # outputs, pred_answers = model.forward(questions, contexts, start_pos=start_pos, end_pos=end_pos, return_pred_answer=True)  # Longformer SQuAD
            # print(pred_answers)

        metric = evaluator.get_metrics(gt_answers, pred_answers)

        if return_scores_by_sample:
            total_accuracies.extend(metric['accuracy'])
            total_anls.extend(metric['anls'])
        else:
            total_accuracies += sum(metric['accuracy'])
            total_anls += sum(metric['anls'])

        if return_answers:
            all_pred_answers.extend(pred_answers)

    if not return_scores_by_sample:
        total_accuracies = total_accuracies/len(data_loader.dataset)
        total_anls = total_anls/len(data_loader.dataset)

    return total_accuracies, total_anls, all_pred_answers


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args)

    dataset = SingleDocVQA(config['imdb_dir'], split='val')
    val_data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=singledocvqa_collate_fn)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    accuracy_list, anls_list, answers = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=True, return_answers=True)
    accuracy, anls = np.mean(accuracy_list), np.mean(anls_list)
    logger.log_val_metrics(accuracy, anls, update_best=False)

    save_data = {
        "Model": config["model_name"],
        "Model_weights": config["model_weights"],
        "Mean accuracy": np.mean(accuracy),
        "Mean ANLS": np.mean(anls),
        "Sample_accuracy": accuracy,
        "Sample_anls": anls,
        "Answers": answers,
    }

    save_json("{:s}/{:}_results.json".format(config['save_dir'], save_data['Model']), save_data)

