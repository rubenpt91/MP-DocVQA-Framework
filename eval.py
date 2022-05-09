
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.SingleDocVQA import SingleDocVQA, singledocvqa_collate_fn
from models.Longformer import Longformer
from metrics import Evaluator
from utils import load_config, save_json


def evaluate(data_loader, model, evaluator, **kwargs):

    return_scores_by_sample = kwargs['return_scores_by_sample'] if 'return_scores_by_sample' in kwargs else False
    return_answers = kwargs['return_answers'] if 'return_answers' in kwargs else False

    if return_scores_by_sample:
        total_accuracies = []
        total_anls = []

    else:
        total_accuracies = 0
        total_anls = 0

    answers = []
    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        questions = batch['questions']
        contexts = batch['contexts']
        gt_answers = batch['answers']

        with torch.no_grad():
            outputs, pred_answers = model.forward(questions, contexts, return_pred_answer=True)
            # print(pred_answers)

        metric = evaluator.get_metrics(gt_answers, pred_answers)

        if return_scores_by_sample:
            total_accuracies.extend(metric['accuracy'])
            total_anls.extend(metric['anls'])
        else:
            total_accuracies += sum(metric['accuracy'])
            total_anls += sum(metric['anls'])

        if return_answers:
            answers.extend(pred_answers)

    if not return_scores_by_sample:
        total_accuracies = total_accuracies/len(data_loader.dataset)
        total_anls = total_anls/len(data_loader.dataset)

    return total_accuracies, total_anls, answers


if __name__ == '__main__':
    config = load_config("configs/longformer.yml")

    dataset = SingleDocVQA(config['imdb_dir'], split='val')
    val_data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=singledocvqa_collate_fn)
    longformer_model = Longformer(config)
    longformer_model.model.to(config['device'])

    evaluator = Evaluator(case_sensitive=False)
    accuracy, anls, answers = evaluate(val_data_loader, longformer_model, evaluator, return_scores_by_sample=True, return_answers=True)

    save_data = {
        "Model": config["Model"],
        "Mean accuracy": np.mean(accuracy),
        "Mean ANLS": np.mean(anls),
        "Sample_accuracy": accuracy,
        "Sample_anls": anls,
        "Answers": answers,
    }

    save_json("{:s}/{:}_results.json".format(config['save_dir'], save_data['Model']), save_data)

