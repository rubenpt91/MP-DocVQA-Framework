import random
import torch
from bertviz import model_view

from visualization.vis_utils import parse_vis_args
from utils import load_config
from build_utils import build_dataset, build_model
from metrics import Evaluator

from transformers import AutoTokenizer, AutoModel, utils


def forward_record(model, record):
    fake_batch = {k: [v] for k, v in record.items()}
    return model.forward(fake_batch, return_pred_answer=True)


def get_attention_data(args):

    config = load_config(args)
    evaluator = Evaluator(case_sensitive=False)

    dataset = build_dataset(config, args.dataset_split)
    dataset.get_doc_id = True
    record = dataset.sample(question_id=question_id)

    model = build_model(config)
    outputs, pred_answer, pred_answer_page, attention_dict = model.forward_record_and_retrieve_attention_dict(record)

    answering_metrics = evaluator.get_metrics([record['answers']], pred_answer)
    retrieval_metrics = evaluator.get_retrieval_metric([record['answer_page_idx']], pred_answer_page)
    acc = answering_metrics['accuracy'][0]
    anls = answering_metrics['anls'][0]
    ret_prec = retrieval_metrics[0]

    attention_dict = {
        "Model": config["model_name"],
        "Model_weights": config["model_weights"],
        "Dataset": config["dataset_name"],
        "Page retrieval": config.get('page_retrieval', '-').capitalize(),
        "Pred. Answer": pred_answer,
        "Pred. Answer Page": pred_answer_page,
        "Accuracy": acc,
        "ANLS": anls,
        "Ret. Precision": ret_prec,
        "question_id": record['question_id'],
        "doc_id": record['doc_id'],
        "question": record['questions'],
        "GT Answers": record['answers'],
        "encoder_att": attention_dict['encoder_att'],
        "decoder_att": attention_dict['decoder_att'],
        "cross_att": attention_dict['cross_att'],
        "encoder_text": attention_dict['encoder_text'],
        "encoder_boxes": attention_dict['encoder_boxes'],
        "answer_text": attention_dict['answer_text'],
        "decoder_input_text": attention_dict['decoder_input_text'],
        "config": config,
    }

    return attention_dict


if __name__ == '__main__':

    args = parse_vis_args()
    # config = load_config(args)

    question_id = args.question_id

    """
    # model_name = "microsoft/xtremedistil-l12-h384-uncased"  # Find popular HuggingFace models here: https://huggingface.co/models
    # input_text = "The cat sat on the mat"
    # model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
    # outputs = model(inputs)  # Run model
    # attention = outputs[-1]  # Retrieve attention from model outputs
    # tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
    # html_model_view = model_view(attention, tokens, html_action='return')  # Display model view

    # 16401 What is the date when the approval form was filled?
    # 16404 What is the title of William M. Coleman?
    # 61876 What is the total costs for proposed project period?
    # 16384 betty royal
    # 57346 5%,
    """
    # dataset = build_dataset(config, args.dataset_split)
    # dataset.get_doc_id = True
    # record = dataset.sample(question_id=question_id)

    """
    # 54854, 54855, 54856 visual document
    # record = dataset.sample(idx=100)
    # 49340
    """
    # model = build_model(config)
    # outputs, pred_answer, pred_answer_page, attention_dict = model.forward_record_and_retrieve_attention_dict(record)

    # answering_metrics = evaluator.get_metrics([record['answers']], pred_answer)
    # retrieval_metrics = evaluator.get_retrieval_metric([record['answer_page_idx']], pred_answer_page)
    # acc = answering_metrics['accuracy'][0]
    # anls = answering_metrics['anls'][0]
    # ret_prec = retrieval_metrics[0]

    save_dict = get_attention_data(args)

    viz_attention_name = "save/attention_viz/Attention_{:s}__{:s}_{:d}".format(save_dict['Model'], save_dict['Dataset'], save_dict['question_id'])
    torch.save(save_dict, viz_attention_name)

    print("Attentions correctly saved in: {:s}".format(viz_attention_name))
