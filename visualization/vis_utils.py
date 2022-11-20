import ast
import argparse

from utils import parse_multitype2list_arg

def parse_vis_args():
    parser = argparse.ArgumentParser(description='Baselines for MP-DocVQA')

    # Required
    # parser.add_argument('-m', '--model', type=str, required=True, help='Path to yml file with model configuration.')
    # parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to yml file with dataset configuration.')
    parser.add_argument('-f', '--att_file', type=str, help='File with the attentions to plot.')

    # Optional
    parser.add_argument('--model', type=str, help='Path to yml file with model configuration.')
    parser.add_argument('--dataset', type=str, help='Path to yml file with dataset configuration.')
    parser.add_argument('--question_id', type=int, help='Question ID to plot')

    parser.add_argument('-e', '--encoder', action='store_true', help='Plot encoder attentions.')
    parser.add_argument('-d', '--decoder', action='store_true', help='Plot encoder attentions.')
    parser.add_argument('-c', '--cross',   action='store_true', help='Plot encoder attentions.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print each attention plotted.')

    parser.add_argument('--page_idx', type=str, help='Plot encoder attentions.')
    parser.add_argument('--att_layers', type=str, help='Plot encoder attentions.')
    parser.add_argument('--att_heads', type=str, help='Plot encoder attentions.')

    parser.add_argument('--parallel-plot', action='store_true', help='Parallelize the plotting of encoder attentions.')

    args = parser.parse_args()
    assert args.att_file or (args.model and args.dataset), 'You must provide either with the attention file to plot, or the model and dataset to get the attentions.'

    args.page_idx = parse_multitype2list_arg(args.page_idx)
    args.att_layers = parse_multitype2list_arg(args.att_layers)
    args.att_heads = parse_multitype2list_arg(args.att_heads)

    return args
