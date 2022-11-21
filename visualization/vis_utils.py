import argparse, warnings
from PIL import Image
from utils import parse_multitype2list_arg


head_colors = ['#6E0065', '#1F1FFF', '#4E2CD2', '#7D38A5', '#DB504A', '#EB8E2D', '#FACB0F', '#07A0C3', '#33A78D', '#5FAD56', '#7E8080', '#222222']


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
    parser.add_argument('-d', '--decoder', action='store_true', help='Plot decoder attentions.')
    parser.add_argument('-c', '--cross',   action='store_true', help='Plot cross attentions.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print a message for each attention plotted.')

    parser.add_argument('--page_idx', type=str, help='Plot the attention of specific pages.')
    parser.add_argument('--att_layers', type=str, help='Plot the attention of specific layers.')
    parser.add_argument('--att_heads', type=str, help='Plot the attention of specific heads.')
    parser.add_argument('--token_idx', type=str, help='Plot the attention of specific tokens. Only used in plot_enc_attention_img.')

    parser.add_argument('--format', type=str, help='Format of plots (jpg, png, pdf).', default='.jpg')
    parser.add_argument('--parallel-plot', action='store_true', help='Parallelize the plotting of encoder attentions.')

    args = parser.parse_args()
    assert args.att_file or (args.model and args.dataset), 'You must provide either with the attention file to plot, or the model and dataset to get the attentions.'

    args.page_idx = parse_multitype2list_arg(args.page_idx)
    args.att_layers = parse_multitype2list_arg(args.att_layers)
    args.att_heads = parse_multitype2list_arg(args.att_heads)

    return args


def get_plot_format(plt_format):
    if plt_format.strip('.') not in ['png', 'jpg', 'pdf']:
        new_plt_format = '.' + plt_format if '.' not in plt_format else plt_format
        warnings.warn(f"Plot format '{plt_format}' not between the supported ones. Using '{new_plt_format}' doing our best.")

    else:
        new_plt_format = '.' + plt_format if '.' not in plt_format else plt_format

    return new_plt_format


def get_plot_ranges(att_layers, att_heads, args, pages=None):

    if args.att_layers:
        assert att_layers >= max(args.att_layers), f'Attention only has {att_layers} layers, while you specified attention layers: {args.att_layers}'
        att_layers = args.att_layers
    else:
        att_layers = list(range(att_layers))

    if args.att_heads:
        assert att_heads >= max(args.att_heads), f'Attention only has {att_heads} layers, while you specified attention heads: {args.att_heads}'
        att_heads = args.att_heads
    else:
        att_heads = list(range(att_heads))

    if pages is not None:
        if args.page_idx:
            assert pages >= max(args.page_idx), f'Attention only has {pages} pages, but you specified: {args.page_idx} pages'
            pages = args.page_idx
        else:
            pages = list(range(pages))

        return pages, att_layers, att_heads
    else:
        return att_layers, att_heads


def get_concat_h_resize(im1, im2, h_margin=0, resample=Image.BICUBIC, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width + h_margin, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width + h_margin, 0))
    return dst
