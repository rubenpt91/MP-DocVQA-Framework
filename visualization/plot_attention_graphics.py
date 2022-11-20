import os
import torch
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from visualization.vis_utils import parse_vis_args


head_colors = ['#6E0065', '#1F1FFF', '#4E2CD2', '#7D38A5', '#DB504A', '#EB8E2D', '#FACB0F', '#07A0C3', '#33A78D', '#5FAD56', '#7E8080', '#222222']


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


def plot_attentions(attentions, x, y, figure_dir, verbose, args):
    att_layers = len(attentions)
    att_heads = attentions[0].shape[1]
    att_layers, att_heads = get_plot_ranges(att_layers, att_heads, args)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    for att_layer in att_layers:
        for att_head in att_heads:

            if verbose:
                print(f"LAYER: {att_layer}   HEAD: {att_head}")

            fig_color = head_colors[att_head]
            att = attentions[att_layer].squeeze(dim=0)[att_head]
            fig = create_plot(x, y, att, fig_color)
            plt.savefig(os.path.join(figure_dir, f'layer_{att_layer}__head_{att_head}.jpg'))
            plt.close()

        if verbose:
            print('')


def plot_encoder_hierarchical_attentions(attentions, page_tokens, input_text, figure_dir, verbose, args):
    pages = len(attentions)
    att_layers = len(attentions[0])
    att_heads = attentions[0][0].shape[1]
    pages, att_layers, att_heads = get_plot_ranges(att_layers, att_heads, args, pages)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    for page_idx in pages:
        resumed_input_text = input_text[page_idx][:page_tokens]

        for att_layer in att_layers:
            for att_head in att_heads:

                if verbose:
                    print(f"PAGE: {page_idx}   LAYER: {att_layer}   HEAD: {att_head}")

                fig_color = head_colors[att_head]
                att = attentions[page_idx][att_layer].squeeze(dim=0)[att_head]

                resumed_att = att[:page_tokens]
                unpadded_text = input_text[page_idx][:att.shape[1]]

                fig = create_plot(resumed_input_text, unpadded_text, resumed_att, fig_color)
                plt.savefig(os.path.join(figure_dir, f'page_{page_idx}__layer_{att_layer}__head_{att_head}.jpg'))
                plt.close()

            if verbose:
                print('')

        if verbose:
            print('')


def parallel_plot_encoder_attentions(page_att, page_idx, page_tokens, input_text, figure_dir, args):
    att_layers = len(page_att)
    att_heads = page_att[0].shape[1]
    att_layers, att_heads = get_plot_ranges(att_layers, att_heads, args)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    resumed_input_text = input_text[page_idx][:page_tokens]
    for att_layer in att_layers:
        for att_head in att_heads:
            fig_color = head_colors[att_head]

            att = page_att[att_layer].squeeze(dim=0)[att_head]
            resumed_att = att[:page_tokens]
            fig = create_plot(resumed_input_text, input_text[page_idx], resumed_att, fig_color)

            # resumed_att = att[:page_tokens, :page_tokens]
            # fig = plot_attentions(resumed_input_text, resumed_input_text, resumed_att, fig_color)

            plt.savefig(os.path.join(figure_dir, f'page_{page_idx}__layer_{att_layer}__head_{att_head}.jpg'))
            plt.close()


def create_plot(text_V, text_KQ, att, color):
    num_V_tok = len(text_V)
    num_KQ_tok = len(text_KQ)
    fig, axs = plt.subplots(len(att),1, figsize=(max(num_KQ_tok/4, 20), num_V_tok*4))

    for V_tok_idx in range(num_V_tok):
        V_tok = text_V[V_tok_idx]
        axs[V_tok_idx].set_title(V_tok, loc='left')

        labels_pos = range(num_KQ_tok)
        axs[V_tok_idx].bar(labels_pos, att[V_tok_idx], color=[color]) # , metric_dict[method], width, label=method.split('_')[0])

        axs[V_tok_idx].set_xticks(labels_pos)
        axs[V_tok_idx].set_xticklabels(text_KQ, rotation=90)

        axs[V_tok_idx].set_xlim(-0.75, max(labels_pos) + 0.75)

    fig.tight_layout()
    return fig


if __name__ == '__main__':

    args = parse_vis_args()
    verbose = args.verbose

    if args.att_file is None:
        question_id = 10
        att_filename = 'Attention_{:s}__MP-DocVQA_{:d}'.format(args.model, question_id)

        extract_attentions

    else:
        attention_data = torch.load(args.att_file)
        att_filename = args.att_file.split()[-1]

    page_tokens = attention_data['config']['page_tokens']

    encoder_att = attention_data['encoder_att']
    decoder_att = attention_data['decoder_att']
    cross_att = attention_data['cross_att']

    input_text = attention_data['encoder_text']
    answer_text = attention_data['answer_text']
    decoder_input_text = attention_data['decoder_input_text']

    if args.cross:
        figure_dir = os.path.join('save', 'attention_viz', 'plots', att_filename, 'cross/')
        plot_attentions(cross_att, answer_text, decoder_input_text, figure_dir, verbose, args)
        print(f'Cross attentions plotted in {figure_dir}\n')

    if args.decoder:
        figure_dir = os.path.join('save', 'attention_viz', 'plots', att_filename, 'decoder/')
        plot_attentions(decoder_att, answer_text, answer_text, figure_dir, verbose, args)
        print(f'Decoder attentions plotted in {figure_dir}\n')

    if args.encoder:

        figure_dir = os.path.join('save', 'attention_viz', 'plots', att_filename, 'encoder/')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        if not args.parallel_plot:
            plot_encoder_hierarchical_attentions(encoder_att, page_tokens, input_text, figure_dir, verbose, args)

        else:
            pages = args.page_idx if args.page_idx else list(range(len(encoder_att)))
            cpu_jobs = min(max(1, os.cpu_count()-1), len(pages))
            print(f'Plotting pages {pages} encoder attentions in {cpu_jobs} jobs...')
            Parallel(n_jobs=cpu_jobs)(delayed(parallel_plot_encoder_attentions)(encoder_att[page_idx], page_idx, page_tokens, input_text, figure_dir, args) for page_idx in pages)

        print(f'Encoder attentions plotted in {figure_dir}\n')

    print("Plotting finished")
