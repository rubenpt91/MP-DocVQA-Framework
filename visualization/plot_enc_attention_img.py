import os
import torch
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visualization.vis_utils import parse_vis_args, get_plot_format, get_plot_ranges, get_concat_h_resize
from utils import load_config, parse_multitype2list_arg
from build_utils import build_dataset


def resize_image(img, base_width=512):
    wpercent = (base_width/float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.ANTIALIAS)

    return img


def add_margin(img, top=0, right=0, bottom=0, left=0, color=(255, 255, 255)):
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result


def get_pretext(encoder_text):
    pretext = []
    context_found = False
    for word in encoder_text:
        pretext.append(word)

        if 'context' in word:
            context_found = True

        if context_found and word == ':':
            break

    num_lines = int(len(pretext)/10)+1
    return pretext, num_lines


def draw_pretext_att(img, pretext, lines_pretext, words_per_line, att_list, font):
    img = add_margin(img, top=lines_pretext * 25)

    draw = ImageDraw.Draw(img, 'RGBA')

    pretext_draw_boxes = []
    for line_ix in range(lines_pretext):
        for word_ix, word in enumerate(pretext[line_ix * words_per_line: line_ix * words_per_line + words_per_line]):
            x = 5 + word_ix * 55
            y = 5 + line_ix * 20
            draw.text((x, y), '_' if word == '▁' else word.strip('▁'), (0, 0, 0), font=font)

            extra_att = 255
            # extra_att = 255
            att = int(125 * att_list[word_ix + line_ix * words_per_line])
            if att > 125:
                # extra_att = att-255
                extra_att = extra_att - (att - 255)
                att = 125

            if word == '▁date':
                # draw.rectangle([(x,y), (x+55, y+15)], fill=(255, 220, 0, 100))  # Font size 14
                draw.rectangle([(x, y), (x + 45, y + 15)], fill=(255, 220, 0, 100))  # Font size 12

            else:
                # draw.rectangle([(x,y), (x+55, y+15)], fill=(255, extra_att, 0, att)) # Font size 14
                draw.rectangle([(x, y), (x + 45, y + 15)], fill=(255, extra_att, 0, att))  # Font size 12

    return img


def draw_ocr_attentions(img, text, boxes, att_list, len_pretext, num_visual_tokens=197):
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img, 'RGBA')
    accumulated_attentions = []
    for idx, [word, box, att] in enumerate(zip(text[len_pretext:-num_visual_tokens], boxes[len_pretext:-num_visual_tokens], att_list[len_pretext:-num_visual_tokens])):
        # print(word, box, att.item())

        accumulated_attentions.append(att.item())
        if all(encoder_boxes[len(pretext) + idx] == encoder_boxes[len(pretext) + idx + 1]):
            continue

        extra_att = 255
        avg_att = int((255 / 4) * np.mean(accumulated_attentions))
        if avg_att > 255:
            extra_att = extra_att - (avg_att - 255)
            avg_att = 125

        x1 = box[0] * img_w
        y1 = box[1] * img_h
        x2 = box[2] * img_w
        y2 = box[3] * img_h

        draw.rectangle([(x1, y1), (x2, y2)], fill=(255, extra_att, 0, avg_att))
        accumulated_attentions = []

    return img


def draw_vis_attentions(img, att_list, num_visual_tokens=197):
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img, 'RGBA')
    for idx, [word, box, att] in enumerate(zip(encoder_text[-num_visual_tokens:], encoder_boxes[-num_visual_tokens:], att_list[-num_visual_tokens:])):
        x1 = box[0] * img_w
        y1 = box[1] * img_h
        x2 = box[2] * img_w
        y2 = box[3] * img_h

        extra_att = 0
        if att > 125:
            extra_att = att - 255
            att = 125

        draw.rectangle([(x1, y1), (x2, y2)], fill=(255, extra_att, 0, int(255 * att)))

    return img


def scale_attention(att):
    att = att * 709 * 0.5

    """
    new_att = plot_encoder_att * 709 * 0.5
    new_att_normalized = (new_att - min(new_att)) / (max(new_att) - min(new_att))
    print(new_att.mean())
    print(new_att_normalized.mean())
    plot_encoder_att_prime = new_att
    # plot_encoder_att_prime = vn[plot_token_idx]
    """
    return att


if __name__ == '__main__':

    args = parse_vis_args()
    config = load_config(args)
    verbose = args.verbose

    dataset = build_dataset(config, split='test')
    words_per_line = 10
    text_font = ImageFont.truetype('visualization/fonts/Arial.ttf', size=14)
    plot_file_format = get_plot_format(args.format)
    if plot_file_format.strip('.') not in ['png', 'jpg', 'pdf']:
        new_format = '.' + plot_file_format if '.' not in plot_file_format else plot_file_format
        warnings.warn(f'Plot format {plot_file_format} not between the supported ones. Using {new_format} doing our best.')

    if args.att_file is None:
        question_id = 10
        att_filename = 'Attention_{:s}__MP-DocVQA_{:d}'.format(args.model, question_id)

        extract_attentions

    else:
        attention_data = torch.load(args.att_file)
        att_filename = args.att_file.split()[-1]
        gt_record = dataset.sample(question_id=attention_data['question_id'])
        figure_dir = os.path.join('save', 'attention_viz', 'plots', att_filename, 'encoder_page/')

    page_tokens = attention_data['config']['page_tokens']

    encoder_att = attention_data['encoder_att']
    decoder_att = attention_data['decoder_att']
    cross_att = attention_data['cross_att']

    input_text = attention_data['encoder_text']
    answer_text = attention_data['answer_text']
    decoder_input_text = attention_data['decoder_input_text']

    pages, att_layers, att_heads = get_plot_ranges(pages=len(input_text), att_layers=len(encoder_att[0]), att_heads=encoder_att[0][0].shape[1], args=args)
    token_idxs = parse_multitype2list_arg(args.token_idx)

    num_plots = len(att_layers) * len(att_heads) * len(pages) * len(token_idxs)
    if num_plots > 10:
        warnings.warn(f"The number of plots ({num_plots}) is very big and might take a lot of time.\n"
                      f"Please, consider plotting a specific attention layer, head and token through the"
                      f" --att-layer, --att-head and --token_idx in the input arguments. Currently:"
                      f"\n\t{att_layers} attention layers, \n\t{att_heads} attention layers, \n\t{pages} pages\n\t{token_idxs} tokens.")

    for page_idx in pages:
        for att_layer in att_layers:
            for att_head in att_heads:
                image = resize_image(gt_record['images'][page_idx], base_width=600)

                encoder_text = attention_data['encoder_text'][page_idx]
                encoder_boxes = attention_data['encoder_boxes'][page_idx]
                encoder_att = attention_data['encoder_att'][page_idx]

                for plot_token_idx in token_idxs:
                    attentions = encoder_att[att_layer].squeeze(dim=0)[att_head][plot_token_idx]

                    pretext, pretext_lines = get_pretext(attention_data['encoder_text'][page_idx])
                    attentions = scale_attention(attentions)

                    text_att_img = draw_ocr_attentions(image.copy(), encoder_text, encoder_boxes, attentions, len_pretext=len(pretext))
                    text_att_img = draw_pretext_att(text_att_img, pretext, pretext_lines, words_per_line, attentions, text_font)

                    vis_att_img = draw_vis_attentions(image.copy(), attentions, num_visual_tokens=197)
                    vis_att_img = add_margin(vis_att_img, top=pretext_lines * 25)

                    att_results = get_concat_h_resize(text_att_img, vis_att_img)

                    if not os.path.exists(figure_dir):
                        os.makedirs(figure_dir)

                    plot_name = os.path.join(figure_dir, f'encoder_page_{page_idx}_layer_{att_layer}_head_{att_head}_token_{plot_token_idx}{plot_file_format}')
                    txt_plot_name = os.path.join(figure_dir, f'encoder_page_{page_idx}_layer_{att_layer}_head_{att_head}_token_{plot_token_idx}_txt{plot_file_format}')
                    vis_plot_name = os.path.join(figure_dir, f'encoder_page_{page_idx}_layer_{att_layer}_head_{att_head}_token_{plot_token_idx}_vis{plot_file_format}')

                    att_results.save(plot_name)
                    text_att_img.save(txt_plot_name)
                    vis_att_img.save(vis_plot_name)

                    if verbose:
                        print(f'Image saved at: {plot_name}')
