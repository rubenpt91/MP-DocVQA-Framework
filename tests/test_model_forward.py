import unittest
import torch

from utils import load_config
from test_utils import get_mock_args, set_test_config

from build_utils import build_dataset, build_model
from torch.utils.data import DataLoader
from datasets.dataset_utils import docvqa_collate_fn


def print_answers_func(gt_answers, pred_answers):
    if gt_answers is not None:
        print("".join(["\t\tGT: {:} \tPred: {:}\n".format(gt_answer, pred_answer) for gt_answer, pred_answer in zip(gt_answers, pred_answers)]))
    else:
        print("".join(["\t\tGT: {:} \tPred: {:}\n".format(None, pred_answer) for pred_answer in pred_answers]))


def run_forward(model, config, print_answers):
    for split in ["train", "val", "test"]:
        dataset = build_dataset(config, split)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=docvqa_collate_fn)
        batch = next(iter(data_loader))
        outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)
        if print_answers:
            print_answers_func(batch['answers'], pred_answers)


def test_forward_on_spdocvqa(unit_test, model):
    model_name = unit_test.args.model
    print("\tTest {:s} on SP-DocVQA:".format(model_name))
    unit_test.args.dataset = 'SP-DocVQA'

    config = set_test_config(load_config(unit_test.args))
    model.model.to(config.device)
    unit_test.assertTrue(model.page_retrieval == "none")
    run_forward(model, config, unit_test.print_answers)

    print("\tTest {:s} on SP-DocVQA passed.\n".format(model_name))


def test_forward_on_mpdocvqa(unit_test, model):
    model_name = unit_test.args.model
    print("\tTest {:} on MP-DocVQA:".format(model_name))
    unit_test.args.page_retrieval = None
    unit_test.args.dataset = 'MP-DocVQA'

    print("\t\t- Oracle")
    unit_test.args.page_retrieval = "oracle"
    config = set_test_config(load_config(unit_test.args))
    model.page_retrieval = "oracle"
    run_forward(model, config, unit_test.print_answers)

    print("\t\t- Max Conf.")
    unit_test.args.page_retrieval = "logits"
    if not unit_test.is_hierarchical_method:
        config = set_test_config(load_config(unit_test.args))
        model.page_retrieval = "logits"
        run_forward(model, config, unit_test.print_answers)

    else:
        with unit_test.assertRaises(ValueError):
            load_config(unit_test.args)

    print("\t\t- Concat.")
    unit_test.args.page_retrieval = "concat"
    if not unit_test.is_hierarchical_method:
        config = set_test_config(load_config(unit_test.args))
        model.page_retrieval = "concat"
        run_forward(model, config, unit_test.print_answers)
    else:
        with unit_test.assertRaises(ValueError):
            load_config(unit_test.args)

    if not unit_test.is_hierarchical_method:
        unit_test.args.page_retrieval = "custom"
        with unit_test.assertRaises(ValueError):
            load_config(unit_test.args)
    else:
        print("\t\t- Custom")
        unit_test.args.page_retrieval = "custom"
        config = set_test_config(load_config(unit_test.args))
        model.page_retrieval = "custom"
        run_forward(model, config, unit_test.print_answers)

    print("\tTest {:s} on MP-DocVQA passed.\n".format(model_name))


def test_forward_on_dude(unit_test, model):
    model_name = unit_test.args.model
    print("\tTest {:s} on DUDE:".format(model_name))
    unit_test.args.page_retrieval = None
    unit_test.args.dataset = 'DUDE'

    print("\t\t- Oracle")
    unit_test.args.page_retrieval = "oracle"
    with unit_test.assertRaises(ValueError):
        load_config(unit_test.args)
        # Oracle set-up is not valid for DUDE, since there is no GT for the answer page.

    print("\t\t- Max Conf.")
    if not unit_test.is_hierarchical_method:
        unit_test.args.page_retrieval = "logits"
        config = set_test_config(load_config(unit_test.args))
        model.page_retrieval = "logits"
        run_forward(model, config, unit_test.print_answers)

    else:
        with unit_test.assertRaises(ValueError):
            load_config(unit_test.args)

    print("\t\t- Concat.")
    if not unit_test.is_hierarchical_method:
        unit_test.args.page_retrieval = "concat"
        config = set_test_config(load_config(unit_test.args))
        model.page_retrieval = "concat"
        run_forward(model, config, unit_test.print_answers)

    else:
        with unit_test.assertRaises(ValueError):
            load_config(unit_test.args)

    print("\t\t- Custom")
    unit_test.args.page_retrieval = "custom"
    if not unit_test.is_hierarchical_method:
        with unit_test.assertRaises(ValueError):
            load_config(unit_test.args)
    else:
        config = set_test_config(load_config(unit_test.args))
        model.page_retrieval = "custom"
        run_forward(model, config, unit_test.print_answers)

    print("\tTest {:} on DUDE passed.\n".format(model_name))


def test_forward(unit_test, model_name):
    print("Testing Forward {:s}...".format(model_name))

    unit_test.args.page_retrieval = None
    unit_test.args.model = model_name
    unit_test.args.dataset = 'SP-DocVQA'

    config = set_test_config(load_config(unit_test.args))
    model = build_model(config)
    model.model.to(config.device)

    test_forward_on_spdocvqa(unit_test, model)
    test_forward_on_mpdocvqa(unit_test, model)
    test_forward_on_dude(unit_test, model)


class TestModelForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = get_mock_args()
        cls.args.model = 'BertQA'
        cls.print_answers = True


    def test_forward_bert(self):
        self.is_hierarchical_method = False
        test_forward(self, "BertQA")
        
    def test_forward_longformer(self):
        self.is_hierarchical_method = False
        test_forward(self, "Longformer")

    def test_forward_bigbird(self):
        self.is_hierarchical_method = False
        test_forward(self, "BigBird")

    def test_forward_t5(self):
        self.is_hierarchical_method = False
        test_forward(self, "T5")

    def test_forward_longt5(self):
        self.is_hierarchical_method = False
        test_forward(self, "LongT5")

    def test_forward_vt5(self):
        self.is_hierarchical_method = False
        test_forward(self, "VT5")

    def test_forward_layoutlmv2(self):
        self.is_hierarchical_method = False
        test_forward(self, "LayoutLMv2")

    def test_forward_layoutlmv3(self):
        self.is_hierarchical_method = False
        test_forward(self, "LayoutLMv3")

    def test_forward_hivt5(self):
        self.is_hierarchical_method = True
        test_forward(self, "HiVT5")


if __name__ == '__main__':
    with torch.no_grad():
        unittest.main()
