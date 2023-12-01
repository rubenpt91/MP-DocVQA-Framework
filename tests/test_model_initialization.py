import unittest
import torch
from utils import load_config
from build_utils import build_model
from tests.test_utils import get_mock_args, set_test_config


def test_init_model(unittest):
    config = set_test_config(load_config(unittest.args))
    model = build_model(config)
    model.model.to(config.device)
    print("\tModel {:s} Initialization Test passed.\n".format(unittest.args.model))


class TestInitModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = get_mock_args()
        cls.args.dataset = 'SP-DocVQA'

    def test_init_bert(self):
        self.args.model = 'BertQA'
        test_init_model(self)

    def test_init_longformer(self):
        self.args.model = 'Longformer'
        test_init_model(self)

    def test_init_bigbird(self):
        self.args.model = 'BigBird'
        test_init_model(self)

    def test_init_layoutlmv2(self):
        self.args.model = 'LayoutLMv2'
        test_init_model(self)

    def test_init_layoutlmv3(self):
        self.args.model = 'LayoutLMv3'
        test_init_model(self)

    def test_init_t5(self):
        self.args.model = 'T5'
        test_init_model(self)

    def test_init_longt5(self):
        self.args.model = 'LongT5'
        test_init_model(self)

    def test_init_vt5(self):
        self.args.model = 'VT5'
        test_init_model(self)

    def test_init_hivt5(self):
        self.args.model = 'HiVT5'
        test_init_model(self)

    def test_init_wrong(self):
        self.args.model = 'QWERTYUIOPLKJHGFDSAZXCVBNM'
        with self.assertRaises(FileNotFoundError):
            test_init_model(self)


if __name__ == '__main__':
    with torch.no_grad():
        unittest.main()
