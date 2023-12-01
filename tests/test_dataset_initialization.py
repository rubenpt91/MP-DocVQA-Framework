import unittest
import torch
from utils import load_config
from build_utils import build_dataset
from tests.test_utils import get_mock_args, set_test_config


def test_init_dataset(unittest):
    config = set_test_config(load_config(unittest.args))
    dataset = build_dataset(config, 'train')
    dataset = build_dataset(config, 'val')
    dataset = build_dataset(config, 'test')
    print("\tDataset {:s} Initialization Test passed.\n".format(unittest.args.dataset))


class TestInitDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = get_mock_args()
        cls.args.model = 'BertQA'

    def test_init_spdocvqa(self):
        self.args.dataset = 'SP-DocVQA'
        test_init_dataset(self)

    def test_init_mpdocvqa(self):
        self.args.dataset = 'MP-DocVQA'
        test_init_dataset(self)

    def test_init_dude(self):
        self.args.dataset = 'DUDE'
        test_init_dataset(self)


if __name__ == '__main__':
    with torch.no_grad():
        unittest.main()
