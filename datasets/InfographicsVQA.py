from datasets.SP_DocVQA import SPDocVQA


class InfographicsVQA(SPDocVQA):

    def __init__(self, imbd_dir, images_dir, split, kwargs):
        super(InfographicsVQA, self).__init__(imbd_dir, images_dir, split, kwargs)
