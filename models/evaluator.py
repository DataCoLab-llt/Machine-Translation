from utils import read_text_file, shuffle_dataset
from preprocessing import Preprocessing


class Evaluate:
    def __init__(self, path):
        self.data = read_text_file(path)

    def prepare_data(self):
        pre_obj = Preprocessing()
        main_data = shuffle_dataset(pre_obj.clean(self.data, 1000, 30))
        return main_data

