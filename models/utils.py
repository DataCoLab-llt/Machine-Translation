import codecs
import numpy as np
from sklearn.utils import shuffle


def read_text_file(file_name):
    '''
           read text file
        Args:
           path(string): path to text file
        Returns:
           string output
    '''
    with codecs.open(file_name, 'r', encoding="utf-8") as file:
        return file.read()


def shuffle_dataset(data):
    data = np.array(data)
    data = shuffle(data)
    return data


def get_max_length(lines):
    return max(len(line.split()) for line in lines)
