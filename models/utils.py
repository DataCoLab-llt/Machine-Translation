import codecs
import numpy as np
from sklearn.utils import shuffle


def read_text_file(path_to_file):
    """
           read text file
        Args:
           path_to_file (string): path to text file
        Returns:
           string output
    """
    with codecs.open(path_to_file, 'r', encoding="utf-8") as file:
        return file.read()


def shuffle_dataset(data):
    """
        shuffle data
    Args:
        data (list): list of data
    Returns:
        numpy array output
    """
    data = np.array(data)
    data = shuffle(data)
    return data


def get_max_length(items):
    """
        get maximum list items and return length
    Args:
        items (list): list of data
    Returns:
        int output
    """
    return max(len(item.split()) for item in items)
