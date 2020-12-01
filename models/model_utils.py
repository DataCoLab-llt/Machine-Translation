import tensorflow as tf
from preprocessing import create_dataset
from sklearn.model_selection import train_test_split


def tokenize(lang):
    """
    Create tokenizer according input,
    create a word index and reverse word index
    and pad each sentence to a maximum length.
    Args:
        lang (tuple): tuple of sentences for tokenizer
    Returns:
        tensor: numpy-ndarray for lang input
        lang_tokenizer: keras tokenizer object for lang input
    """
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    """
    Load dataset and prepare tensor and tokenizer for input and output language.
    Args:
        path (string): path to dataset
        num_examples (int or None): number of word pairs required
    Returns:
        return tensor and tokenizer for input and output language.
    """
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang_tokenizer, tensor):
    """
    Helper function for show word mapping(tensor to tokenizer index)
    Args:
        tensor (numpy array)
        lang_tokenizer (tokenizer object)
    Return:
        nothing to return
    """
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang_tokenizer.index_word[t]))


def main_process(path_to_file):
    """
    just for test, it's removed later.
    """
    num_examples = 30000
    input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset(path_to_file, num_examples)
    # Split arrays or matrices into random train and test subsets
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    print("Input Language; index to word mapping")
    convert(inp_lang_tokenizer, input_tensor_train[0])
    print("Target Language; index to word mapping")
    convert(targ_lang_tokenizer, target_tensor_train[0])
