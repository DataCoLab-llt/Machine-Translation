import tensorflow as tf
from preprocessing import create_dataset
from sklearn.model_selection import train_test_split

path_to_file = '/media/moslem/Private/Project/MTP/Machine-Translation/dataset/manythings/pes.txt'


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


def train_tensor():
    num_examples = 30000
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    print("Input Language; index to word mapping")
    convert(inp_lang, input_tensor_train[0])
    print("Target Language; index to word mapping")
    convert(targ_lang, target_tensor_train[0])


train_tensor()
