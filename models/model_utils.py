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


def convert(lang_tokenizer, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang_tokenizer.index_word[t]))


def create_tf_dataset(input_tensor_train, target_tensor_train):
    buffer_size = len(input_tensor_train)
    batch_size = 64
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def main_proccess():
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


main_proccess()
