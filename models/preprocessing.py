from hazm import word_tokenize
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import utils

class Preprocessing:
    """
    this class is provided to do text preprocessing like cleaning and normalizing
    """

    def __init__(self):
        pass

    def clean(self, text, count, sent_size):
        lines = list()
        sents = text.strip().split('\n')
        for i in range(2 * count):
            english, persian = sents[i].rstrip().split('\t', 1)
            persian = persian.rstrip().split('\t', 1)[0]
            persian_tokens = word_tokenize(persian)
            if len(english) <= sent_size and len(persian_tokens) <= sent_size:
                lines.append([english, ' '.join(persian_tokens)])
        return lines

    def create_tokenizer(self, data):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(data)
        return tokenizer

    def get_tokenizer(self, data):
        tokenizer = self.create_tokenizer(data)
        vocab_size = len(tokenizer.word_index) + 1
        length = utils.get_max_length(data)
        print('Vocabulary Size: {}'.format(vocab_size))
        print('Max Length: {}'.format(length))
        return tokenizer, vocab_size, length

    def encode_sequences(self, tokenizer, length, lines):
        return pad_sequences(tokenizer.texts_to_sequences(lines), maxlen=length, padding='post')
