import unicodedata
import re
import io


def is_english(S):
    """
    Recognizes whether the input language of the sentence is English or not.
    Args:
        S (string): input sentence
    Returns:
        Boolean
    """
    try:
        S.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def unicode_to_ascii(S):
    """
    Convert unicode character to Ascii.
    Args:
        S (string): input sentence
    Returns:
        sentence (string): converted sentence to ascii
    """
    sentence = ''.join(c for c in unicodedata.normalize('NFD', S) if unicodedata.category(c) != 'Mn')
    return sentence


def preprocess_sentence(sentence):
    """
    Some preprocessing on sentence.
    Args:
        sentence (string): input sentence
    Returns:
        sentence (string) preprocessed sentence
    """
    sentence = unicode_to_ascii(sentence.lower().strip())
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.strip()
    if is_english(sentence):
        sentence = '<start> ' + sentence + ' <end>'
    else:
        sentence = '<end> ' + sentence + ' <start>'
    return sentence


def create_dataset(path, num_examples):
    """
    Some work on dataset and prepare cleaned dataset.
    Args:
        path (string): path to file - dataset file
        num_examples (int): number of word pairs required
    Returns:
        list of word pairs
    """
    clean_lines = list()
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    for i in lines:
        english, persian = i.rstrip().split('\t', 1)
        persian = persian.rstrip().split('\t', 1)[0]
        clean_lines.append(persian + '\t' + english)
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in clean_lines[:num_examples]]
    return zip(*word_pairs)
