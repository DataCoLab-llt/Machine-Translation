import unicodedata
import re
import io


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.strip()
    if is_english(w):
        w = '<start> ' + w + ' <end>'
    else:
        w = '<end> ' + w + ' <start>'
    return w


def create_dataset(path, num_examples):
    clean_lines = list()
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    for i in lines:
        english, persian = i.rstrip().split('\t', 1)
        persian = persian.rstrip().split('\t', 1)[0]
        clean_lines.append(persian + '\t' + english)
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in clean_lines[:num_examples]]
    return zip(*word_pairs)
