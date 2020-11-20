from nltk.translate.bleu_score import sentence_bleu


def evaluation(translations, references):
    """
    evaluating a translation sentence against one or more reference sentences.
    Args:
        translations (string): translated sentence
        references (list): list of reference sentences
    Returns:
        nothings for now, just print BLEU scores
    """
    print('translations=[%s], references=%s' % (translations, references))
    trans_token = translations.split()
    refer_token = list()
    for sen in references:
        refer_token.append(sen[0].split())

    print('Individual 1-gram: %f' % sentence_bleu(refer_token, trans_token, weights=(1, 0, 0, 0)))
    print('Individual 2-gram: %f' % sentence_bleu(refer_token, trans_token, weights=(0, 1, 0, 0)))
    print('Individual 3-gram: %f' % sentence_bleu(refer_token, trans_token, weights=(0, 0, 1, 0)))
    print('Individual 4-gram: %f' % sentence_bleu(refer_token, trans_token, weights=(0, 0, 0, 1)))


# reference = [['i love cats'], ['i love hats']]
# candidate = 'i love cat'
# evaluation(candidate, reference)
