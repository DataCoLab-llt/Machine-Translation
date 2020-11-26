from nltk.translate.bleu_score import sentence_bleu


def bleu_evaluation(prediction, truth, verbose=False):
    """
    evaluating a translation sentence against one or more reference sentences.
    Args:
        prediction (string): translated sentence
        truth (list): list of reference sentences
        verbose (boolean): to print or not to print BLEU scores
    Returns:
        nothings for now, just print BLEU scores
    Example:
        truth = [['i love cats'], ['i love hats']]
        candidate = 'i love cat'
        bleu_evaluation(candidate, truth)
    """
    print('translations=[%s], references=%s' % (prediction, truth))
    prediction_token = prediction.split()
    truth_token = list()
    for sen in prediction:
        truth_token.append(sen[0].split())

    if verbose:
        print('BLEU, Individual 1-gram: %f' % sentence_bleu(truth_token, prediction_token, weights=(1, 0, 0, 0)))
        print('BLEU, Individual 2-gram: %f' % sentence_bleu(truth_token, prediction_token, weights=(0, 1, 0, 0)))
        print('BLEU, Individual 3-gram: %f' % sentence_bleu(truth_token, prediction_token, weights=(0, 0, 1, 0)))
        print('BLEU, Individual 4-gram: %f' % sentence_bleu(truth_token, prediction_token, weights=(0, 0, 0, 1)))


