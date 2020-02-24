from collections import Counter
import codecs


## helper code
def read_conll_file(file_name):
    """
    read in conll file
    
    :param file_name: path to read from
    :return: list of pairs of words/tags
    """
    data = []
    current_words = []
    current_tags = []

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()
        
        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split('\t')
            if '-' in tok[0] or '.' in tok[0]:
                continue # skip special tokenized words
            word = tok[1]
            tag = tok[3]
            
            current_words.append(word)
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != [] and not raw:
        data.append((current_words, current_tags))
    return data

def evaluate(golds, preds):
    """
    evaluate model on a test file in CoNLL format

    :param golds: list of lists of tags
    :param preds: list of lists of tags
    :return: (sentence accuracy, word accuracy)
    """
    correct_words = 0
    correct_sentences = 0

    words_total = 0.0
    sentences_total = 0.0

    for gold, pred in zip(golds, preds):
        # check whether entire tag sequence was correct
        sentences_total += 1
        if pred == gold:
            correct_sentences += 1

        # check individual tags for correctness
        for predicted_tag, gold_tag in zip(pred, gold):
            words_total += 1
            if predicted_tag == gold_tag:
                correct_words += 1

    return (correct_sentences/sentences_total, correct_words/words_total)


