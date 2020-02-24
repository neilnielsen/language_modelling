from hmm import HMM
import myutils

def main():
    # load data
    train_data = myutils.read_conll_file("data/da_ddt-ud-train.conllu")
    dev_data = myutils.read_conll_file("data/da_ddt-ud-dev.conllu")

    hmm = HMM()

    # fit model to training data
    hmm.fit(train_data)

    # get most likely tag predictions 
    most_likely_predictions = hmm.predict(dev_data, method='most_likely')
    viterbi_predictions = hmm.predict(dev_data, method='viterbi')

    # evaluate
    gold = [x[1] for x in dev_data]

    sent_level, word_level = myutils.evaluate(gold, most_likely_predictions)
    print('most likely scores:')
    print('sent level:  {:.4f}'.format(sent_level))
    print('word level:  {:.4f} \n'.format(word_level))

    sent_level, word_level = myutils.evaluate(gold, viterbi_predictions)
    print('viterbi scores:')
    print('sent level:  {:.4f}'.format(sent_level))
    print('word level:  {:.4f} \n'.format(word_level))

if __name__ == "__main__":
    main()