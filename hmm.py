import numpy as np
from collections import defaultdict

class HMM(object):
    def __init__(self):
        """
        initialize model parameters
        :return:
        """
        self.START = '_START_'
        self.UNK = 'UNK'
        self.STOP = '_STOP_'
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        self.tags = set()
        self.token_tag_dist = defaultdict(lambda: defaultdict(float))

    def fit(self, train_data):
        """
        fit model to a file in CoNLL format.
        :param file_name:
        :return:
        """
        counts = defaultdict(int)

        # record all used tags and words
        for (words, tags) in train_data:
            
            # iterate over sentence
            for i, (word, tag) in enumerate(zip(words, tags)):
                self.tags.add(tag)
                self.vocabulary.add(word)
                counts[word] += 1
                
                # save tags to tokens for most frequent baseline, 
                # would normalization provide anything useful downstream?
                self.token_tag_dist[word][tag] += 1
                

        ## collect counts 
        for (words, tags) in train_data:
            
            # add stop symbol
            words=words+[self.STOP]
            tags=tags+[self.STOP]

            # iterate over sentence
            for i, (word, tag) in enumerate(zip(words, tags)):

                if i==0:
                    prev_tag=self.START

                    # record only transition from start
                    self.transitions[prev_tag][tag] += 1
                    
                else:
                    prev_tag=tags[i-1]

                    # record count for transition
                    self.transitions[prev_tag][tag] += 1

                    if i < len(words)-1:
                        # record count for emissions
                        if counts[word] < 2:
                            self.emissions[tag][self.UNK] += 1
                        # note that infrequent words are counted twice
                        self.emissions[tag][word] += 1

        ## e(tag|word) = count(t->word)/count(word)
        for tag in self.emissions:
            total_tag=sum(self.emissions[tag].values())
            for word in self.emissions[tag]:
                prob_word_given_tag = self.emissions[tag][word] / float(total_tag)
                self.emissions[tag][word] = prob_word_given_tag

        ## t(tag|prevtag) = count(prevtag,tag)/ count(prevtag)
        for prevtag in self.transitions:
            total_prevtag=sum(self.transitions[prevtag].values())
            for tag in self.transitions[prevtag]:
                prob_tag_given_prevtag = self.transitions[prevtag][tag] / float(total_prevtag)
                self.transitions[prevtag][tag] =  prob_tag_given_prevtag

    def predict(self, data, method):
        """
        predict the most likely tag sequence for all sentences in data

        :param data: a list of sentences
        :param method: viterbi or most likely decoding
        :return: list of predicted tag sequences
        """
        results = []
        for sentence in data:
            if method == 'viterbi':
                results.append(self.predict_viterbi(sentence[0]))
            else:
                results.append(self.predict_most_likely(sentence[0]))
        return results


    def predict_most_likely(self, sentence):
        """
        predict the single most likely tag (from training data) for every token in sentence
        (i.e., just looks at a single tag at a time, no context)
            
        :sentence: list of tokens
        :return: list of tags
        """
        tagSeq = []
        for word in sentence:
            #TODO implement, replace the string with the most likely tag for this word
            try:
                tag = max(self.token_tag_dist[word], key=self.token_tag_dist[word].get)
                tagSeq.append(tag)
            except Exception: #handle unknown words
                tagSeq.append(self.UNK)
        return tagSeq

    def predict_viterbi(self,sentence):
        """
        predict the most likely tag sequences using the Viterbi algorithm

        :sentence: list of tokens
        :return: list of tags
        """

        # replace unknown words for simplicity
        for i in range(len(sentence)):
            if sentence[i] not in self.vocabulary:
                sentence[i] = self.UNK

        # prepare data structures
        N = len(sentence)
        viterbiProbs = np.zeros((N, len(self.tags)))
        # viterbiBacktrace can be used to remember which previous tag was used
        viterbiBacktrace = np.zeros((N, len(self.tags)), dtype=int)
        # make self.tags a list, so we can use indexes
        self.tags = sorted(self.tags)

        # initialize first step (from START)
        for tagIdx, tag in enumerate(self.tags):
            emisProb = self.emissions[tag][sentence[0]]
            transProb = self.transitions[self.START][tag]
            viterbiProbs[0][tagIdx] = emisProb * transProb

        viterbiBacktrace[0][np.argmax(viterbiProbs[0])] = 1
        
        # process the rest of the sentence
        for t in range(1,N):
            for tagIdx, tag in enumerate(self.tags):
                emisProb = self.emissions[tag][sentence[t]]
                
                prev_tag_idx = np.argmax(viterbiBacktrace[t-1])
                prev_tag = self.tags[prev_tag_idx]
                if t != N-1: #final step
                    transProb = self.transitions[prev_tag][tag]
                else:
                    transProb = self.transitions[tag][self.STOP]
                
                viterbiProbs[t][tagIdx] = emisProb * transProb

            viterbiBacktrace[t][np.argmax(viterbiProbs[t])] = 1


        tagged_sen = [self.tags[np.argmax(tags)] for tags in viterbiBacktrace]
            
        return tagged_sen