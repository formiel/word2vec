"""
Implementation of Skip-gram with Negative Sampling from scratch

Author: Hang LE
Email: hangtp.le@gmail.com
"""
from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
import pickle # to save and load the model
import os # to join the path
import random # for randomly generated numbers
import time # to record time for the report
import matplotlib # to plot the loss of experiments for the report
# matplotlib.use('TkAgg') # macOS
matplotlib.use("agg")   # Linux
import matplotlib.pyplot as plt 

def text2sentences(path):
    """
    Function to load, clean and tokenize the text

    - Input: path to text file used to train model
    - Output: a list of tokenized sentences
    """
    # Initialize list to store processed sentences
    sentences = []
    # Define punctuations to be removed
    table = str.maketrans('', '', '''!"#$%&()*+,./:;<=>?@[\]^_`{|}~''')

    # Process and append each sentence to sentences
    with open(path, encoding="utf8") as f:
        for l in f:
            # Convert to lower case and split into words
            sent = l.lower().split()
            # Remove punctuation
            sent_removed_punk = [word.translate(table) for word in sent]
            # Append sent_removed to the list of sentences
            sentences.append(sent_removed_punk)

    return sentences

def loadPairs(path):
    """
    Function to load test data

    - Input: path to test file
    - Output: pairs of word1, word2 and pre-annotated similarity 
    """
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])

    return pairs

def sigmoid(x):
    """
    Return logarithmic value of the sigmoid function for a given x
    """
    return 1.0 /(1 + np.exp(-x))

class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        self.sentences = sentences
        self.nEmbed = nEmbed # dimension of embedding
        self.negativeRate = negativeRate # ratio of negative samples over positive samples
        self.winSize = winSize
        self.minCount = max(minCount, 1)
        
        self.word2idx = None
        self.unigram = None


    def compute_word2idx_and_unigram(self, unigram_power=0.75):
        """
        Function perform word count and mapping from word to index and vice versa

        - Input: a list of tokenized sentences
        - Output: 
            -- V: vocabulary size
            -- word_list: list of vocabulary sorted by alphabetical order
            -- word2idx: dictionary with word as key, index as value
            -- word_freq: dictionary with word as key, number of word occurence as value
        """
        # Initialize a dictionary of word frequency
        word_freq = {}
        # Iterate over each sentence in the list of sentences
        for sent in self.sentences:
            # Iterate over each word in sentence
            for word in sent:
                # Create the frequency dictionary to count each word
                word_freq[word] = word_freq.get(word, 0) + 1

        # Remove words that have frequency < minCount
        if self.minCount > 1:
            word_freq = {word:freq for word, freq in word_freq.items() if freq >= self.minCount}

        # Create word2idx and idx2word dictionaries from word_list
        self.word2idx = {w: idx for (idx, w) in enumerate(word_freq.keys())}

        ############### Compute unigram
        # Initialize an array of unigram
        unigram = np.zeros(len(self.word2idx))
        # Iterate over list of words and calculate the probability for each word
        for word, frequency in word_freq.items():
            # Raise each word frequency to the power chosen
            f = frequency ** unigram_power
            # Update unigram array
            unigram[self.word2idx[word]] = f
        
        # Normalization
        self.unigram = unigram / np.sum(unigram)


    def compute_positives(self):
        """
        Generate data for training

        - Input: list of tokenized sentences
        - Output:
            -- P: 2D list where P[t] is the list of positive samples wrt a word t
        """
        # Initialize useful lists for optimization phase
        P = [] # Indexes of Positive samples of t

        V = len(self.word2idx)
        number_of_sentences = len(self.sentences)

        # I represent the sentences using word indices (rather than words themselves)
        # If the word does not exist in the dictionary (due to setting minCount > 1) then set its index to -1
        sentences_index_form = [None]*number_of_sentences
        for idx, sent in enumerate(self.sentences):
            sentences_index_form[idx] = [self.word2idx.get(w, -1) for w in sent]

        # For efficiency, pre-compute the number of positives for each word
        number_of_positives = np.zeros(V, dtype=int)
        for idx, sent_word_indices in enumerate(sentences_index_form):
            for i, word_idx in enumerate(sent_word_indices):
                if word_idx < 0:
                    continue
                first = max(0, i-self.winSize)
                last = min(i+self.winSize+1, len(sent_word_indices))
                number_of_positives[word_idx] += (last - first - 1)

        # Now I can allocate the memory for P in advance
        P = [None]*V
        for word_idx in range(V):
            P[word_idx] = np.zeros(number_of_positives[word_idx], dtype=int)

        # Now start filling P
        # For each word t, I keep track of the last position that has
        # been filled in P[t] using P_next_position[t]
        P_next_position = [0]*V

        # Iterate over each sentence in the corpus to extract the target word and context words
        for idx, sent_word_indices in enumerate(sentences_index_form):
            if (idx + 1) % 100 == 0:
                print('Processing sentence', idx + 1, '/', number_of_sentences)
            # Iterate over each word in sentence and add to its Positive list
            for i, word_idx in enumerate(sent_word_indices):
                if word_idx < 0:
                    continue
                first = max(0, i-self.winSize)
                last = min(i+self.winSize+1, len(sent_word_indices))
                number_of_words = (last - first - 1)
                position = P_next_position[word_idx]
                # print('first =', first, 'last =', last, 'position =', position)
                P[word_idx][position:position+number_of_words] = np.asarray(sent_word_indices[first:i] + sent_word_indices[i+1:last])
                P_next_position[word_idx] += number_of_words

        # If minCount > 1 then a lot of -1 in P. I will remove them
        # print('V =', V)
        if self.minCount > 1:
            for word_idx in range(V):
                P[word_idx] = np.delete(P[word_idx], np.where(P[word_idx] < 0))
                # if np.any(P[word_idx] >= V):
                #     print(P[word_idx])
                #     raise "error"
        # Remove duplicates
        for word_idx in range(V):
            P[word_idx] = np.unique(P[word_idx])
        return P


    def negative_sampling(self, t, Pt):
        """
        Draw negative samples from unigram distribution
        t: a target word
        Pt: list of words w such that (t, w) is a positive samples
        - Output: a list of indexes of the negative samples
        """
        # Remove indices of t and Pt as they cannot be negative wrt to t
        invalid_indices = Pt.tolist() + [t]
        # print('invalid_indices =', invalid_indices)
        
        # Now for each postive sample (i.e. each element of Pt),
        # I will randomly generate self.negativeRate negative samples.
        # To avoid mistakenly obtaining postive samples or t itself,
        # I set the probabilities of these indices to 0.
        probabilities = np.copy(self.unigram)
        probabilities[invalid_indices] = 0
        probabilities /= np.sum(probabilities)
        negative_samples = np.random.choice(len(self.unigram), size=self.negativeRate, p=probabilities)

        return negative_samples



    def train(self, stepsize, epochs, patience=3, save_model_path=None):
        """
        Train the model

        - Input: 
            -- path_to_model: path to store model
        - Output:
            -- Return trained embeddings and saved to path_to_model
        """
        print('Start compute_word2idx_and_unigram')
        start = time.time()
        self.compute_word2idx_and_unigram()
        print('Took', time.time() - start, '(s).')

        print('Start compute_positives')
        start = time.time()
        P = self.compute_positives()
        print('Took', time.time() - start, '(s).')

        if save_model_path is not None:
            save_model_path = os.path.expanduser(save_model_path)
            save_dir = os.path.dirname(save_model_path)
            if save_dir != '' and not os.path.exists(save_dir):
                os.makedirs(save_dir)

        V = len(self.word2idx)
        print('Total number of words =', V)

        # Initialization
        W = np.random.rand(self.nEmbed, V)
        C = np.random.rand(V, self.nEmbed)

        losses = []
        loss_best = 1e100
        epochs_no_improvement = 0

        start = time.time()
        for epoch_idx in range(epochs):
            print('Epoch', epoch_idx + 1)
            # print('-----------------------------------------')
            # I iterate through the index of each word directly,
            # because index is the only thing we need, we dont need the word itself
            loss_epoch = 0.0 # accumulate the loss for all words
            for t in range(V): # it is the index of the target word in the vocab
                # Get the current embedding vectors
                wt = W[:, t]
                positive_samples = P[t]
                for p in positive_samples:
                    negative_samples = self.negative_sampling(t, positive_samples)

                    # print('number of positive samples =', len(positive_samples))
                    # print('number of negative samples =', len(negative_samples))

                    # context vector of the postive sample and the negative ones
                    cp = C[p, :]
                    C_neg = C[negative_samples, :]

                    # intermediate values that are helpful
                    sp = sigmoid(-np.dot(cp, wt))
                    s_neg = sigmoid(np.dot(C_neg, wt))
                    # print('S_neg.shape =', s_neg.shape)

                    # Compute partial derivatives
                    dwt = -sp*cp + np.dot(s_neg, C_neg)
                    dcp = -sp*wt
                    dC_neg = np.outer(s_neg, wt)

                    # Gradient descent update
                    wt -= stepsize*dwt
                    cp -= stepsize*dcp
                    C_neg -= stepsize*dC_neg
                    
                    loss = -np.log(sigmoid(np.dot(cp, wt))) \
                            + np.sum(-np.log(sigmoid(-np.dot(C_neg, wt))))
                    loss_epoch += loss
                    # # print('Epoch', epoch_idx, ', step', t,': loss =', loss)
                    # losses[epoch_idx, t] = loss

                if (t+1)%100 == 0:
                    print('\t step ' + str(t + 1) + '/' + str(V) + '\t loss: %.2f'%(loss) + ' accul.loss: %.2f'%(loss_epoch), end='\r')
            # print('\n', end='\r')
            # Done updating for all words
            losses.append(loss_epoch)
            print('\t Loss: %.2f'%loss_epoch, 'Ellapse time:', time.time() - start, '(s)--------')
            if loss_epoch < loss_best:
                loss_best = loss_epoch
                epochs_no_improvement = 0
                # Only save the best parameters
                self.W = W
                self.C = C
                if save_model_path is not None:
                    self.save(save_model_path)
            else:
                epochs_no_improvement += 1
                print('\t No improvement for', epochs_no_improvement, 'epochs')
            
            fname = 'losses' + '_nEmbed' + str(self.nEmbed) \
                    + '_negativeRate' + str(self.negativeRate) \
                    + '_winSize' + str(self.winSize) \
                    + '_minCount' + str(self.minCount) \
                    + '_stepsize' + str(stepsize)

            np.save(fname + '.npy', losses)
            # Plot the loss and save for report
            plt.xlabel('epoch')
            plt.xlabel('loss')
            plt.plot(losses, 'r-')
            plt.savefig(fname + '.png')

            if epochs_no_improvement >= patience:
                print('EARLY STOPPING.')
                break
            

    def save(self, path):
        """
        save the data to file
        """
        data = {'word2idx': self.word2idx,
                'W': self.W,
                'C': self.C,
                'negativeRate': self.negativeRate,
                'nEmbed': self.nEmbed,
                'winSize': self.winSize,
                'minCount': self.minCount}

        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


    def similarity(self, word1, word2):
        """
        Computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float in [0,1] indicating the similarity (the higher the more similar)
        """
        # Get the indices of the words, if not exist then take the first index
        idx1 = self.word2idx.get(word1, 0)
        idx2 = self.word2idx.get(word2, 0)

        # Get learned embedding vectors
        w1 = self.W[:, idx1]
        w2 = self.W[:, idx2]

        # Calculate cosine similarity score
        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        score = np.dot(w1, w2)/ (norm1 * norm2)

        return score

    @staticmethod
    def load(path):
        """
        Load the parameters for testing
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        sg = SkipGram(sentences=None,
                      nEmbed=data['nEmbed'],
                      negativeRate=data['negativeRate'],
                      winSize=data['winSize'],
                      minCount=data['minCount'])
        sg.W = data['W']
        sg.C = data['C']
        sg.word2idx = data['word2idx']
        return sg


def main():
    """
    main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode (for submission to Prof.)', action='store_true')
    parser.add_argument('--validate', help='enters validation mode (compute cross-validation with ground-truth)', action='store_true')
    parser.add_argument('--nEmbed', help='embedding dimension', type=int, default=100)
    parser.add_argument('--negativeRate', help='ratio of negative sampling', type=int, default=5)
    parser.add_argument('--winSize', help='context window size', type=int, default=5)
    parser.add_argument('--minCount', help='take into account only words with more appearances than this number', type=int, default=5)
    parser.add_argument('--stepsize', help='stepsize for gradient descent', type=float, default=0.0001)
    parser.add_argument('--epochs', help='number of training epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5, help='patience (in number of epochs) for early stopping: stop training if the loss has not been improved over the last N epochs')

    opts = parser.parse_args()

    if not opts.test:
        if not opts.validate:
            print('Read text to sentences')
            start = time.time()
            sentences = text2sentences(opts.text)
            print('Took', time.time() - start, '(s). Total', len(sentences), 'sentences.')

            sg = SkipGram(sentences, opts.nEmbed, opts.negativeRate, opts.winSize, opts.minCount)
            print('Start training')
            start = time.time()
            sg.train(stepsize=opts.stepsize, epochs=opts.epochs, patience=opts.patience, save_model_path=opts.model)
            print('Total training time:', time.time() - start, '(s)')
            # sg.save(opts.model) # It is safer to save the model during training
        else:
            print('Validation mode')
            data = pd.read_csv(opts.text, delimiter='\t')
            pairs = zip(data['word1'], data['word2'])
            sim_gt = data['similarity'].values

            sg = SkipGram.load(opts.model)
            sim_predicted = np.zeros(sim_gt.shape)
            for idx, (a,b) in enumerate(pairs):
                if (idx+1)%100 == 0:
                    print(idx + 1, '/', len(sim_predicted))
                sim_predicted[idx] = sg.similarity(a,b)
            # Compute cross-correlation
            corr = np.corrcoef(sim_gt, sim_predicted)
            print('correlation:', corr)
    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))


if __name__ == '__main__':
    main()
