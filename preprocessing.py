"""
The script for data preprocessing.
"""

import torch 
import nltk

from time import time
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('wordnet')
"""
The script for data preprocessing.
"""
class Vocabulary():
    def __init__(self):
        self.token2index = {'<UNK>': 0, '<PAD>': 1}
        self.token_ls = ['<UNK>', '<PAD>']
        self.token_cnt = 2

    def add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.token_cnt
            self.token_ls.append(token)
            self.token_cnt += 1
        assert self.token_cnt == len(self.token_ls) == len(self.token2index)

    def add_token_ls(self, token_ls):
        for token in token_ls:
            self.add_token(token)

    def get_token_ls(self):
        return self.token_ls

    def get_token2index(self):
        return self.token2index

    def get_vocab_size(self):
        return self.token_cnt

    def __str__(self):
        return "Vocabulary of {} tokens".format(self.token_cnt)

"""
The main class that will be used to process the data for topic classification.
"""
class DataProcessor():
    def __init__(self, lemmatizer):
        self.vocab = Vocabulary()
        self.lemmatizer = lemmatizer
        self.label2index = {}

    """ 
    Parse the input txt file and return a processed format

    Args:
        file_name: the name of the input file that need to be parsed
        mode: "train" / "evaluate"
              "train": parsing the training data, build vocabulary and 
                       label mapping on the fly.
              "evaluate": parsing the non-training data.
    Return:
            (label_ls, tokens_ls)
            label_ls: A list of topic labels in index format. 
                      [topic_idx_1, topic_idx_2, ...]
            tokens_id_ls: A list of tokens that has been converted to index
                          based on the vocabulary build on training set.
                          [[t11, t12, t13, ...], [t21, t22, ...], ...]
    """
    def process(self, file_name, mode):
        start_time = time()
        f_read = open(file_name, 'r')
        success_line_cnt = 0
        fail_line_cnt = 0
        topic_cnt = 0
        label_ls = []
        tokens_id_ls = []

        for line in f_read:
            try:
                assert len(line.split('|||')) == 2
                success_line_cnt += 1
            except:
                fail_line_cnt += 1
                continue

            splited_line = line.split('|||')
            topic= splited_line[0].lower().strip()
            text = splited_line[1].lower().strip()

            token_ls = self.tokenize(text)

            if mode == "train":
                # update label-to-index mapping
                if topic not in self.label2index:
                    self.label2index[topic] = topic_cnt
                    topic_cnt += 1

                # update vocabulary
                self.vocab.add_token_ls(token_ls)

            # append tokens and label in a index version
            label_ls.append(self.get_label_idx(topic))
            tokens_id_ls.append(self.get_tokens_id(token_ls))

        print("Data processing on {} data finished in {:.1f}s. "
              "[{} line sucessful, {} line failed]".format(
                mode, time() - start_time, success_line_cnt, fail_line_cnt))

        return label_ls, tokens_id_ls


    def get_label_idx(self, topic):
        try:
            label_idx = self.label2index[topic]
        except:
            raise ValueError("the topic {} is not defined in the training set".format(
                topic))

        return label_idx

    def get_tokens_id(self, token_ls):
        tokens_id = []
        for token in token_ls:
            try:
                tokens_id.append(self.vocab.get_token2index()[token])
            except:
                tokens_id.append(self.vocab.get_token2index()['UNK'])

        assert len(tokens_id) == len(token_ls)
        return tokens_id

    def tokenize(self, text):
        token_ls = text.split()
        token_ls = list(map(self.tokenize_function, token_ls))
        return token_ls

    def tokenize_function(self, token):
        token = token.strip()
        if self.lemmatizer is not None:
            token = self.lemmatizer.lemmatize(token)
            token = self.lemmatizer.lemmatize(token, 'v')
            token = token.strip()
        return token

    def get_vocab(self):
        return self.vocab

if __name__ == '__main__':
    data_processor = DataProcessor(WordNetLemmatizer())
    train_label_ls, train_tokens_id_ls = data_processor.process(
        'topicclass/topicclass_train.txt', 'train')
    valid_label_ls, valid_tokens_id_ls = data_processor.process(
        'topicclass/topicclass_valid.txt', 'validation')
    test_label_ls, test_tokens_id_ls = data_processor.process(
        'topicclass/topicclass_test.txt', 'test')
    vocabulary = data_processor.get_vocab()
    print("Overall vocabulary size is {}".format(vocabulary))

