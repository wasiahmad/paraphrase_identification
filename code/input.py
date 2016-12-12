import numpy as np
from nltk import wordpunct_tokenize
import nltk
import itertools
import operator
import sklearn
import re, string
import math

SENTENCE_START_TOKEN = "sentence_start"
SENTENCE_END_TOKEN = "sentence_end"
UNKNOWN_TOKEN = "unknown_token"


def load_data(loc='./data/'):
    "loading MSRP dataset"
    trainloc = loc + 'msr_paraphrase_train.txt'
    testloc = loc + 'msr_paraphrase_test.txt'

    sent1_train, sent2_train, sent1_test, sent2_test = [], [], [], []
    label_train, label_dev, label_test = [], [], []

    with open(trainloc, 'r', encoding='utf8') as f:
        f.readline()  # skipping the header of the file
        for line in f:
            text = line.strip().split('\t')
            sent1_train.append("%s %s %s" % (SENTENCE_START_TOKEN, text[3], SENTENCE_END_TOKEN))
            sent2_train.append("%s %s %s" % (SENTENCE_START_TOKEN, text[4], SENTENCE_END_TOKEN))
            label_train.append(int(text[0]))

    with open(testloc, 'r', encoding='utf8') as f:
        f.readline()  # skipping the header of the file
        for line in f:
            text = line.strip().split('\t')
            sent1_test.append("%s %s %s" % (SENTENCE_START_TOKEN, text[3], SENTENCE_END_TOKEN))
            sent2_test.append("%s %s %s" % (SENTENCE_START_TOKEN, text[4], SENTENCE_END_TOKEN))
            label_test.append(int(text[0]))

    return [sent1_train, sent2_train], [sent1_test, sent2_test], [label_train, label_test]


def build_dictionary(loc='./data/', vocabulary_size=-1):
    "load MSRP dataset and construct a dictionary"
    trainloc = loc + 'msr_paraphrase_train.txt'
    testloc = loc + 'msr_paraphrase_test.txt'

    document_frequency = {}
    total_document = 0
    with open(trainloc, 'r', encoding='utf8') as f:
        f.readline()  # skipping the header of the file
        for line in f:
            text = line.strip().split('\t')
            sentence1 = my_tokenizer(text[3])
            sentence2 = my_tokenizer(text[4])

            for token in set(sentence1):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            for token in set(sentence2):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            total_document = total_document + 2

    with open(testloc, 'r', encoding='utf8') as f:
        f.readline()  # skipping the header of the file
        for line in f:
            text = line.strip().split('\t')
            sentence1 = my_tokenizer(text[3])
            sentence2 = my_tokenizer(text[4])

            for token in set(sentence1):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            for token in set(sentence2):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            total_document = total_document + 2

    for key, value in document_frequency.items():
        document_frequency[key] = math.log(total_document / document_frequency[key])

    vocab = sorted(document_frequency.items(), key=operator.itemgetter(1), reverse=True)

    word_to_index = {}
    index_to_word = {}
    word_to_index[SENTENCE_START_TOKEN] = 0
    word_to_index[SENTENCE_END_TOKEN] = 1
    word_to_index[UNKNOWN_TOKEN] = 2
    index_to_word[0] = SENTENCE_START_TOKEN
    index_to_word[1] = SENTENCE_END_TOKEN
    index_to_word[2] = UNKNOWN_TOKEN

    counter = 3
    for key, value in vocab:
        if (len(key) < 4):
            continue
        elif counter == vocabulary_size:
            break
        word_to_index[key] = counter
        index_to_word[counter] = key
        counter = counter + 1

    return word_to_index, index_to_word


def my_tokenizer(input):
    tokenList = []
    tokens = wordpunct_tokenize(input.lower())
    tokenList.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return tokenList


def get_train_data(vocabulary_size):
    word_to_index, index_to_word = build_dictionary(vocabulary_size=vocabulary_size)
    [sent1_train, sent2_train], [sent1_test, sent2_test], [label_train, label_test] = load_data()

    sent1_train_tokenized = [my_tokenizer(sent) for sent in sent1_train]
    sent2_train_tokenized = [my_tokenizer(sent) for sent in sent2_train]

    for i, sent in enumerate(sent1_train_tokenized):
        sent1_train_tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
    for i, sent in enumerate(sent2_train_tokenized):
        sent2_train_tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    sent1_train_indices = []
    for sentence in sent1_train_tokenized:
        sent1_train_indices.append([word_to_index[word] for word in sentence])

    sent2_train_indices = []
    for sentence in sent2_train_tokenized:
        sent2_train_indices.append([word_to_index[word] for word in sentence])

    return sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train


def get_train_data_reversed(vocabulary_size):
    sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train = get_train_data(vocabulary_size)
    sent1_train_indices_reversed = []
    for index_list in sent1_train_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent1_train_indices_reversed.append(temp)

    sent2_train_indices_reversed = []
    for index_list in sent2_train_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent2_train_indices_reversed.append(temp)

    return sent1_train_indices_reversed, sent2_train_indices_reversed, word_to_index, index_to_word, label_train


def get_train_sentences(vocabulary_size):
    sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train = get_train_data(vocabulary_size)
    all_sentences = []
    all_sentences.extend(sent1_train_indices)
    all_sentences.extend(sent2_train_indices)

    X_train = np.asarray([[w for w in sentence[:-1]] for sentence in all_sentences])
    y_train = np.asarray([[w for w in sentence[1:]] for sentence in all_sentences])

    return X_train, y_train, word_to_index, index_to_word


def get_train_sentences_reversed(vocabulary_size):
    sent1_train_indices_reversed, sent2_train_indices_reversed, word_to_index, index_to_word, label_train = get_train_data_reversed(
        vocabulary_size)
    all_sentences = []
    all_sentences.extend(sent1_train_indices_reversed)
    all_sentences.extend(sent2_train_indices_reversed)

    X_train = np.asarray([[w for w in sentence[:-1]] for sentence in all_sentences])
    y_train = np.asarray([[w for w in sentence[1:]] for sentence in all_sentences])

    return X_train, y_train, word_to_index, index_to_word


def get_test_data(vocabulary_size):
    word_to_index, index_to_word = build_dictionary(vocabulary_size=vocabulary_size)
    [sent1_train, sent2_train], [sent1_test, sent2_test], [label_train, label_test] = load_data()

    sent1_test_tokenized = [my_tokenizer(sent) for sent in sent1_test]
    sent2_test_tokenized = [my_tokenizer(sent) for sent in sent2_test]

    for i, sent in enumerate(sent1_test_tokenized):
        sent1_test_tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
    for i, sent in enumerate(sent2_test_tokenized):
        sent2_test_tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    sent1_test_indices = []
    for sentence in sent1_test_tokenized:
        sent1_test_indices.append([word_to_index[word] for word in sentence])

    sent2_test_indices = []
    for sentence in sent2_test_tokenized:
        sent2_test_indices.append([word_to_index[word] for word in sentence])

    return sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test



def get_test_data_reversed(vocabulary_size):
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test = get_test_data(vocabulary_size)

    sent1_test_indices_reversed = []
    for index_list in sent1_test_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent1_test_indices_reversed.append(temp)

    sent2_test_indices_reversed = []
    for index_list in sent2_test_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent2_test_indices_reversed.append(temp)

    return sent1_test_indices_reversed, sent2_test_indices_reversed, word_to_index, index_to_word, label_test


def get_test_sentences(vocabulary_size):
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test = get_test_data(vocabulary_size)
    all_sentences = []
    all_sentences.extend(sent1_test_indices)
    all_sentences.extend(sent2_test_indices)

    x_test = np.asarray([[w for w in sentence] for sentence in all_sentences])

    return x_test, word_to_index, index_to_word


def get_test_sentences_reversed(vocabulary_size):
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test= get_test_data_reversed(
        vocabulary_size)
    all_sentences = []
    all_sentences.extend(sent1_test_indices)
    all_sentences.extend(sent2_test_indices)

    x_test = np.asarray([[w for w in sentence] for sentence in all_sentences])

    return x_test, word_to_index, index_to_word
