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
    trainloc = loc + '20_news_group_sentences.txt'
    sentences = []

    with open(trainloc, 'r', encoding='utf8') as f:
        for line in f:
            sentences.append("%s %s %s" % (SENTENCE_START_TOKEN, line, SENTENCE_END_TOKEN))

    return sentences


def build_dictionary(loc='./data/', vocabulary_size=-1):
    trainloc = loc + '20_news_group_sentences.txt'
    document_frequency = {}
    total_document = 0

    with open(trainloc, 'r', encoding='utf8') as f:
        for line in f:
            sentence = my_tokenizer(line)
            for token in set(sentence):
                if token in document_frequency:
                    document_frequency[token] += 1
                else:
                    document_frequency[token] = 1
            total_document += 1

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
        if len(key) < 4:
            continue
        elif counter == vocabulary_size:
            break
        word_to_index[key] = counter
        index_to_word[counter] = key
        counter += 1

    return word_to_index, index_to_word


def my_tokenizer(input):
    token_list = []
    tokens = wordpunct_tokenize(input.lower())
    token_list.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return token_list


def get_train_data(vocabulary_size):
    word_to_index, index_to_word = build_dictionary(vocabulary_size=vocabulary_size)
    sentences = load_data()

    sentences_tokenized = [my_tokenizer(sent) for sent in sentences]

    for i, sent in enumerate(sentences_tokenized):
        sentences_tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    sentences_indices = []
    for sentence in sentences_tokenized:
        sentences_indices.append([word_to_index[word] for word in sentence])

    return sentences_indices, word_to_index, index_to_word


def get_train_data_reversed(vocabulary_size):
    sentences_indices, word_to_index, index_to_word = get_train_data(vocabulary_size)
    sentences_indices_reversed = []
    for index_list in sentences_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sentences_indices_reversed.append(temp)

    return sentences_indices_reversed, word_to_index, index_to_word


def get_train_sentences(vocabulary_size):
    sentences_indices, word_to_index, index_to_word = get_train_data(vocabulary_size)
    all_sentences = []
    all_sentences.extend(sentences_indices)

    x_train = np.asarray([[w for w in sentence[:-1]] for sentence in all_sentences])
    y_train = np.asarray([[w for w in sentence[1:]] for sentence in all_sentences])

    return x_train, y_train, word_to_index, index_to_word


def get_train_sentences_reversed(vocabulary_size):
    sentences_indices_reversed, word_to_index, index_to_word = get_train_data_reversed(vocabulary_size)
    all_sentences = []
    all_sentences.extend(sentences_indices_reversed)

    x_train = np.asarray([[w for w in sentence[:-1]] for sentence in all_sentences])
    y_train = np.asarray([[w for w in sentence[1:]] for sentence in all_sentences])

    return x_train, y_train, word_to_index, index_to_word
