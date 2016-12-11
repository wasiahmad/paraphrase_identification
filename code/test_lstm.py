import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as pr
from sklearn.metrics import recall_score as rc
import numpy as np
from input import *
from utils import *
from lstm import LSTM
from deep_lstm import DLSTM


VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "14000"))


def generate_sentence_embedding(model, test_sent):
    context_vectors = model.cell_states(test_sent)
    return context_vectors


def generate_sentence_embeddings(model, x_test, word_to_index, index_to_word):
    test_context_vectors = [];
    for test_sent in x_test:
        test_context_vectors.append(generate_sentence_embedding(model, test_sent))
    return test_context_vectors

def make_feature_vector(var_matrix, n_row, n_col):
    max_matrix = np.zeros(shape=(n_row,n_col))
    mean_matrix = np.zeros(shape=(n_row,n_col))
    (o_row, o_col) = var_matrix.shape
    r_div = o_row / n_row
    c_div = o_col / n_col
    r_start_idx = 0
    r_float_idx = 0
    for i in range(n_row):
        r_end_idx = int(np.round(r_float_idx + r_div))
        r_float_idx += r_div
        c_start_idx = 0
        c_float_idx = 0
        for j in range(n_col):
            c_end_idx = int(np.round(c_float_idx + c_div))
            c_float_idx += c_div
            pooled_matrix = var_matrix[r_start_idx:r_end_idx, c_start_idx:c_end_idx]
            max_matrix[i][j] = np.max(np.ravel(pooled_matrix))
            mean_matrix[i][j] = np.mean(np.ravel(pooled_matrix))
            c_start_idx = c_end_idx
        r_start_idx = r_end_idx
    max_matrix = np.ravel(max_matrix)
    mean_matrix = np.ravel(mean_matrix)

    return np.ravel([mean_matrix])

###########################################Test Single Layer LSTM############################################
#x_test, word_to_index, index_to_word = get_test_sentences(VOCABULARY_SIZE)


model = load_model_parameters_theano('./models/Forward_Single_LSTM-2016-12-05-13-04-14000-48-64.dat.npz', LSTM)

sent1_train_indices, sent2_train_indices, word_to_index, index_to_word,label_train = get_train_data(VOCABULARY_SIZE)
first_train_sentences = generate_sentence_embeddings(model, sent1_train_indices, word_to_index, index_to_word)
second_train_sentences = generate_sentence_embeddings(model, sent2_train_indices, word_to_index, index_to_word)
assert len(first_train_sentences) == len(second_train_sentences)
feature_vector_train = []
for idx in range(len(first_train_sentences)):
    first_sentence = first_train_sentences[idx]
    second_sentence = second_train_sentences[idx]
    len1 = len(first_sentence)
    len2 = len(second_sentence)
    variable_matrix_train = np.zeros((len1,len2))
    for i in range(len1):
        word1 = first_sentence[i]
        word1_normalized = word1 / np.linalg.norm(word1,2)
        for j in range(len2):
            word2 = second_sentence[j]
            word2_normalized = word2 / np.linalg.norm(word2, 2)
            variable_matrix_train[i][j] = np.dot(word1_normalized, word2_normalized)
    feature_vector = make_feature_vector(variable_matrix_train , n_row = 8, n_col = 8)
    feature_vector_train.append(feature_vector)
feature_vector_train = np.asarray(feature_vector_train)
print(feature_vector_train.shape)

sent1_test_indices, sent2_test_indices, word_to_index, index_to_word,label_test = get_test_data(VOCABULARY_SIZE)
first_test_sentences = generate_sentence_embeddings(model, sent1_test_indices, word_to_index, index_to_word)
second_test_sentences = generate_sentence_embeddings(model, sent2_test_indices, word_to_index, index_to_word)
assert len(first_test_sentences) == len(second_test_sentences)
feature_vector_test = []
for idx in range(len(first_test_sentences)):
    first_sentence = first_test_sentences[idx]
    second_sentence = second_test_sentences[idx]
    len1 = len(first_sentence)
    len2 = len(second_sentence)
    variable_matrix = np.zeros((len1,len2))
    for i in range(len1):
        word1 = first_sentence[i]
        word1_normalized = word1 / np.linalg.norm(word1,2)
        for j in range(len2):
            word2 = second_sentence[j]
            word2_normalized = word2 / np.linalg.norm(word2, 2)
            variable_matrix[i][j] = np.dot(word1_normalized, word2_normalized)
    feature_vector = make_feature_vector(variable_matrix , n_row = 8, n_col = 8)
    feature_vector_test.append(feature_vector)
feature_vector_test = np.asarray(feature_vector_test)
print(feature_vector_test[0])
print(feature_vector_test.shape)

clf = MLPClassifier()
clf.fit(feature_vector_train, label_train)
predicted = clf.predict(feature_vector_test)
print("Accuracy:\t" , acc(label_test, predicted))
print("Precision:\t", pr(label_test, predicted))
print("Recall:\t", rc(label_test, predicted))
print("F-score:\t" , f1(label_test, predicted))
##############################################Test 2-Layer LSTM##############################################
# x_test, word_to_index, index_to_word = get_test_sentences(VOCABULARY_SIZE)
# model = load_model_parameters_theano('./model/pretrained_lstm.npz', DLSTM)
# generate_sentence_embeddings(model, x_test, word_to_index, index_to_word)

###########################################Test Single Layer BLSTM###########################################
# x_test, word_to_index, index_to_word = get_test_sentences(VOCABULARY_SIZE)
# model = load_model_parameters_theano('./model/pretrained_lstm.npz', LSTM)
# generate_sentence_embeddings(model, x_test, word_to_index, index_to_word)

#############################################Test 2-Layer BLSTM##############################################
# x_test, word_to_index, index_to_word = get_test_sentences(VOCABULARY_SIZE)
# model = load_model_parameters_theano('./model/pretrained_lstm.npz', DLSTM)
# generate_sentence_embeddings(model, x_test, word_to_index, index_to_word)
