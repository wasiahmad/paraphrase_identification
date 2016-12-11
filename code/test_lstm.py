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

def make_feature_vector(var_matrix, n_row, n_col, criterion='max'):
    max_matrix = np.zeros(shape=(n_row,n_col))
    mean_matrix = np.zeros(shape=(n_row,n_col))
    min_matrix = np.zeros(shape=(n_row,n_col))
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
            min_matrix[i][j] = np.min(np.ravel(pooled_matrix))
            c_start_idx = c_end_idx
        r_start_idx = r_end_idx
    max_matrix = np.ravel(max_matrix)
    mean_matrix = np.ravel(mean_matrix)
    min_matrix = np.ravel(min_matrix)

    if criterion == 'max':
        return max_matrix
    if criterion == 'mean':
        return mean_matrix
    if criterion == 'min':
        return min_matrix
    if criterion == 'max_mean' or criterion == 'mean_max':
        return np.ravel([max_matrix, mean_matrix])
    if criterion == 'max_min' or criterion == 'min_max':
        return np.ravel([max_matrix, min_matrix])
    if criterion == 'mean_min' or criterion == 'min_mean':
        return np.ravel([min_matrix, mean_matrix])
    return np.ravel([max_matrix, mean_matrix, min_matrix])


'''Pooling Criteria'''
POOLING_CRITERION = 'max'

"""###########################################Test Single Layer LSTM############################################"""
print("Single layer LSTM")
#'''Load Forward Model'''
model = load_model_parameters_theano('./models/forward_single_layer_lstm-2016-12-05-13-04-14000-48-64.dat.npz', LSTM)


#'''Generate Sentence embedding with forward model'''
sent1_train_indices, sent2_train_indices, word_to_index, index_to_word,label_train = get_train_data(VOCABULARY_SIZE)
first_train_sentences = generate_sentence_embeddings(model, sent1_train_indices, word_to_index, index_to_word)
second_train_sentences = generate_sentence_embeddings(model, sent2_train_indices, word_to_index, index_to_word)

for i in range(10):
    first_train = first_train_sentences[i]
    second_train = second_train_sentences[i]
    for j in range(len(first_train)):
        print(np.linalg.norm(np.subtract(first_train[j],second_train[j]) , 2))

assert len(first_train_sentences) == len(second_train_sentences)

#'''generate feature vector by all pair comparison, then pooling'''
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
    feature_vector = make_feature_vector(variable_matrix_train , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
    feature_vector_train.append(feature_vector)
feature_vector_train = np.asarray(feature_vector_train)
print(feature_vector_train.shape)

#'''Generate test data embeddings'''
sent1_test_indices, sent2_test_indices, word_to_index, index_to_word,label_test = get_test_data(VOCABULARY_SIZE)
first_test_sentences = generate_sentence_embeddings(model, sent1_test_indices, word_to_index, index_to_word)
second_test_sentences = generate_sentence_embeddings(model, sent2_test_indices, word_to_index, index_to_word)

assert len(first_test_sentences) == len(second_test_sentences)

#'''Generate feature vector for test; all pair comparison then pooling'''
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
    feature_vector = make_feature_vector(variable_matrix , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
    feature_vector_test.append(feature_vector)
feature_vector_test = np.asarray(feature_vector_test)
print(feature_vector_test[0])
print(feature_vector_test.shape)

#'''Building the Fully connected layer'''
clf = MLPClassifier()
clf.fit(feature_vector_train, label_train)
predicted = clf.predict(feature_vector_test)
print(predicted)
print("Accuracy:\t" , acc(label_test, predicted))
print("Precision:\t", pr(label_test, predicted))
print("Recall:\t", rc(label_test, predicted))
print("F-score:\t" , f1(label_test, predicted))


"""##############################################Test 2-Layer LSTM##############################################"""
# print("2 layer LSTM")
# #'''Load Forward Model'''
# model = load_model_parameters_theano('./models/forward_double_layer_lstm-2016-12-09-19-34-14000-48-64.dat.npz', DLSTM)
#
#
# #'''Generate Sentence embedding with forward model'''
# sent1_train_indices, sent2_train_indices, word_to_index, index_to_word,label_train = get_train_data(VOCABULARY_SIZE)
# first_train_sentences = generate_sentence_embeddings(model, sent1_train_indices, word_to_index, index_to_word)
# second_train_sentences = generate_sentence_embeddings(model, sent2_train_indices, word_to_index, index_to_word)
#
#
# assert len(first_train_sentences) == len(second_train_sentences)
#
# #'''generate feature vector by all pair comparison, then pooling'''
# feature_vector_train = []
# for idx in range(len(first_train_sentences)):
#     first_sentence = first_train_sentences[idx]
#     second_sentence = second_train_sentences[idx]
#     len1 = len(first_sentence)
#     len2 = len(second_sentence)
#     variable_matrix_train = np.zeros((len1,len2))
#     for i in range(len1):
#         word1 = first_sentence[i]
#         word1_normalized = word1 / np.linalg.norm(word1,2)
#         for j in range(len2):
#             word2 = second_sentence[j]
#             word2_normalized = word2 / np.linalg.norm(word2, 2)
#             variable_matrix_train[i][j] = np.dot(word1_normalized, word2_normalized)
#     feature_vector = make_feature_vector(variable_matrix_train , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
#     feature_vector_train.append(feature_vector)
# feature_vector_train = np.asarray(feature_vector_train)
# print(feature_vector_train.shape)
#
# #'''Generate test data embeddings'''
# sent1_test_indices, sent2_test_indices, word_to_index, index_to_word,label_test = get_test_data(VOCABULARY_SIZE)
# first_test_sentences = generate_sentence_embeddings(model, sent1_test_indices, word_to_index, index_to_word)
# second_test_sentences = generate_sentence_embeddings(model, sent2_test_indices, word_to_index, index_to_word)
#
# assert len(first_test_sentences) == len(second_test_sentences)
#
# #'''Generate feature vector for test; all pair comparison then pooling'''
# feature_vector_test = []
# for idx in range(len(first_test_sentences)):
#     first_sentence = first_test_sentences[idx]
#     second_sentence = second_test_sentences[idx]
#     len1 = len(first_sentence)
#     len2 = len(second_sentence)
#     variable_matrix = np.zeros((len1,len2))
#     for i in range(len1):
#         word1 = first_sentence[i]
#         word1_normalized = word1 / np.linalg.norm(word1,2)
#         for j in range(len2):
#             word2 = second_sentence[j]
#             word2_normalized = word2 / np.linalg.norm(word2, 2)
#             variable_matrix[i][j] = np.dot(word1_normalized, word2_normalized)
#     feature_vector = make_feature_vector(variable_matrix , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
#     feature_vector_test.append(feature_vector)
# feature_vector_test = np.asarray(feature_vector_test)
# print(feature_vector_test[0])
# print(feature_vector_test.shape)
#
# #'''Building the Fully connected layer'''
# clf = MLPClassifier()
# clf.fit(feature_vector_train, label_train)
# predicted = clf.predict(feature_vector_test)
# print("Accuracy:\t" , acc(label_test, predicted))
# print("Precision:\t", pr(label_test, predicted))
# print("Recall:\t", rc(label_test, predicted))
# print("F-score:\t" , f1(label_test, predicted))

"""###########################################Test Single Layer BLSTM###########################################"""
#
# print("1 layer BLSTM")
# #'''Load Forward Model'''
# model = load_model_parameters_theano('./models/forward_single_layer_lstm-2016-12-05-13-04-14000-48-64.dat.npz', LSTM)
#
# #'''Load backward model'''
# backward_model = load_model_parameters_theano(
#     './models/backward_single_layer_lstm-2016-11-30-21-07-14000-48-64.dat.npz', LSTM)
#
# #'''Generate Sentence embedding with forward model'''
# sent1_train_indices, sent2_train_indices, word_to_index, index_to_word,label_train = get_train_data(VOCABULARY_SIZE)
# first_train_sentences = generate_sentence_embeddings(model, sent1_train_indices, word_to_index, index_to_word)
# second_train_sentences = generate_sentence_embeddings(model, sent2_train_indices, word_to_index, index_to_word)
#
#
# #'''Generate Sentence embedding with backward model'''
# first_train_sentences_r = generate_sentence_embeddings(
#     backward_model, sent1_train_indices, word_to_index, index_to_word)
# second_train_sentences_r = generate_sentence_embeddings(
#     backward_model, sent2_train_indices, word_to_index, index_to_word)
#
# #'''Combine first train sentence (forward and backward embedding)'''
# first_train_sentences_combined = []
# train_pair_length = len(first_train_sentences)
# for i in range(train_pair_length):
#     sent_combined = []
#     forward = first_train_sentences[i]
#     backward = first_train_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     first_train_sentences_combined.append(sent_combined)
#
# #'''Combine second train sentence (forward and backward embedding)'''
# second_train_sentences_combined = []
# train_pair_length = len(second_train_sentences)
# for i in range(train_pair_length):
#     sent_combined = []
#     forward = second_train_sentences[i]
#     backward = second_train_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     second_train_sentences_combined.append(sent_combined)
# assert len(first_train_sentences_combined) == len(second_train_sentences_combined)
#
# #'''generate feature vector by all pair comparison, then pooling'''
# feature_vector_train = []
# for idx in range(len(first_train_sentences_combined)):
#     first_sentence = first_train_sentences_combined[idx]
#     second_sentence = second_train_sentences_combined[idx]
#     len1 = len(first_sentence)
#     len2 = len(second_sentence)
#     variable_matrix_train = np.zeros((len1,len2))
#     for i in range(len1):
#         word1 = first_sentence[i]
#         word1_normalized = word1 / np.linalg.norm(word1,2)
#         for j in range(len2):
#             word2 = second_sentence[j]
#             word2_normalized = word2 / np.linalg.norm(word2, 2)
#             variable_matrix_train[i][j] = np.dot(word1_normalized, word2_normalized)
#     feature_vector = make_feature_vector(variable_matrix_train , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
#     feature_vector_train.append(feature_vector)
# feature_vector_train = np.asarray(feature_vector_train)
# print(feature_vector_train.shape)
#
# #'''Generate test data embeddings'''
# sent1_test_indices, sent2_test_indices, word_to_index, index_to_word,label_test = get_test_data(VOCABULARY_SIZE)
# first_test_sentences = generate_sentence_embeddings(model, sent1_test_indices, word_to_index, index_to_word)
# second_test_sentences = generate_sentence_embeddings(model, sent2_test_indices, word_to_index, index_to_word)
#
# #'''Generate test data embedding backward'''
# first_test_sentences_r = generate_sentence_embeddings(
#     backward_model, sent1_test_indices, word_to_index, index_to_word)
# second_test_sentences_r = generate_sentence_embeddings(
#     backward_model, sent2_test_indices, word_to_index, index_to_word)
#
# #'''combine first sentence test embedding (forward and backward)'''
# first_test_sentences_combined = []
# test_pair_length = len(first_test_sentences)
# for i in range(test_pair_length):
#     sent_combined = []
#     forward = first_test_sentences[i]
#     backward = first_test_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     first_test_sentences_combined.append(sent_combined)
#
# #'''combine second sentence test embedding (forward and backward)'''
# second_test_sentences_combined = []
# test_pair_length = len(second_test_sentences)
# for i in range(test_pair_length):
#     sent_combined = []
#     forward = second_test_sentences[i]
#     backward = second_test_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     second_test_sentences_combined.append(sent_combined)
#
# assert len(first_test_sentences_combined) == len(second_test_sentences_combined)
#
# #'''Generate feature vector for test; all pair comparison then pooling'''
# feature_vector_test = []
# for idx in range(len(first_test_sentences_combined)):
#     first_sentence = first_test_sentences_combined[idx]
#     second_sentence = second_test_sentences_combined[idx]
#     len1 = len(first_sentence)
#     len2 = len(second_sentence)
#     variable_matrix = np.zeros((len1,len2))
#     for i in range(len1):
#         word1 = first_sentence[i]
#         word1_normalized = word1 / np.linalg.norm(word1,2)
#         for j in range(len2):
#             word2 = second_sentence[j]
#             word2_normalized = word2 / np.linalg.norm(word2, 2)
#             variable_matrix[i][j] = np.dot(word1_normalized, word2_normalized)
#     feature_vector = make_feature_vector(variable_matrix , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
#     feature_vector_test.append(feature_vector)
# feature_vector_test = np.asarray(feature_vector_test)
# print(feature_vector_test[0])
# print(feature_vector_test.shape)
#
# #'''Building the Fully connected layer'''
# clf = MLPClassifier()
# clf.fit(feature_vector_train, label_train)
# predicted = clf.predict(feature_vector_test)
# print("Accuracy:\t" , acc(label_test, predicted))
# print("Precision:\t", pr(label_test, predicted))
# print("Recall:\t", rc(label_test, predicted))
# print("F-score:\t" , f1(label_test, predicted))


"""#############################################Test 2-Layer BLSTM##############################################"""
# print("2 Layer BLSTM")
# #'''Load Forward Model'''
# model = load_model_parameters_theano('./models/forward_double_layer_lstm-2016-12-09-19-34-14000-48-64.dat.npz', DLSTM)
#
# #'''Load backward model'''
# backward_model = load_model_parameters_theano(
#     './models/backward_double_layer_lstm-2016-12-09-19-49-14000-48-64.dat.npz', DLSTM)
#
#
#
# #'''Generate Sentence embedding with forward model'''
# sent1_train_indices, sent2_train_indices, word_to_index, index_to_word,label_train = get_train_data(VOCABULARY_SIZE)
# first_train_sentences = generate_sentence_embeddings(model, sent1_train_indices, word_to_index, index_to_word)
# second_train_sentences = generate_sentence_embeddings(model, sent2_train_indices, word_to_index, index_to_word)
#
#
# #'''Generate Sentence embedding with backward model'''
# first_train_sentences_r = generate_sentence_embeddings(
#     backward_model, sent1_train_indices, word_to_index, index_to_word)
# second_train_sentences_r = generate_sentence_embeddings(
#     backward_model, sent2_train_indices, word_to_index, index_to_word)
#
# #'''Combine first train sentence (forward and backward embedding)'''
# first_train_sentences_combined = []
# train_pair_length = len(first_train_sentences)
# for i in range(train_pair_length):
#     sent_combined = []
#     forward = first_train_sentences[i]
#     backward = first_train_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     first_train_sentences_combined.append(sent_combined)
#
# #'''Combine second train sentence (forward and backward embedding)'''
# second_train_sentences_combined = []
# train_pair_length = len(second_train_sentences)
# for i in range(train_pair_length):
#     sent_combined = []
#     forward = second_train_sentences[i]
#     backward = second_train_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     second_train_sentences_combined.append(sent_combined)
# assert len(first_train_sentences_combined) == len(second_train_sentences_combined)
#
# #'''generate feature vector by all pair comparison, then pooling'''
# feature_vector_train = []
# for idx in range(len(first_train_sentences_combined)):
#     first_sentence = first_train_sentences_combined[idx]
#     second_sentence = second_train_sentences_combined[idx]
#     len1 = len(first_sentence)
#     len2 = len(second_sentence)
#     variable_matrix_train = np.zeros((len1,len2))
#     for i in range(len1):
#         word1 = first_sentence[i]
#         word1_normalized = word1 / np.linalg.norm(word1,2)
#         for j in range(len2):
#             word2 = second_sentence[j]
#             word2_normalized = word2 / np.linalg.norm(word2, 2)
#             variable_matrix_train[i][j] = np.dot(word1_normalized, word2_normalized)
#     feature_vector = make_feature_vector(variable_matrix_train , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
#     feature_vector_train.append(feature_vector)
# feature_vector_train = np.asarray(feature_vector_train)
# print(feature_vector_train.shape)
#
# #'''Generate test data embeddings'''
# sent1_test_indices, sent2_test_indices, word_to_index, index_to_word,label_test = get_test_data(VOCABULARY_SIZE)
# first_test_sentences = generate_sentence_embeddings(model, sent1_test_indices, word_to_index, index_to_word)
# second_test_sentences = generate_sentence_embeddings(model, sent2_test_indices, word_to_index, index_to_word)
#
# #'''Generate test data embedding backward'''
# first_test_sentences_r = generate_sentence_embeddings(
#     backward_model, sent1_test_indices, word_to_index, index_to_word)
# second_test_sentences_r = generate_sentence_embeddings(
#     backward_model, sent2_test_indices, word_to_index, index_to_word)
#
# #'''combine first sentence test embedding (forward and backward)'''
# first_test_sentences_combined = []
# test_pair_length = len(first_test_sentences)
# for i in range(test_pair_length):
#     sent_combined = []
#     forward = first_test_sentences[i]
#     backward = first_test_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     first_test_sentences_combined.append(sent_combined)
#
# #'''combine second sentence test embedding (forward and backward)'''
# second_test_sentences_combined = []
# test_pair_length = len(second_test_sentences)
# for i in range(test_pair_length):
#     sent_combined = []
#     forward = second_test_sentences[i]
#     backward = second_test_sentences_r[i]
#     assert len(forward) == len(backward)
#     sent_len = len(forward)
#     for j in range(sent_len):
#         forward_word = forward[j]
#         backward_word = backward[j]
#         sent_combined.append(np.ravel([forward_word, backward_word]))
#     second_test_sentences_combined.append(sent_combined)
#
# assert len(first_test_sentences_combined) == len(second_test_sentences_combined)
#
# #'''Generate feature vector for test; all pair comparison then pooling'''
# feature_vector_test = []
# for idx in range(len(first_test_sentences_combined)):
#     first_sentence = first_test_sentences_combined[idx]
#     second_sentence = second_test_sentences_combined[idx]
#     len1 = len(first_sentence)
#     len2 = len(second_sentence)
#     variable_matrix = np.zeros((len1,len2))
#     for i in range(len1):
#         word1 = first_sentence[i]
#         word1_normalized = word1 / np.linalg.norm(word1,2)
#         for j in range(len2):
#             word2 = second_sentence[j]
#             word2_normalized = word2 / np.linalg.norm(word2, 2)
#             variable_matrix[i][j] = np.dot(word1_normalized, word2_normalized)
#     feature_vector = make_feature_vector(variable_matrix , n_row = 8, n_col = 8, criterion=POOLING_CRITERION)
#     feature_vector_test.append(feature_vector)
# feature_vector_test = np.asarray(feature_vector_test)
# print(feature_vector_test[0])
# print(feature_vector_test.shape)
#
# #'''Building the Fully connected layer'''
# clf = MLPClassifier()
# clf.fit(feature_vector_train, label_train)
# predicted = clf.predict(feature_vector_test)
# print("Accuracy:\t" , acc(label_test, predicted))
# print("Precision:\t", pr(label_test, predicted))
# print("Recall:\t", rc(label_test, predicted))
# print("F-score:\t" , f1(label_test, predicted))
