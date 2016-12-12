import os
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def do_pooling(var_matrix, n_row, n_col, criterion='max'):
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

def combine_forward_and_backward_vectors(sentences, sentences_r):
    sentences_combined = []
    pair_length = len(sentences)
    assert pair_length == len(sentences_r)
    for i in range(pair_length):
        sent_combined =[]
        forward = sentences[i]
        backward = sentences_r[i]
        sent_len = len(forward)
        assert sent_len == len(backward)
        for j in range(sent_len):
            forward_word = forward[j]
            backward_word = backward[j]
            sent_combined.append(np.ravel([forward_word, backward_word]))
        sentences_combined.append(sent_combined)
    return sentences_combined

def generate_feature_vector(first_sentences, second_sentences):
    feature_vectors = []
    assert len(first_sentences) == len(second_sentences)
    for idx in range(len(first_sentences)):
        first_sentence = first_sentences[idx]
        second_sentence = second_sentences[idx]
        len1 = len(first_sentence)
        len2 = len(second_sentence)
        variable_matrix = np.zeros((len1, len2))
        for i in range(len1):
            word1 = first_sentence[i]
            word1_normalized = word1 / np.linalg.norm(word1, 2)
            for j in range(len2):
                word2 = second_sentence[j]
                word2_normalized = word2 / np.linalg.norm(word2, 2)
                variable_matrix[i][j] = np.dot(word1_normalized, word2_normalized)
        feature_vector = do_pooling(variable_matrix, n_row=4, n_col=4, criterion=POOLING_CRITERION)
        feature_vectors.append(feature_vector)
    return np.asarray(feature_vectors)

def build_classifier_and_test(train_X, train_y, test_X, test_y, clf, print_train_result = True):
    clf.fit(train_X, train_y)
    if print_train_result == True:
        p_tr = clf.predict(train_X)
        print("Train Accuracy:\t", acc(train_y, p_tr))
        print("Train Precision:\t", pr(train_y, p_tr))
        print("Train Recall_score:\t", rc(train_y, p_tr))
        print("Train F-score:\t", f1(train_y, p_tr))
    predicted = clf.predict(test_X)
    print("Accuracy:\t", acc(test_y, predicted))
    print("Precision:\t", pr(test_y, predicted))
    print("Recall_score:\t", rc(test_y, predicted))
    print("F-score:\t", f1(test_y, predicted))

def test_on_forward_LSTM(model, classifier=MLPClassifier()):
    # '''Generate Sentence embedding with forward model'''
    sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train = get_train_data(
        VOCABULARY_SIZE)
    first_train_sentences = generate_sentence_embeddings(model, sent1_train_indices, word_to_index, index_to_word)
    second_train_sentences = generate_sentence_embeddings(model, sent2_train_indices, word_to_index, index_to_word)
    assert len(first_train_sentences) == len(second_train_sentences)

    # '''generate feature vector by all pair comparison, then pooling'''
    feature_vector_train = generate_feature_vector(first_train_sentences, second_train_sentences)
    print("Train data Shape : " , feature_vector_train.shape)

    # '''Generate test data embeddings'''
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test = get_test_data(VOCABULARY_SIZE)
    first_test_sentences = generate_sentence_embeddings(model, sent1_test_indices, word_to_index, index_to_word)
    second_test_sentences = generate_sentence_embeddings(model, sent2_test_indices, word_to_index, index_to_word)
    assert len(first_test_sentences) == len(second_test_sentences)

    # '''Generate feature vector for test; all pair comparison then pooling'''
    feature_vector_test = generate_feature_vector(first_test_sentences, second_test_sentences)
#    print(feature_vector_test[0])
    print("Test data Shape : " , feature_vector_test.shape)

    # '''Building the Fully connected layer'''
    build_classifier_and_test(
        feature_vector_train, label_train, feature_vector_test, label_test, classifier, print_train_result=False)

def test_on_bidirectional_lstm(forward_model, backward_model, classifier=MLPClassifier()):
    # '''Generate Sentence embedding with forward model'''
    sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train = get_train_data(
        VOCABULARY_SIZE)
    first_train_sentences = generate_sentence_embeddings(forward_model,
                                                         sent1_train_indices, word_to_index, index_to_word)
    second_train_sentences = generate_sentence_embeddings(forward_model,
                                                          sent2_train_indices, word_to_index, index_to_word)

    # '''Generate Sentence embedding with backward model'''
    first_train_sentences_r = generate_sentence_embeddings(backward_model,
                                                           sent1_train_indices, word_to_index, index_to_word)
    second_train_sentences_r = generate_sentence_embeddings(backward_model,
                                                            sent2_train_indices, word_to_index, index_to_word)

    # '''Combine first train sentence (forward and backward embedding)'''
    first_train_sentences_combined = combine_forward_and_backward_vectors(first_train_sentences,
                                                                          first_train_sentences_r)

    # '''Combine second train sentence (forward and backward embedding)'''
    second_train_sentences_combined = combine_forward_and_backward_vectors(second_train_sentences,
                                                                           second_train_sentences_r)
    assert len(first_train_sentences_combined) == len(second_train_sentences_combined)

    # '''generate feature vector by all pair comparison, then pooling'''
    feature_vector_train = generate_feature_vector(first_train_sentences_combined, second_train_sentences_combined)
    print("Train data Shape : " , feature_vector_train.shape)

    # '''Generate test data embeddings'''
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test = get_test_data(VOCABULARY_SIZE)
    first_test_sentences = generate_sentence_embeddings(forward_model,
                                                        sent1_test_indices, word_to_index, index_to_word)
    second_test_sentences = generate_sentence_embeddings(forward_model,
                                                         sent2_test_indices, word_to_index, index_to_word)

    # '''Generate test data embedding backward'''
    first_test_sentences_r = generate_sentence_embeddings(
        backward_model, sent1_test_indices, word_to_index, index_to_word)
    second_test_sentences_r = generate_sentence_embeddings(
        backward_model, sent2_test_indices, word_to_index, index_to_word)

    # '''combine first sentence test embedding (forward and backward)'''
    first_test_sentences_combined = combine_forward_and_backward_vectors(first_test_sentences, first_test_sentences_r)

    # '''combine second sentence test embedding (forward and backward)'''
    second_test_sentences_combined = combine_forward_and_backward_vectors(second_test_sentences,
                                                                          second_test_sentences_r)
    assert len(first_test_sentences_combined) == len(second_test_sentences_combined)

    # '''Generate feature vector for test; all pair comparison then pooling'''
    feature_vector_test = generate_feature_vector(first_test_sentences_combined, second_test_sentences_combined)
    print("Test data Shape : " , feature_vector_test.shape)

    # '''Building the Fully connected layer'''
    build_classifier_and_test(
        feature_vector_train, label_train, feature_vector_test, label_test, classifier,
        print_train_result=False)


'''Pooling Criteria'''
POOLING_CRITERION = 'max'
print("Multi Layer Perceptron")
"""###########################################Test Single Layer LSTM############################################"""
# print("Single layer LSTM")
# model = load_model_parameters_theano('./models/forward_single_layer_lstm-2016-12-05-13-04-14000-48-64.dat.npz', LSTM)
# test_on_forward_LSTM(model, classifier=LogisticRegression())

"""##############################################Test 2-Layer LSTM##############################################"""
# print("2 layer LSTM")
# model = load_model_parameters_theano('./models/forward_double_layer_lstm-2016-12-09-19-34-14000-48-64.dat.npz', DLSTM)
# test_on_forward_LSTM(model, classifier=LogisticRegression())

"""###########################################Test Single Layer BLSTM###########################################"""
print("1 layer BLSTM")
#'''Load Forward Model'''
forward_model = load_model_parameters_theano('./models/forward_single_layer_lstm-2016-12-05-13-04-14000-48-64.dat.npz', LSTM)
#'''Load backward model'''
backward_model = load_model_parameters_theano(
    './models/backward_single_layer_lstm-2016-11-30-21-07-14000-48-64.dat.npz', LSTM)
test_on_bidirectional_lstm(forward_model, backward_model, classifier=LogisticRegression())

"""#############################################Test 2-Layer BLSTM##############################################"""
print("2 Layer BLSTM")
#'''Load Forward Model'''
forward_model = load_model_parameters_theano('./models/forward_double_layer_lstm-2016-12-09-19-34-14000-48-64.dat.npz', DLSTM)
#'''Load backward model'''
backward_model = load_model_parameters_theano(
    './models/backward_double_layer_lstm-2016-12-09-19-49-14000-48-64.dat.npz', DLSTM)
test_on_bidirectional_lstm(forward_model, backward_model, classifier=LogisticRegression())
