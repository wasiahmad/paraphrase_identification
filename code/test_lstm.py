import os
from input import *
from utils import *
from lstm import LSTM
from deep_lstm import DLSTM

VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "14000"))


def generate_sentence_embedding(model, test_sent):
    context_vectors = model.cell_states(test_sent)


def generate_sentence_embeddings(model, x_test, word_to_index, index_to_word):
    for test_sent in x_test:
        generate_sentence_embedding(model, test_sent)


###########################################Test Single Layer LSTM############################################
x_test, word_to_index, index_to_word = get_test_sentences(VOCABULARY_SIZE)
model = load_model_parameters_theano('./model/Forward_Single_LSTM-2016-12-05-13-04-14000-48-64.dat.npz', LSTM)
generate_sentence_embeddings(model, x_test, word_to_index, index_to_word)

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
