import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator


# Theano implementation of a two-layer LSTM
class DLSTM:
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (8, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (8, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((8, hidden_dim))
        c = np.zeros(word_dim)

        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, h_t1_prev, h_t2_prev, c_t1_prev, c_t2_prev):
            # Word embedding layer
            x_e = E[:, x_t]

            # LSTM Layer 1
            i_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(h_t1_prev) + b[0])
            f_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(h_t1_prev) + b[1])
            o_t1 = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(h_t1_prev) + b[2])
            g_t1 = T.tanh(U[3].dot(x_e) + W[3].dot(h_t1_prev) + b[3])
            c_t1 = c_t1_prev * f_t1 + g_t1 * i_t1
            h_t1 = T.tanh(c_t1) * o_t1

            # LSTM Layer 2
            i_t2 = T.nnet.hard_sigmoid(U[4].dot(h_t1) + W[4].dot(h_t2_prev) + b[4])
            f_t2 = T.nnet.hard_sigmoid(U[5].dot(h_t1) + W[5].dot(h_t2_prev) + b[5])
            o_t2 = T.nnet.hard_sigmoid(U[6].dot(h_t1) + W[6].dot(h_t2_prev) + b[6])
            g_t2 = T.tanh(U[7].dot(h_t1) + W[7].dot(h_t2_prev) + b[7])
            c_t2 = c_t2_prev * f_t2 + g_t2 * i_t2
            h_t2 = T.tanh(c_t2) * o_t2

            # Final output calculation
            output_t = T.nnet.softmax(V.dot(h_t2) + c)[0]

            return [output_t, h_t1, h_t2, c_t1, c_t2]

        [output, hidden_state1, hidden_state2, cell_state1, cell_state2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])

        prediction = T.argmax(output, axis=1)
        output_error = T.sum(T.nnet.categorical_crossentropy(output, y))

        # Total cost, we can add regularization here
        cost = output_error

        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # Assign functions
        self.predict = theano.function([x], output)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
        self.cell_states = theano.function([x], cell_state2)
        self.hidden_states = theano.function([x], hidden_state2)

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                     ])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)
