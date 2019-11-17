import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400


class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.sess = tf.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        K.set_session(self.sess)

        # Log variance bounds
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # TODO write your code here
        # Create and initialize your model
        model = self.create_network()
        self.model = model

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(inputs=I, outputs=O)
        return model

    def gauss_loss(self, means, log_vars, next_states):
        """ THis function computes the loss."""
        diff = tf.subtract(means, next_states)
        print('----diff---->', diff)
        cov_mat = tf.matrix_diag(log_vars)
        print('----cov mat---->', cov_mat)
        log_det = tf.log( tf.linalg.det(cov_mat) )
        print('----log_det---->', log_det)

        #do 1/vars
        inverse_vars = tf.divide(1,log_vars)
        loss = tf.reduce_sum(diff * inverse_vars * diff, axis=1) + log_det



        return tf.reduce_mean(loss)

    def get_train_data(self, inputs, targets, batch_size):
        """ return a random batch of data."""
        indices = np.random.randint(0, inputs.shape[0], size=(batch_size,))
        return inputs[indices, :], targets[indices, :]

    def forward(self, input_data):
        output = self.model.predict(input_data)
        mean, log_var = self.get_output(output)
        return mean.squeeze()

    


    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """



        I = tf.placeholder(dtype=tf.float32, shape=[None, inputs.shape[1]])
        y = tf.placeholder(dtype=tf.float32, shape=[None, targets.shape[1]])
        model_output = self.model(I)
        mean, log_var = self.get_output (model_output)
        loss = self.gauss_loss(mean, log_var, y)
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        self.sess.run(tf.global_variables_initializer())

        iters_per_epoch = np.floor(inputs.shape[1]/batch_size)

        for epoch in range(epochs):
            for i in range(iters_per_epoch):
                batch_data, batch_targets = self.get_train_data(inputs, targets, batch_size)
                _, loss_value = self.sess.run([train_op, loss], feed_dict={I:batch_data, y:batch_targets})
                print(e, i, loss_value)
            # shuffle the data at the end of each epoch
            #inputs, targets = shuffle_data(inputs, targets) 

    # TODO: Write any helper functions that you need
