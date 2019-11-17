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

        self.I = tf.placeholder(dtype=tf.float32, shape=[None, state_dim+action_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.model_output = self.model(self.I)
        self.mean, self.log_var = self.get_output (self.model_output)
        self.loss = self.gauss_loss(self.mean, self.log_var,self.y)
        self.rmse = self.get_rmse(self.mean, self.y)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())


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
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.000001))(h3)
        model = Model(inputs=I, outputs=O)
        return model

    def gauss_loss(self, means, log_vars, next_states):
        """ THis function computes the loss."""
        diff = tf.subtract(means, next_states)
        print('----diff---->', diff)
        #cov_mat = tf.matrix_diag(log_vars)
        #print('----cov mat---->', cov_mat)
        #log_det = tf.log( tf.linalg.det(cov_mat) )
        log_det = tf.log( tf.reduce_prod(log_vars, axis=1 ))
        print('----log_det---->', log_det)

        #do 1/vars
        inverse_vars = tf.divide(1,log_vars)
        loss = tf.reduce_sum(diff * inverse_vars * diff, axis=1) + log_det


        return tf.reduce_mean(loss)

    def get_rmse(self, means, next_states):
        
        rmse = tf.sqrt( tf.reduce_mean( (means - next_states)**2))
        return rmse

    def get_train_data(self, inputs, targets, batch_size):
        """ return a random batch of data."""
        indices = np.random.randint(0, inputs.shape[0], size=(batch_size,))
        return inputs[indices, :], targets[indices, :]

    def forward(self, input_data):
        #output = self.model.predict(input_data)
        #mean, log_var = self.get_output(output)
 
        mean_value = self.sess.run(self.mean, feed_dict={self.I:input_data})
        """
        I = tf.placeholder(dtype=tf.float32, shape=[None, input_data.shape[1]])
        model_output = self.model(I)
        op = self.sess.run(model_output, feed_dict={I:input_data})
        mean = op[0:8]
        #tf.reset_default_graph()
        """
        return mean_value.squeeze()

    


    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """


        f = open('loss_log.txt','w')
        #self.sess.graph.finalize()

        print(inputs.shape[0], batch_size)
        iters_per_epoch = int(np.floor(inputs.shape[0]/batch_size))
        print(iters_per_epoch, '-----MMMM')
        for e in range(epochs):
            for i in range(iters_per_epoch):
                batch_data, batch_targets = self.get_train_data(inputs, targets, batch_size)
                _, loss_value, rmse_value = self.sess.run([self.train_op, self.loss, self.rmse], feed_dict={self.I:batch_data, self.y:batch_targets})
                line = ('Epoch: '+str(e)+' | '+ str(i)+'/'+str(iters_per_epoch)+'---- loss= '+ str(loss_value)+ ' | '+'RMSE= '+str (rmse_value))
                f.write(line+'\n')
                print(line)
            # shuffle the data at the end of each epoch
            #inputs, targets = shuffle_data(inputs, targets) 

    # TODO: Write any helper functions that you need
