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
        """
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
        """

        #Working with the ensemble
        self.lr = 0.001
        self.models = []
        self.input_placeholders = []
        self.target_placeholders = []
        self.means = [] 
        self.losses = []
        self.rmses = []
        self.train_ops  =[]
        for i in range(num_nets):
            model = self.create_network()
            I = tf.placeholder(dtype=tf.float32, shape=[None, state_dim+action_dim])
            y = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
            model_output = model(I)
            mean, log_var = self.get_output(model_output)
            loss = self.gauss_loss(mean, log_var, y)
            rmse = self.get_rmse(mean, y)
            train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
            self.models.append(model)
            self.input_placeholders.append(I)
            self.target_placeholders.append(y)
            self.means.append(mean)
            self.losses.append(loss)
            self.rmses.append(rmse)
            self.train_ops.append(train_op)
            
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
        log_det = tf.reduce_sum(log_vars, axis=1)
        
        #do 1/vars
        vars_ = tf.exp(log_vars)
        inverse_vars = tf.divide(1, vars_)
        
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
        feed_dict = {}
        for n in range(len(self.models)):
            feed_dict[self.input_placeholders[n]] = input_data 
        mean_values = self.sess.run(self.means, feed_dict=feed_dict)
        ind = np.random.randint(0, len(mean_values))
        """
        I = tf.placeholder(dtype=tf.float32, shape=[None, input_data.shape[1]])
        model_output = self.model(I)
        op = self.sess.run(model_output, feed_dict={I:input_data})
        mean = op[0:8]
        #tf.reset_default_graph()
        """
        return mean_values[ind].squeeze()

    


    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        iters_per_epoch = int(np.floor(inputs.shape[0]/batch_size))
        for e in range(epochs):
            for i in range(iters_per_epoch):
                feed_dict = {}
                for n in range(len(self.models)):
                    batch_data, batch_targets = self.get_train_data(inputs, targets, batch_size)
                    feed_dict[self.input_placeholders[n]] = batch_data
                    feed_dict[self.target_placeholders[n]] = batch_targets
                _, loss_value, rmse_value = self.sess.run([self.train_ops, self.losses, self.rmses], feed_dict=feed_dict)
                line = ('Epoch: '+str(e)+' | '+ str(i)+'/'+str(iters_per_epoch)+'---- loss= '+ str(loss_value)+ ' | '+'RMSE= '+str (rmse_value))
                print(line)
            # shuffle the data at the end of each epoch
            #inputs, targets = shuffle_data(inputs, targets) 

    # TODO: Write any helper functions that you need
