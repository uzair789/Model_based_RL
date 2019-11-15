import os
import tensorflow as tf
import numpy as np
import gym
import copy


class MPC:
    def __init__(self, env, plan_horizon, model, pop_size, num_elites, max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """

        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
        :param pop_size: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
        :param use_mpc: Whether to use only the first action of a planned trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """
        self.env = env
        self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
        self.num_particles = num_particles
        self.plan_horizon = plan_horizon
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # TODO: write your code here
        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        #raise NotImplementedError
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.num_elites = num_elites
        self.mu = np.zeros([1, self.plan_horizon*self.action_dim]).squeeze()
        self.sigma = 0.5 * np.identity(self.plan_horizon * self.action_dim)
        #self.opt = self.cem_optimizer
        self.opt = self.random_optimizer
        self.actions = []
        self.goal = []


    def cem_optimizer(self, start_state):
        """This functions performs the CEM rollout and returns the mu.
        """
        mu = self.mu #np.zeros([self.plan_horizon, self.action_dim]).reshape(-1)
        sigma = self.sigma #0.5 * np.identity(self.plan_horizon * self.action_dim)
        "------work from here"
        #print('----', mu.shape, mu.squeeze().shape)

        for _ in range(self.max_iters):
            # We are sampling entire trajectory at once. Hence we request for pop_size number of trajectories
            action_seqs_raw = np.random.multivariate_normal(mu, sigma, (self.pop_size))
            #print('actions raw', action_seqs_raw.shape)
            # the shape of action_seqs will be (self.pop_size (200), T * action_dim (5*8))
            # so we reshape this into (200, 5, 8)
            action_seqs = action_seqs_raw.reshape(self.pop_size, self.plan_horizon, self.action_dim)
            # THis reshape might be expensive. If compute is slow, remove this and select the actions by inexing in steps of 8

            cost_per_trajectory = []
            for m in range(self.pop_size):
                # states will be 1 more than actions tau = (s1,a1,s2,a2,...sT+1)
                states  = self.get_trajectory_gt(start_state, action_seqs[m])  	
                cost_per_state  = [self.obs_cost_fn(state) for state in states ]
                cost_per_trajectory.append( np.sum(cost_per_state)) 

            positions = np.argsort(cost_per_trajectory)
            sorted_action_sequences = action_seqs_raw[positions, :] 
            #print('321',sorted_action_sequences.shape)
            #top_elite = sorted_action_sequences[-self.num_elites:, :]
            top_elite = sorted_action_sequences[0:self.num_elites, :]
            #print('31231;', top_elite.shape)
            mu, sigma = self.update_actions_mu_sigma(top_elite)
            #print('updated mu', mu.shape)
        return mu.reshape(self.plan_horizon, self.action_dim)


    def random_optimizer(self, start_state):
        """This functions performs the random action sequence and returns the action trajectory.
        """
        mu = self.mu #np.zeros([self.plan_horizon, self.action_dim]).reshape(-1)
        sigma = self.sigma #0.5 * np.identity(self.plan_horizon * self.action_dim)
        "------work from here"
        #print('----', mu.shape, mu.squeeze().shape)

        for _ in range(self.max_iters):
            # We are sampling entire trajectory at once. Hence we request for pop_size number of trajectories
            action_seqs_raw = np.random.multivariate_normal(mu, sigma, (self.pop_size))
            #print('actions raw', action_seqs_raw.shape)
            # the shape of action_seqs will be (self.pop_size (200), T * action_dim (5*8))
            # so we reshape this into (200, 5, 8)
            action_seqs = action_seqs_raw.reshape(self.pop_size, self.plan_horizon, self.action_dim)
            # THis reshape might be expensive. If compute is slow, remove this and select the actions by inexing in steps of 8

            cost_per_trajectory = []
            for m in range(self.pop_size):
                # states will be 1 more than actions tau = (s1,a1,s2,a2,...sT+1)
                states  = self.get_trajectory_gt(start_state, action_seqs[m])  	
                cost_per_state  = [self.obs_cost_fn(state) for state in states ]
                cost_per_trajectory.append( np.sum(cost_per_state)) 

            positions = np.argsort(cost_per_trajectory)
            sorted_action_sequences = action_seqs_raw[positions, :] 
            #print('321',sorted_action_sequences.shape)
            #top_elite = sorted_action_sequences[-self.num_elites:, :]
            best_action_sequence = sorted_action_sequences[0, :]
            #print('31231;', top_elite.shape)
            #print('updated mu', mu.shape)
        return best_action_sequence.reshape(self.plan_horizon, self.action_dim)

    def update_actions_mu_sigma(self, elite_actions):
        """This function updates the mean and the std
        """
        #print(elite_actions.shape)
        return np.mean(elite_actions, axis=0), np.cov(np.transpose(elite_actions))


    def get_trajectory_gt(self, state, actions):
        """This function returns a full trajectory of states based on the GT model
        """
        #i = 0
        states = []
        states.append(state)
        #while not done:
        #    action = actions[i]
        #    i +=1
        #    next_state, reward, done, _ = self.env.step(action)
        #    states.append(next_state)

        for action in actions:
            next_state = self.predict_next_state_gt(state, action)
            state = next_state
            states.append(state)

        return states

    def obs_cost_fn(self, state):
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        # TODO: write your code here
        raise NotImplementedError

    def predict_next_state_gt(self, state, action):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        # TODO: write your code here (DONE)
        return self.env.get_nxt_state(state, action)

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        # TODO: write your code here
        raise NotImplementedError

    def reset(self):
        # TODO: write your code here (Done)
        self.mu = np.zeros([1, self.plan_horizon*self.action_dim]).squeeze()
        self.sigma = 0.5 * np.identity(self.plan_horizon * self.action_dim)
        #raise NotImplementedError

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        # TODO: write your code here
        self.goal = state[-2:]
        if t % self.plan_horizon ==0:
            self.actions = self.opt(state)
            #print(t, self.actions.shape, 'actions ---- in act')

        if t >= self.plan_horizon:
            t = t%self.plan_horizon
        #print(self.actions.shape, '--->>>>')
        return self.actions[t]

    # TODO: write any helper functions that you need
