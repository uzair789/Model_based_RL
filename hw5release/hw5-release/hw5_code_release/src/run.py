import matplotlib.pyplot as plt
import numpy as np
import gym
import envs
import os.path as osp
import time
import argparse
import time
import collections


from agent import Agent, RandomPolicy
from mpc import MPC
from model import PENN

# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8

LOG_DIR = './data'




class ExperimentGTDynamics(object):
    def __init__(self, env_name='Pushing2D-v1', mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        # Does not need model
        self.warmup = False
        mpc_params['use_gt_dynamics'] = True
        self.cem_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        for j in range(num_episodes):
            print('Test episode {}'.format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success
PATH = './plots'

class ExperimentModelDynamics:
    def __init__(self, env_name='Pushing2D-v1', num_nets=1, mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        mpc_params['use_gt_dynamics'] = False
        self.model = PENN(num_nets, STATE_DIM, len(self.env.action_space.sample()), LR)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)
        self.random_policy_no_mpc = RandomPolicy(len(self.env.action_space.sample()))
        self.states_buffer = collections.deque(maxlen=40*100) 
        self.actions_buffer = collections.deque(maxlen=40*100)
        self.rewards_buffer = collections.deque(maxlen=40*100)
        self.buffer_flag = True

    def plot_graph(self, data, title, xlabel, ylabel):
        plt.figure(figsize=(12,5))
        plt.title(title)
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(osp.join(PATH, title+'.png'))


    def plot_loss_rmse(self, suffix):
        loss, rmse = self.model.get_loss_rmse()
        #print('------in plotting---')
        #print(loss, loss.shape, rmse, rmse.shape)
        for i in range(loss.shape[1]):
             data = loss[:, i]
             title = suffix+'ModelLoss_'+str(i)
             xlabel = 'epochs'
             ylabel = 'Loss'
             self.plot_graph(data, title, xlabel, ylabel)

        for i in range(rmse.shape[1]):
             data = rmse[:, i]
             title = suffix+'ModelRMSE_'+str(i)
             xlabel = 'epochs'
             ylabel = 'RMSE'
             self.plot_graph(data, title, xlabel, ylabel)


    def test(self, num_episodes, optimizer='cem'):
        samples = []
        for j in range(num_episodes):
            print('Test episode {} -- policy = {}'.format(j, optimizer))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success

    def model_warmup(self, num_episodes, num_epochs):
        """ Train a single probabilistic model using a random policy """

        samples = []
        for i in range(num_episodes):
            samples.append(self.agent.sample(self.task_horizon, self.random_policy_no_mpc))

        self.cem_policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples],
            epochs=num_epochs
        )

    def train(self, num_train_epochs, num_episodes_per_epoch, evaluation_interval, test_episodes):
        """ Jointly training the model and the policy """
        f = open('pets_20_with_'+str(self.buffer_flag)+'.txt','w')


        cem_reward = []
        cem_success = []

        random_reward = []
        random_success = []


        for i in range(num_train_epochs):
            start1 = time.time() 
            print("####################################################################")
            print("Starting training epoch %d." % (i + 1))
            line = ("Starting training epoch %d." % (i + 1))
            f.write(line+'\n')

            samples = []
            for j in range(num_episodes_per_epoch):
                samples.append(
                    self.agent.sample(
                        self.task_horizon, self.cem_policy
                    )
                )
            print("Rewards obtained:", [sample["reward_sum"] for sample in samples])
            line = ("Rewards obtained:" + str( [sample["reward_sum"] for sample in samples]))
            f.write(line+'\n')

            self.states_buffer.extend( [sample['obs'] for sample in samples])
            self.actions_buffer.extend([sample['ac'] for sample in samples])
            self.rewards_buffer.extend([sample['rewards'] for sample in samples])

            if self.buffer_flag:
                self.cem_policy.train(
                     self.states_buffer,
                     self.actions_buffer,
                     self.rewards_buffer,
                    epochs=5
                )

            else:
                self.cem_policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples],
                    epochs=5
                )
            time_for_train = np.round(time.time() - start1,3)
            f.write('Time taken for train part of the epoch: '+str(time_for_train)+'\n')
            print('Time taken for train part of the epoch: '+str(time_for_train))
            if (i + 1) % evaluation_interval == 0:
                start = time.time()
                avg_return, avg_success = self.test(test_episodes, optimizer='cem')
                
                cem_reward.append(avg_return)
                cem_success.append(avg_success)



                time_taken_cem = np.round(time.time() - start, 3) 
                print('Test success CEM + MPC:', avg_success, 'Time taken: ',time_taken_cem)
                line = ('Test success CEM + MPC:' + str(avg_success)+' Time taken: '+str(time_taken_cem) )
                f.write(line+'\n')

                start = time.time()
                avg_return, avg_success = self.test(test_episodes, optimizer='random')
                
                random_reward.append(avg_return)
                random_success.append(avg_success)


                time_taken_random = time.time() - start
                print('Test success Random + MPC:', avg_success, 'Time taken: ', time_taken_random)
                line = ('Test success Random + MPC:'+str( avg_success)+' TIme taken: '+str(time_taken_random))
                f.write(line+'\n')
        f.close()
        self.plot_graph(cem_success, 'cem_success'+str(self.buffer_flag), 'epochs', 'success')
        self.plot_graph(random_success, 'random_success'+str(self.buffer_flag), 'epochs', 'success')

def test_cem_gt_dynamics(num_episode=10):
    
    f1 = open('cem_gt_dynamics.txt','w')
    print('CEM PushingEnv without MPC')
    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    line = ('CEM PushingEnv without MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)

    #
    print('CEM PushingEnv with MPC')
    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    line = ('CEM PushingEnv with MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)
    #
    print('CEM PushingEnv Noisy without MPC')
    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    line = ('CEM PushingEnv Noisy without MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)
    #
    print('CEM PushingEnv Noisy with MPC')
    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    line = ('CEM PushingEnv Noisy with MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)
    
    print('RANDOM PushingEnv without MPC')
    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    line = ('RANDOM PushingEnv without MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)

    print('RANDOM PushingEnv with MPC')
    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    line = ('RANDOM PushingEnv with MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)


    print('RANDOM PushingEnv Noisy without MPC')
    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    line = ('RANDOM PushingEnv Noisy without MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)

    print('RANDOM PushingEnv Noisy with MPC')
    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    line = ('RANDOM PushingEnv Noisy with MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)
    f1.close()



def train_single_dynamics(num_test_episode=50):
    num_nets = 1
    num_episodes = 1000
    num_epochs = 100


    f1 = open('train_single_dynamics_vec_r.txt','w')
     
    # CEM WITH MPC
    start = time.time()
    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)
    exp.model_warmup(num_episodes=num_episodes, num_epochs=num_epochs)
    exp.plot_loss_rmse('single_cem_mpc_vec')
    avg_reward, avg_success = exp.test(num_test_episode, optimizer='cem')
    line = ('CEM PushingEnv with MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    f1.write(line)
    f1.write('Time taken: '+str(np.round(time.time() - start, 3))+'\n')
    """
    # Random WITH  MPC
    start = time.time()
    mpc_params = {'use_mpc':True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)
    exp.model_warmup(num_episodes=num_episodes, num_epochs=num_epochs)
    exp.plot_loss_rmse('single_random_mpc_vec')
    avg_reward, avg_success = exp.test(num_test_episode, optimizer='random')
    line = ('RANDOM PushingEnv with MPC: avg_reward: {}, avg_success: {}\n'.format(avg_reward, avg_success))
    print(line)
    
    f1.write(line)
    f1.write('Time taken: '+str(np.round(time.time() - start, 3))+'\n')
    """    
    f1.close()
    
def train_pets():
    num_nets = 2
    num_epochs = 500
    evaluation_interval = 50
    num_episodes_per_epoch = 1
    test_episodes = 20
    warmup_episodes = 100

    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)
    exp.model_warmup(num_episodes=warmup_episodes, num_epochs=10)
    exp.train(num_train_epochs=num_epochs,
              num_episodes_per_epoch=num_episodes_per_epoch,
              evaluation_interval=evaluation_interval,
              test_episodes=test_episodes)

    exp.plot_loss_rmse('pets_')


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='HW5')
  
    parser.add_argument('--train', help='gt_model or single_model or pets')
    args = parser.parse_args()

    #test_cem_gt_dynamics(50)
    if args.train == 'gt_model':
        test_cem_gt_dynamics(50)
    elif args.train == 'single_model':
        train_single_dynamics(50)
    elif args.train == 'pets':
        train_pets()
    """
    #test_cem_gt_dynamics(50)
    #train_single_dynamics(50)
    train_pets()
