import numpy as np
import time

class Agent:
    def __init__(self, env):
        self.env = env

    def sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        for t in range(horizon):
            # print('time step: {}/{}'.format(t, horizon))
            #print('time step in Agent', t)
            s = time.time() 
            actions.append(policy.act(states[t], t))
            print('time taken to act: ', np.round(time.time()-s,3), 'sec')
            #print('policy acted')
            state, reward, done, info = self.env.step(actions[t])
            #print('step taken')
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            if done:
                # print(info['done'])
                break

        # print("Rollout length: ", len(actions))
        print('Reward--',reward_sum)
        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }


class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def reset(self):
        pass

    def act(self, arg1, arg2):
        return np.random.uniform(size=self.action_dim) * 2 - 1
