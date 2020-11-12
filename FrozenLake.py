#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import time


# In[2]:


from gym.envs.toy_text import frozen_lake, discrete
from gym.envs.registration import register


register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False})

register(
    id='Deterministic-8x8-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '8x8',
            'is_slippery': False})

register(
    id='Stochastic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': True})


# In[3]:


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):


    value_function = np.zeros(nS)
    updated_value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        for state in range(nS):
            for next_state in P[state][policy[state]]:
                probability = next_state[0]
                new_state = next_state[1]
                reward = next_state[2]/len(P[state][policy[state]])
                #print(gamma*probability*value_function[new_state])
                updated_value_function[state] += (reward + gamma*probability*value_function[new_state])
                
        diff = max(np.abs(updated_value_function - value_function))
        if diff <= tol:
            break
        else:
            value_function = updated_value_function.copy()
            updated_value_function = np.zeros(nS)
            #print(updated_value_function, '\n')
            #print(value_function)

    ############################
    return value_function


# In[4]:


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):


    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #
    for state in range(nS):
        actions = []
        for action in range(nA):
            value = 0
            for next_state in P[state][action]:
                probability = next_state[0]
                new_state = next_state[1]
                reward = next_state[2]/len(P[state][action])
                value += (reward + gamma*probability*value_from_policy[new_state])
            actions.append(value)
        new_policy[state] = np.argmax(actions)
            
    ############################
    return new_policy


# In[5]:


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    for i in range(10):
        value_from_policy = policy_evaluation(P, nS, nA, policy, gamma, tol)
        policy = policy_improvement(P, nS, nA, value_from_policy, policy, gamma)
        #policy = new_policy.copy()
        #diff = max(np.abs(value_from_policy - value_function))


    ############################
    return value_from_policy, policy


# In[6]:


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):

    value_function = np.zeros(nS)
    updated_value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    count = 0
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        count += 1
        for state in range(nS):
            for next_state in P[state][policy[state]]:
                probability = next_state[0]
                new_state = next_state[1]
                reward = next_state[2]/len(P[state][policy[state]])
                #print(gamma*probability*value_function[new_state])
                updated_value_function[state] += (reward + gamma*probability*value_function[new_state])
                
        policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        
        diff = max(np.abs(updated_value_function - value_function))
        if diff <= tol and count>10:
            break
        else:
            value_function = updated_value_function.copy()
            updated_value_function = np.zeros(nS)
            #print(updated_value_function, '\n')
            #print(value_function)


    ############################
    return value_function, policy


# In[10]:


def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	# env = gym.make("Stochastic-4x4-FrozenLake-v0")

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)


# In[ ]:




