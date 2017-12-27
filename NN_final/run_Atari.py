"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain2 import DeepQNetwork

env = gym.make('SpaceInvaders-v0')
env = env.unwrapped

print("action_space:", env.action_space)
print("observation_space:", env.observation_space)
print("observation_space.high:", env.observation_space.high)
print("observation_space.low:", env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0

ep_rhistory = []


for i_episode in range(500):

    observation = env.reset()
    ep_r = 0
    while True:
        # env.render()

        action = RL.choose_action(observation.flatten())

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        # print(observation_)
        # x, x_dot, theta, theta_dot = observation_
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # reward = r1 + r2

        RL.store_transition(observation.flatten(), action, reward, observation_.flatten())

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))

            ep_rhistory.append(ep_r)
            break

        observation = observation_
        total_steps += 1

print(ep_rhistory)
import _pickle
_pickle.dump(ep_rhistory, open("ep_rhistory.pickle", 'wb'))

RL.plot_cost()

