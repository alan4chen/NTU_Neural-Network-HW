"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain import DeepQNetwork

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


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation.flatten())

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation.flatten(), action, reward, observation_.flatten())

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
