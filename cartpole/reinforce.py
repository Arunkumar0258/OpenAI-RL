import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')

bestLength = 0
episodes_length = []

best_weights = np.zeros(4)
observation = env.reset()
print(observation)

for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    length = []
    for j in range(100):
        observation = env.reset()
        done = False
        count = 0

        while not done:
            count += 1

            action = 1 if np.dot(observation, new_weights) > 0 else 0

            observation, reward, done, _ = env.step(action)

            if done:
                break
            length.append(count)
        average_length = float(sum(length) / len(length))

        if average_length > bestLength:
            bestLength = average_length
            best_weights = new_weights
        episodes_length.append(average_length)
        if i % 10 == 0:
            print('Best length is: ', bestLength)

done = False
count = 0
env = wrappers.Monitor(env, 'Movie1', force=True)
observation = env.reset()


while not done:
    count += 1

    action = 1 if np.dot(observation, best_weights) > 0 else 0

    observation, reward, done, _ = env.step(action)

    if done:
        break

print('Best game lasted for: ', count, 'moves')
