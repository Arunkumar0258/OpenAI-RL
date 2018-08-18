import gym
import numpy as np
from gym import wrappers

env = gym.make('Pendulum-v0')

bestLength = 0
episodes_length = []

best_weights = np.zeros(3)
observation = env.reset()
print(observation)

for i in range(100):
    new_weights = np.random.uniform(-2, 2, 3)

    length = []
    for j in range(100):
        observation = env.reset()
        done = False
        count = 0

        while not done:
            count += 1

            action = (-8, -2, -2) if np.dot(observation, new_weights) > 0 else (8, -2, -2)

            observation, reward, done, info = env.step(action)

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

    action = best_weights

    observation, reward, done, _ = env.step(action)

    if done:
        break

print('Best game lasted for: ', count, 'moves')
