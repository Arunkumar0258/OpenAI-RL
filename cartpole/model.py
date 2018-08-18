import gym
import random
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

lr = 0.0001

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_future = 50
initial_games = 10000

def random_games():
    for episode in range(10):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

#random_games()

def generate_train_data():
    train_data = []
    scores = []
    correct_scores = []

    for _ in range(initial_games):
        score = 0
        memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_future:
            correct_scores.append(score)
            for data in memory:
                if data[1] == 1:
                    output = [0,1]
                if data[1] == 0:
                    output = [1,0]
                train_data.append([data[0], output])
        env.reset()
        scores.append(score)
    train_data_save = np.array(train_data)
    np.save('saved.npy', train_data_save)

    print('Average accepted score: {}'.format(mean(correct_scores)))
    print('Median accepted score: {}'.format(median(correct_scores)))
    print(Counter(correct_scores))

    return train_data

def NeuralNet(input_size):
    network = input_data(shape = [None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
     
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate = lr, loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network)

    return model

def train_model(train_data, model=False):
    X = np.array([i[0] for i in train_data]).reshape(-1, len(train_data[0][0]), 1)
    y = [i[1] for i in train_data]

    if not model:
        model = NeuralNet(input_size = len(X[0]))

    #model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai')
    model.fit(X, y, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

data = generate_train_data()
model = train_model(data)

scores = []
choices = []

for game in range(100):
    score = 0
    memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average score: {}'.format(sum(scores)/len(scores)))
print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
