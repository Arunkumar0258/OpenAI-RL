import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

MAXSTATES = 10000
GAMMA = 0.9
ALPHA = 0.01

def max_dict(d):
    max_vector = float('-inf')
    for key, val in d.items():
        if val > max_vector:
            max_vector = val
            max_key = key
    return max_key, max_vector

def create_bins():
    bins = np.zeros((4,10))
    bins[0] = np.linspace(-4.8, 4.8,10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5, 5, 10)

    return bins

def assign_bins(observation, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state

def get_state_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state

def get_all_state_string():
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4))
    return states

def initialize_Q():
    Q = {}

    all_states = get_all_state_string()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q

def play_one_game(bins, Q, eps=0.5):
    observation = env.reset()

    done = False
    count = 0
    state = get_state_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        count += 1
        if np.random.uniform() < eps:
            action = env.action_space.sample()
        else:
            action = max_dict(Q[state])[0]

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done and count < 200:
            reward = -300

        state_new = get_state_string(assign_bins(observation, bins))

        a1, max_q = max_dict(Q[state_new])
        Q[state][action] += ALPHA*(reward + GAMMA*max_q - Q[state][action])
        state, act = state_new, action

    return total_reward, count

def play_many_games(bins, n=10000):
    Q = initialize_Q()

    length = []
    reward = []
    for i in range(n):
        eps = 1.0 / np.sqrt(n+1)

        episode_reward, episode_length = play_one_game(bins, Q, eps)

        if n%100 == 0:
            print(n, '%.4f', eps, episode_reward)
        length.append(episode_length)
        reward.append(episode_reward)
    return length, reward

def plot_avg(totalrewards):
    n = len(totalrewards)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = np.mean(totalrewards[max(0, t-100) : (t+1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == "__main__":
    bins = create_bins()
    episode_length, episode_reward = play_many_games(bins)

    plot_avg(episode_reward)

