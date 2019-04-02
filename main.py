import numpy as np
import game, time
from nn import NN

env = game.Game(500, 500)
n_state = len(env.reset())
n_actions = env.n_actions

agent = NN(n_state, n_actions)

def generate_session(t_max = 1000):
    states, actions = [], []
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        probs = agent.predict_proba([s])[0]
        a = np.random.choice(n_actions, p = probs)
        new_s, r, done = env.step(a)
        env.render()

        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
        	break
    return states, actions, total_reward

import matplotlib.pyplot as plt

def show_progress(batch_rewards, log, percentile, reward_range=[-990,+10]):
    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
    log.append([mean_reward, threshold])

    # clear_output(True)
    print("mean reward = %.3f, threshold=%.3f"%(mean_reward, threshold))
    # plt.figure(figsize=[8,4])
    # plt.subplot(1,2,1)
    # plt.plot(list(zip(*log))[0], label='Mean rewards')
    # plt.plot(list(zip(*log))[1], label='Reward thresholds')
    # plt.legend()
    # plt.grid()

    # plt.subplot(1,2,2)
    # plt.hist(batch_rewards, range=reward_range);
    # plt.vlines([np.percentile(batch_rewards, percentile)], [0], [100], label="percentile", color='red')
    # plt.legend()
    # plt.grid()

def select_elites(states_batch, actions_batch, rewards_batch, percentile = 50):
    reward_threshold = np.percentile(rewards_batch, q = percentile)
    print(rewards_batch)

    elite_states  = np.empty(0)
    elite_actions = np.empty(0)

    for i in range(len(rewards_batch)):
        if (rewards_batch[i] >= reward_threshold):
            elite_states  = np.append(elite_states, [states_batch[i]])
            elite_actions = np.append(elite_actions, [actions_batch[i]])

    return elite_states, elite_actions

n_sessions = 100
percentile = 50
log = []

for i in range(50):
    #generate new sessions
    print(i)
    sessions = [generate_session() for _ in range(n_sessions)]

    batch_states, batch_actions, batch_rewards = map(np.array, zip(*sessions))

    elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards, percentile)
    elite_states = np.array(elite_states).reshape(-1, n_state)
    elite_actions = np.array(elite_actions)

    if (elite_states.shape[0] != 0):
        elite_actions_fit = []
        for a in elite_actions:
            print(a)
            cur = np.zeros((n_actions))
            cur[int(a)] = 1
            elite_actions_fit.append(cur)
        elite_actions = elite_actions_fit
        # print(elite_states, elite_actions)
        agent.fit(elite_states, elite_actions)

    # show_progress(batch_rewards, log, percentile, reward_range=[0, np.max(batch_rewards)])
    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
    print("mean reward = %.3f, threshold=%.3f" % (mean_reward,threshold))
