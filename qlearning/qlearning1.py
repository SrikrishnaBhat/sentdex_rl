import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

done = False

LEARNING_RATE = 0.1
DISCOUNT = 0.95
episodes = 20000
DISPLAY_INTERVAL = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/ DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

EPS = 0.5
EPS_DECAY_START = 1
EPS_DECAY_END = episodes // 2
EPS_DECAY = EPS / (EPS_DECAY_END - EPS_DECAY_START)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(episodes):
    print(episode)
    if episode % DISPLAY_INTERVAL == 0:
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > EPS:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print(f"Done on episode: {episode}!")

        discrete_state = new_discrete_state

    if EPS_DECAY_START <= episode <= EPS_DECAY_END:
        EPS -= EPS_DECAY

env.close()
