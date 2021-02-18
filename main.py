from env.chrome_dino import ChromeDino
from env.cart_pole import CartPole
from bot.naive_bot import NaiveBot
from bot.a2c_bot import A2CBot

import numpy as np
from time import time, sleep
from collections import namedtuple
from random import sample

MAX_EPISODES = 10000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 10
EVAL_INTERVAL = 50
ROLL_OUT_N_STEPS = 10
MEMORY_CAPACITY = ROLL_OUT_N_STEPS
BATCH_SIZE = ROLL_OUT_N_STEPS
REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.00
DONE_PENALTY = -10.
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500


class ReplayMemory(object):
    """
    Replay memory buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.experience = namedtuple("Experience", ("states", "actions", "rewards", "next_states", "dones"))

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = sample(self.memory, batch_size)
        batch = self.experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


def discount_reward(rewards, remain_value):
    discounted_r = np.zeros_like(rewards)
    running_add = remain_value
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * REWARD_DISCOUNTED_GAMMA + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def agg_double_list(l):
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std


def naive_main():
    env = ChromeDino()
    bot = NaiveBot()
    episode = 0
    print("Game will start in 3 seconds...")
    sleep(3)
    while episode < MAX_EPISODES:
        total_reward = 0
        env.reset()
        episode_start_time = time()
        while True:
            state = env.get_raw_state()
            if env.is_over():
                break
            action = bot.run(state)
            total_reward += env.step(action)
            print(f"{'.!'[action]}", end="", flush=True)
            if total_reward % 100 == 0:
                print("\n", end="", flush=True)
        episode_end_time = time()
        episode_time = round(episode_end_time - episode_start_time, 2)
        fps = round(total_reward / episode_time, 1)
        print(f"\nEpisode {episode} ended in {episode_time}sec (FPS={fps})! reward={total_reward}\n")
        episode += 1
        sleep(1)


def a2c_main():
    env = ChromeDino()
    state_dim, action_dim = 4440, 2
    bot = A2CBot(
        state_dim=state_dim, action_dim=action_dim,
        entropy_reg=ENTROPY_REG, max_grad_norm=MAX_GRAD_NORM,
        critic_loss=CRITIC_LOSS
    )
    replay_memory = ReplayMemory(MEMORY_CAPACITY)
    episode = 0
    n_step_cnt = 0
    print("Game will start in 3 seconds...")
    sleep(3)
    while episode < MAX_EPISODES:
        total_reward = 0
        env.reset()
        sleep(1)
        while not env.is_over():
            states, actions, rewards = [], [], []
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1 * n_step_cnt / EPSILON_DECAY)
            for i in range(ROLL_OUT_N_STEPS):
                state = env.get_state()
                states.append(state)
                if np.random.rand() < epsilon:
                    action = np.random.choice(action_dim)
                else:
                    action = bot.action(state)
                actions.append(action)
                reward = -500 if env.is_over() else env.step(action)
                rewards.append(reward)
                total_reward += reward
                if env.is_over():
                    break
            n_step_cnt += 1
            remain_value = 0 if env.is_over() else bot.value(env.get_state(), bot.action(env.get_state()))
            rewards = discount_reward(rewards, remain_value)
            replay_memory.push(states, actions, rewards)
            if episode >= EPISODES_BEFORE_TRAIN:
                batch = replay_memory.sample(BATCH_SIZE)
                bot.train(batch)
        print(f"{episode % 10 if episode % 10 != 0 else episode % 100 // 10}", end="", flush=True)
        episode += 1
        if episode % EVAL_INTERVAL == 0:
            rewards = []
            sleep(1)
            print(f"\nEvaluation Start...")
            for i in range(EVAL_EPISODES):
                env.reset()
                sleep(1)
                total_reward = 0
                rewards_i = []
                while True:
                    state = env.get_state()
                    if env.is_over():
                        break
                    action = bot.action(state)
                    reward = env.step(action)
                    total_reward += reward
                    rewards_i.append(reward)
                    print(f"{'.!'[action]}", end="", flush=True)
                    if total_reward % 100 == 0:
                        print("\n", end="", flush=True)
                rewards.append(rewards_i)
                print(f"\nEvaluation episode {i}, score={total_reward}")
                sleep(1)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print(f"Episode {episode}, Average Reward {round(float(rewards_mu), 2)}")
            sleep(10)
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1 * n_step_cnt / EPSILON_DECAY)
            print(f"Training Start... epsilon={round(epsilon, 5)}")
        sleep(1)


if __name__ == '__main__':
    naive_main()
