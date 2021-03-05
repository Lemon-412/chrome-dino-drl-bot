from collections import namedtuple
from random import sample
import numpy as np


class ReplayMemory(object):
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


def discount_reward(rewards, remain_value, reward_discounted_gamma):
    discounted_r = [0.0 for _ in range(len(rewards))]
    running_add = remain_value
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * reward_discounted_gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def agg_double_list(l):
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std
