from env.chrome_dino import ChromeDino
from bot.naive_bot import NaiveBot
from bot.a2c_bot import A2CBot
from time import time, sleep

import gym
import numpy as np

MAX_EPISODES = 5000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# roll out n steps
ROLL_OUT_N_STEPS = 10
# only remember the latest ROLL_OUT_N_STEPS
MEMORY_CAPACITY = ROLL_OUT_N_STEPS
# only use the latest ROLL_OUT_N_STEPS for training A2C
BATCH_SIZE = ROLL_OUT_N_STEPS

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.00
#
DONE_PENALTY = -10.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2017


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std


def run(env_id="CartPole-v0"):
    env = gym.make(env_id)
    env.seed(RANDOM_SEED)
    env_eval = gym.make(env_id)
    env_eval.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    if len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # print(state_dim, action_dim)

    a2c = A2CBot(
        env=env, memory_capacity=MEMORY_CAPACITY,
        state_dim=state_dim, action_dim=action_dim,
        batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
        done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS,
        reward_gamma=REWARD_DISCOUNTED_GAMMA,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
        critic_loss=CRITIC_LOSS
    )

    episodes = []
    eval_rewards = []
    while a2c.n_episodes < MAX_EPISODES:
        a2c.interact()
        if a2c.n_episodes >= EPISODES_BEFORE_TRAIN:
            a2c.train()
        if a2c.episode_done and ((a2c.n_episodes+1) % EVAL_INTERVAL == 0):
            rewards, _ = a2c.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (a2c.n_episodes+1, rewards_mu))
            episodes.append(a2c.n_episodes+1)
            eval_rewards.append(rewards_mu)


if __name__ == '__main__':
    run()
    input("press enter:")
    env = ChromeDino()
    # bot = NaiveBot()
    bot = A2CBot
    episode = 0
    print("game will start in 3 seconds...")
    sleep(3)
    while True:
        print(f"Episode {episode}, game start!")
        reward = 0
        env.reset()
        episode_start_time = time()
        while True:
            state = env.get_state()
            if env.is_over():
                break
            reward += 1
            action = bot.run(state)
            env.step(action)
            if action == 0:
                print(".", end="", flush=True)
            else:
                print("!", end="", flush=True)
            if reward % 100 == 0:
                print("\n", end="", flush=True)
        episode_end_time = time()
        episode_time = round(episode_end_time - episode_start_time, 2)
        fps = round(reward / episode_time, 1)
        print(f"\nEpisode {episode} ended in {episode_time}sec (FPS={fps})! reward={reward}\n")
        episode += 1
        sleep(1)
