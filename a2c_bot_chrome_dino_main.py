from env.chrome_dino import ChromeDino
from bot.a2c_bot import A2CBot
from utils.utils import ReplayMemory, discount_reward, agg_double_list
import numpy as np
from time import time, sleep
from os import makedirs

MAX_EPISODES = 1000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 5
EVAL_INTERVAL = 10
ROLL_OUT_N_STEPS = 20
MEMORY_CAPACITY = ROLL_OUT_N_STEPS
BATCH_SIZE = ROLL_OUT_N_STEPS
REWARD_DISCOUNTED_GAMMA = 0.65
ENTROPY_REG = 0.00
DONE_PENALTY = -10.
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
EPSILON_START = 0.50
EPSILON_END = 0.0
EPSILON_DECAY = 100


def a2c_main():
    env = ChromeDino()
    state_dim, action_dim = env.get_dim()
    state_dim *= 2

    bot = A2CBot(
        state_dim=state_dim, action_dim=action_dim,
        entropy_reg=ENTROPY_REG, max_grad_norm=MAX_GRAD_NORM,
        actor_hidden_size=64, critic_hidden_size=64,
        actor_lr=2e-3, critic_lr=2e-3,
        critic_loss=CRITIC_LOSS, optimizer_type="rmsprop"
    )
    replay_memory = ReplayMemory(MEMORY_CAPACITY)
    episode = 0
    n_step_cnt = 0
    path = "./models/" + str(int(time()))

    print(f"time: {time()} a2c solution for chrome://dino")
    print("Game will start in 3 seconds...")
    sleep(3)
    while episode < MAX_EPISODES:
        total_reward = 0
        env.reset()
        pre_state = None
        sleep(1)
        while not env.is_over():
            states, actions, rewards = [], [], []
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1 * n_step_cnt / EPSILON_DECAY)
            for _ in range(ROLL_OUT_N_STEPS):
                state = env.get_state()
                if pre_state is None:
                    pre_state = state
                compressed_state = np.hstack((pre_state, state))
                states.append(compressed_state)
                action = bot.action(compressed_state)
                pre_state = state
                print(f"{'.!'[action]}", end="", flush=True)
                if np.random.rand() < epsilon:
                    action = np.random.choice(action_dim)
                actions.append(action)
                reward = -50 if env.is_over() else env.step(action)
                rewards.append(reward)
                total_reward += reward
                if env.is_over():
                    break
            n_step_cnt += 1
            compressed_state = np.hstack((pre_state, env.get_state()))
            remain_value = 0 if env.is_over() else float(bot.value(compressed_state, bot.action(compressed_state)))
            rewards = discount_reward(rewards, remain_value, REWARD_DISCOUNTED_GAMMA)
            replay_memory.push(states, actions, rewards)
            if episode >= EPISODES_BEFORE_TRAIN:
                batch = replay_memory.sample(BATCH_SIZE)
                loss = bot.train(batch)
                print(f" (remain={round(remain_value, 1)} actor_loss={round(loss[0], 3)}, critic_loss={round(loss[1], 3)}) ==> {[round(x, 1) for x in rewards]}")
        print(f"eisode {episode} total_reward: {total_reward}\n")
        # print(f"{episode % 10 if episode % 10 != 0 else episode % 100 // 10}", end="", flush=True)
        episode += 1
        if episode % EVAL_INTERVAL == 0:
            rewards = []
            sleep(1)
            print(f"\nEvaluation Start...")
            for i in range(EVAL_EPISODES):
                env.reset()
                pre_state = None
                sleep(1)
                total_reward = 0
                rewards_i = []
                while True:
                    state = env.get_state()
                    if pre_state is None:
                        pre_state = state
                    if env.is_over():
                        break
                    action = bot.action(np.hstack((pre_state, state)))
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
            print("=" * 100)
            print(f"Episode {episode}, Average Reward {round(float(rewards_mu), 2)}")
            print("=" * 100)
            file_name = f"/Episode_{episode}_Reward_{round(float(rewards_mu), 2)}"
            makedirs(path + file_name)
            bot.save_model(path + file_name)
            sleep(1)
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1 * n_step_cnt / EPSILON_DECAY)
            print(f"Training Start... epsilon={round(epsilon, 5)}")
        sleep(1)


if __name__ == '__main__':
    env = ChromeDino()
    ret = env.get_state()
    for i in range(10, 17):
        for j in range(1, 47):
            print(ret[(i - 10) * 46 + j - 1], end="")
        print()
    sleep(5)
    a2c_main()
