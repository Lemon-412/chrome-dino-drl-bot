from env.chrome_dino_simulator import ChromeDinoSimulator
from bot.a2c_bot import A2CBot
from utils.utils import ReplayMemory, discount_reward, agg_double_list
import numpy as np

MAX_EPISODES = 5000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 10
EVAL_INTERVAL = 20
ROLL_OUT_N_STEPS = 20
MEMORY_CAPACITY = ROLL_OUT_N_STEPS * 1
BATCH_SIZE = ROLL_OUT_N_STEPS
REWARD_DISCOUNTED_GAMMA = 0.85
ENTROPY_REG = 0.05
DONE_PENALTY = -10.
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
EPSILON_START = 0.50
EPSILON_END = 0.02
EPSILON_DECAY = 1000


def a2c_bot_dino_simulator_main():
    env = ChromeDinoSimulator()
    state_dim, action_dim = env.get_dim()
    state_dim *= 2
    bot = A2CBot(
        state_dim=state_dim, action_dim=action_dim,
        entropy_reg=ENTROPY_REG, max_grad_norm=MAX_GRAD_NORM,
        actor_hidden_size=64, critic_hidden_size=64,
        actor_lr=2e-3, critic_lr=5e-3,
        critic_loss=CRITIC_LOSS, optimizer_type="adam"
    )
    replay_memory = ReplayMemory(MEMORY_CAPACITY)
    episode = 0
    n_step_cnt = 0
    while episode < MAX_EPISODES:
        total_reward = 0
        env.reset()
        pre_state = None
        while not env.is_over():
            states, actions, rewards = [], [], []
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1 * n_step_cnt / EPSILON_DECAY)
            for i in range(ROLL_OUT_N_STEPS):
                state = env.get_state()
                if pre_state is None:
                    pre_state = state
                states.append(np.hstack((pre_state, state)))
                if np.random.rand() < epsilon:
                    action = np.random.choice(action_dim)
                else:
                    action = bot.action(np.hstack((pre_state, state)))
                actions.append(action)
                pre_state = state
                reward = -15 if env.is_over() else env.step(action)
                rewards.append(reward)
                total_reward += reward
                if env.is_over():
                    break
            n_step_cnt += 1
            tmp = np.hstack((pre_state, env.get_state()))
            remain_value = 0 if env.is_over() else bot.value(tmp, bot.action(tmp))
            rewards = discount_reward(rewards, remain_value, REWARD_DISCOUNTED_GAMMA)
            replay_memory.push(states, actions, rewards)
            if episode >= EPISODES_BEFORE_TRAIN:
                batch = replay_memory.sample(BATCH_SIZE)
                bot.train(batch)
        episode += 1
        if episode % EVAL_INTERVAL == 0:
            rewards = []
            for i in range(EVAL_EPISODES):
                env.reset()
                pre_state = None
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
                rewards.append(rewards_i)
            rewards_mu, rewards_std = agg_double_list(rewards)
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1 * n_step_cnt / EPSILON_DECAY)
            print(f"Episode {episode}, epsilon {round(epsilon, 4)}, Average Reward {round(float(rewards_mu), 2)}")


if __name__ == '__main__':
    a2c_bot_dino_simulator_main()
