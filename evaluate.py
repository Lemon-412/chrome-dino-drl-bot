from env.chrome_dino import ChromeDino
from bot.a2c_bot import A2CBot
import numpy as np
from time import sleep

MAX_EPISODES = 50
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


if __name__ == '__main__':
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

    ret = env.get_state()
    for i in range(10, 17):
        for j in range(1, 47):
            print(ret[(i - 10) * 46 + j - 1], end="")
        print()
    sleep(5)

    bot.load_model("./models/1615433372/Episode_540_Reward_470.8")
    print(f"Evaluation Start...")

    for episode in range(MAX_EPISODES):
        rewards = []
        sleep(1)
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
            # print(f"{'.!'[action]}", end="", flush=True)
            # if total_reward % 100 == 0:
            #     print("\n", end="", flush=True)
        rewards.append(rewards_i)
        print(f"Evaluation episode {episode}, score={total_reward}")
        sleep(1)
