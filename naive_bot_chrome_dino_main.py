from bot.naive_bot import NaiveBot
from env.chrome_dino import ChromeDino
from time import sleep

MAX_EPISODES = 50


def naive_bot_chrome_dino_main():
    env = ChromeDino()
    bot = NaiveBot()
    sleep(3)
    for episode in range(MAX_EPISODES):
        env.reset()
        sleep(1)
        total_reward = 0
        while True:
            action = bot.run(env.get_state())
            if env.is_over():
                break
            reward = env.step(action)
            total_reward += reward
            # print(f"{'.!'[action]}", end="", flush=True)
            # if total_reward % 100 == 0:
            #     print("\n", end="", flush=True)
        print(f"Evaluation episode {episode}, score={total_reward}")
        sleep(1)


if __name__ == '__main__':
    naive_bot_chrome_dino_main()
