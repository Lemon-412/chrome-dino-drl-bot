from env.chrome_dino_simulator import ChromeDinoSimulator
from bot.naive_bot import NaiveBot
from time import sleep

MAX_EPISODES = 20


def print_state(scene):
    for i in range(10, 17):
        for j in range(1, 47):
            print(scene[(i - 10) * 46 + j - 1], end="")
        print()


def naive_bot_dino_simulator_main():
    env = ChromeDinoSimulator()
    bot = NaiveBot()
    episode = 0
    while episode < MAX_EPISODES:
        total_reward = 0
        env.reset()
        while True:
            state = env.get_state()
            if env.is_over():
                break
            action = bot.run(state)
            # print_state(state)
            # print(action)
            # input()
            total_reward += env.step(action)
        print(f"Episode {episode} reward={total_reward}")
        episode += 1
        sleep(0.1)


if __name__ == '__main__':
    naive_bot_dino_simulator_main()