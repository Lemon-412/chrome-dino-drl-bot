from bot.naive_bot.naive_bot import NaiveBot
from env.chrome_dino import ChromeDino
from time import sleep

if __name__ == '__main__':
    env = ChromeDino()
    bot = NaiveBot()
    epoch_cnt = 0
    print("game will start in 3 seconds...")
    sleep(3)
    while True:
        print(f"\nepoch count: {epoch_cnt}, game start!")
        reward = 0
        env.reset()
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
            if reward % 50 == 0:
                print("\n", end="", flush=True)
        print(f"\nepoch count: {epoch_cnt}, game over! reward = {reward}\n")
        epoch_cnt += 1
        sleep(1)
