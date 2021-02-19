import gym


class CartPole:
    def __init__(self):
        self.__env = gym.make("CartPole-v0")
        self.__is_over = True
        self.__prev_state = None

    @staticmethod
    def get_dim():
        return 4, 2

    def is_over(self):
        return self.__is_over

    def reset(self):
        self.__prev_state = self.__env.reset()
        self.__is_over = False

    def get_state(self):
        return self.__prev_state

    def step(self, action):
        assert action in {0, 1}, "invalid action!"
        self.__prev_state, reward, self.__is_over, _ = self.__env.step(action)
        return reward
