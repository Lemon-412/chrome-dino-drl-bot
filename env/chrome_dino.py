import pyautogui
from PIL import ImageGrab
from copy import deepcopy

SCREEEN_RESIZE = tuple([1920, 1080])


class ChromeDino:
    def __init__(self):
        self.__is_reset = False
        self.__reward = None
        self.__prev_score = None

    def is_over(self):
        return not self.__is_reset

    def reset(self):
        pyautogui.press("up")
        self.__is_reset = True

    def get_state(self):
        screen = ImageGrab.grab().resize(SCREEEN_RESIZE).convert("L")
        # print(screen)
        # print("==============")
        # print(self.__prev_score)
        # print("=============")
        # print(screen == self.__prev_score)
        # print(screen is self.__prev_score)
        # input()
        if screen == self.__prev_score:
            self.__is_reset = False
        else:
            self.__prev_score = deepcopy(screen)
        return self.__prev_score

    def step(self, action):
        assert action in {0, 1}, "invalid action!"
        assert self.__is_reset, "reset the game before taking any action!"
        if action == 1:
            pyautogui.press("up")
