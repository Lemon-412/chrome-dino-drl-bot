import pyautogui
from PIL import ImageGrab
import numpy as np

SCREEN_RESIZE = tuple([48, 27])
CUT_RESIZE = tuple([1, 10, 47, 17])


class ChromeDino:
    def __init__(self):
        self.__is_reset = False
        self.__prev_screen = None

    @staticmethod
    def get_dim():
        return (CUT_RESIZE[2] - CUT_RESIZE[0]) * (CUT_RESIZE[3] - CUT_RESIZE[1]), 2

    def is_over(self):
        return not self.__is_reset

    def reset(self):
        assert not self.__is_reset
        pyautogui.press("up")
        self.__is_reset = True

    def get_raw_state(self):
        screen = ImageGrab.grab().resize(SCREEN_RESIZE).convert("L")
        if screen == self.__prev_screen:
            self.__is_reset = False
        else:
            self.__prev_screen = screen
        return self.__prev_screen

    def get_state(self):
        screen = ImageGrab.grab().resize(SCREEN_RESIZE).convert("L").crop(CUT_RESIZE)
        if screen == self.__prev_screen:
            self.__is_reset = False
        else:
            self.__prev_screen = screen
        ret = np.array(screen).flatten()
        ret[ret < 60] = 0
        ret[ret >= 60] = 1
        return ret

    def step(self, action):
        assert action in {0, 1}, "invalid action!"
        assert self.__is_reset, "reset the game before taking any action!"
        if action == 1:
            pyautogui.press("up")
        return 1
