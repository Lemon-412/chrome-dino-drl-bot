import pyautogui
from PIL import ImageGrab
import numpy as np

SCREEN_RESIZE = tuple([192, 108])
CUT_RESIZE = tuple([5, 45, 190, 69])


class ChromeDino:
    def __init__(self):
        self.__is_reset = False
        self.__prev_score = None

    def is_over(self):
        return not self.__is_reset

    def reset(self):
        assert not self.__is_reset
        pyautogui.press("up")
        self.__is_reset = True

    def get_raw_state(self):
        screen = ImageGrab.grab().resize(SCREEN_RESIZE).convert("L")
        if screen == self.__prev_score:
            self.__is_reset = False
        else:
            self.__prev_score = screen
        return self.__prev_score

    def get_state(self):
        screen = ImageGrab.grab().resize(SCREEN_RESIZE).convert("L").crop(CUT_RESIZE)
        if screen == self.__prev_score:
            self.__is_reset = False
        else:
            self.__prev_score = screen
        ret = np.array(screen).flatten()
        ret[ret < 100] = 0
        ret[ret >= 100] = 1
        return ret

    def step(self, action):
        assert action in {0, 1}, "invalid action!"
        assert self.__is_reset, "reset the game before taking any action!"
        if action == 1:
            pyautogui.press("up")
        return 1
