from PIL import ImageGrab


class NaiveBot:
    def __init__(self):
        self.__x1, self.__x2, self.__y1, self.__y2 = 540, 660, 600, 680

    def __detect_enemy(self, scene):
        aux_color = scene.getpixel((int(self.__x1), self.__y1))
        for x in range(self.__x1, self.__x2):
            for y in range(self.__y1, self.__y2):
                color = scene.getpixel((x, y))
                if color != aux_color:
                    return True
        return False

    def __update_strategy(self):
        self.__x1 += 0.6
        self.__x2 += 0.6

    def run(self, state):
        is_enemy = self.__detect_enemy(state)
        if is_enemy:
            return 1
        else:
            return 0
