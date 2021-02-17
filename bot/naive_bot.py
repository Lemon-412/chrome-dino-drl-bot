class NaiveBot:
    def __init__(self):
        self.__x1, self.__x2, self.__y1, self.__y2 = 54, 66, 58, 66

    def __detect_enemy(self, scene):
        aux_color = scene.getpixel((int(self.__x1), self.__y1))
        for x in range(self.__x1, self.__x2):
            for y in range(self.__y1, self.__y2):
                color = scene.getpixel((x, y))
                if color != aux_color:
                    return True
        return False

    def run(self, state):
        is_enemy = self.__detect_enemy(state)
        return 1 if is_enemy else 0
