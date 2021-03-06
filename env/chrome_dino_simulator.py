import random
import numpy as np

SCREEN_RESIZE = tuple([48, 27])
CUT_RESIZE = tuple([1, 10, 47, 17])

TREE_TYPE = [
    {
        "size": [3, 4],
        "name": "big",
        "sample": [
            [
                [0, 0, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 0]
            ],
            [
                [0, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1]
            ],
            [
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0]
            ],
            [
                [0, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 0]
            ],
            [
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0]
            ],
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0]
            ]
        ]
    },
    {
        "size": [2, 2],
        "name": "small",
        "sample": [
            [
                [0, 0],
                [1, 1]
            ],
            [
                [0, 1],
                [0, 1]
            ],
            [
                [0, 1],
                [1, 1]
            ]
        ]
    },
    {
        "size": [2, 3],
        "name": "ss",
        "sample": [
            [
                [0, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 0],
                [1, 1, 1]
            ]
        ]
    },
    {
        "size": [2, 5],
        "name": "sss",
        "sample": [
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0]
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1]
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ],
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0]
            ]
        ]
    },
    {
        "size": [3, 5],
        "name": "bb",
        "sample": [
            [
                [1, 0, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0]
            ],
            [
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0]
            ],
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0]
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0]
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0]
            ]
        ]
    },
    {
        "size": [3, 8],
        "name": "bbsb",
        "sample": [
            [
                [0, 1, 0, 1, 0, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0]
            ],
            [
                [0, 1, 0, 1, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1]
            ],
            [
                [0, 0, 1, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0]
            ],
            [
                [0, 1, 0, 1, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0]
            ]
        ]
    }
]

DINOSAUR = [
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0]
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
]

SPEED = [2, 3, 3, 2, 4, 3, 2, 2, 3, 3]


class Tree:
    def __init__(self):
        self.tree_type = random.randint(0, 5)

    def name(self):
        return tuple(TREE_TYPE[self.tree_type]["name"])

    def size(self):
        return tuple(TREE_TYPE[self.tree_type]["size"])

    def get_sample(self):
        tmp = random.randint(0, len(TREE_TYPE[self.tree_type]["sample"]) - 1)
        return TREE_TYPE[self.tree_type]["sample"][tmp]


class ChromeDinoSimulator:
    def __init__(self):
        self.__is_reset = False
        self.__size = tuple([CUT_RESIZE[3] - CUT_RESIZE[1], CUT_RESIZE[2] - CUT_RESIZE[0]])
        self.__prev_screen = None
        self.__dino_status = None
        self.__obstacles = None
        self.__obstacle_span_tip = None

    @staticmethod
    def get_dim():
        return (CUT_RESIZE[2] - CUT_RESIZE[0]) * (CUT_RESIZE[3] - CUT_RESIZE[1]), 2

    def __paint_dino(self):
        for i in range(self.__size[0]):
            for j in range(len(DINOSAUR[0][0])):
                self.__prev_screen[i][j] = DINOSAUR[self.__dino_status][i][j]

    def __paint_obstacles(self):
        for elem in self.__obstacles:
            start_pos = elem["pos"]
            sample = elem["tree"].get_sample()
            x, y = elem["tree"].size()
            for i in range(x):
                for j in range(y):
                    real_x = self.__size[0] - x + i
                    real_y = start_pos + j
                    if 0 <= real_x < self.__size[0] and 0 <= real_y < self.__size[1] and sample[i][j]:
                        if self.__prev_screen[real_x][real_y] != 0:
                            self.__is_reset = False
                        self.__prev_screen[real_x][real_y] = 1

    def is_over(self):
        return not self.__is_reset

    def reset(self):
        assert not self.__is_reset
        self.__is_reset = True
        self.__prev_screen = [[0 for _ in range(self.__size[1])] for _ in range(self.__size[0])]
        self.__obstacles = []
        self.__dino_status = 0
        self.__obstacle_span_tip = 10
        self.__paint_dino()

    def get_state(self):
        return np.array(self.__prev_screen).flatten()

    def step(self, action):
        assert action in {0, 1}, "invalid action!"
        assert self.__is_reset, "reset the game before taking any action!"
        if self.__dino_status != 0:
            self.__dino_status = (self.__dino_status + 1) % len(DINOSAUR)
        elif action == 1:
            self.__dino_status = 1
        self.__prev_screen = [[0 for _ in range(self.__size[1])] for _ in range(self.__size[0])]
        self.__paint_dino()
        speed = SPEED[random.randint(0, len(SPEED) - 1)]
        for elem in self.__obstacles:
            elem["pos"] -= speed
        while len(self.__obstacles) != 0:
            if self.__obstacles[0]["pos"] < -10:
                del self.__obstacles[0]
            else:
                break
        if self.__obstacle_span_tip > 0:
            self.__obstacle_span_tip -= 1
        elif random.random() < 0.5:
            self.__obstacle_span_tip = 10
            self.__obstacles.append({"pos": self.__size[1], "tree": Tree()})
        self.__paint_obstacles()
        return 1


if __name__ == '__main__':
    env = ChromeDinoSimulator()
    while True:
        total_reward = 0
        env.reset()
        while True:
            ret = env.get_state()
            if env.is_over():
                break
            for i in range(10, 17):
                for j in range(1, 47):
                    print(ret[(i - 10) * 46 + j - 1], end="")
                print()
            print()
            action = int(input("action: "))
            total_reward += env.step(action)
        print(f"episode end: reward={total_reward}\n")
