import matplotlib.pyplot as plt

x1, y1 = [], []
file = open("eval.txt")
for line in file.readlines():
    a, b = map(float, line.split())
    x1.append(a)
    y1.append(b)

x2, y2 = [], []
file = open("train.txt")
for line in file.readlines():
    a, b = map(float, line.split())
    x2.append(a)
    y2.append(b)

x3, y3 = [], []
tot = 0.0
for i in range(len(x2)):
    tot += y2[i]
    x3.append(i)
    y3.append(tot / (i + 1))

x4, y4 = [], []
tot = 0.0
for i in range(len(x1)):
    tot += y1[i]
    x4.append(x1[i])
    y4.append(tot / (i + 1))

plt.title("A2C Implementation on Chrome Dino Game")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.plot(x2, y2, label="Training Round", color='paleturquoise')
plt.plot(x1, y1, label="Evaluation Round", color='plum')
plt.plot(x3, y3, label="Training Average", color='dodgerblue')
plt.plot(x4, y4, label="Evaluation Average", color='darkorchid')
plt.legend()
plt.show()
