# Chrome-Dino DRL Boot
[WIP]

基于 A2C 的 chrome 小恐龙彩蛋 解决方案，用于学习强化学习。

更多关于Chrome Dino的彩蛋的细节，请移步：[Google blog about Chrome Dino](https://www.blog.google/products/chrome/chrome-dino/)

推荐一个同主题更好的实现姿势：[Lumotheninja/dino-reinforcement-learning](https://github.com/Lumotheninja/dino-reinforcement-learning)



<img src="/imgs/dino.gif"> </img>





### 项目

项目结构：

- bot 文件夹下存放 a2c 算法和启发式算法。

- env 文件夹下存放 chrome dino 环境和（用来调参的）chrome dino 模拟器环境。

- utils 文件夹下存放用到的烂代码。

运行：

- 安装 `requirements.txt ` 
- 运行 `a2c_bot_chrome_dino_main.py` 



### 环境

~~环境很屎，目前兼容性为0~~

因为不会前端，只能通过不断截屏 + 挟持键盘操作跳跃键实现与环境的交互。

1. 使用 chrome 浏览器访问链接 [chrome://dino](chrome://dino)
2. 保持鼠标始终聚焦在浏览器上
3. 运行`env/chrome_dino.py` 调整游戏图像到合适的位置

在使用 2k 分辨率显示器，保持网页处于全屏状态，并使用深色 chrome 主题时，可以不调整图像位置。



### State

env 会不断将最新的游戏截取到屏幕中，bot 决策是否按下跳跃键。

将 t 时刻和 t-1 时刻的游戏画面截取合适的部分二值化后降低分辨率再 flatten 后拼接作为 state。

<img src="/imgs/state.png"> </img>

### 难点

1. 在线训练，即时性要求高
2. 输入为高维数据，需要图像处理
3. 惩罚延迟，当前时刻的决策可能要到若干时刻后才会产生后果
4. 环境和游戏本身的随机性



### 效果

人工智障。

运气好时（50%概率），训练200轮游戏实际分数可以在500分左右。

运气不好时可能根本炼不出来 QwQ。

<img src="/imgs/A2C_Implementation_on_Chrome_Dino_Game.png"> </img>

<img src="/imgs/A2C Evaluation Performance.png"> </img>



有这点算力，为什么不挖矿呢？（逃

强化学习没前途的，快跑！



### 参考

Naive Bot 和 环境抓取方案参考：[guilhermej/dino_chrome_bot](https://github.com/guilhermej/dino_chrome_bot)

A2C 算法参考：[ChenglongChen/pytorch-DRL](https://github.com/ChenglongChen/pytorch-DRL)

