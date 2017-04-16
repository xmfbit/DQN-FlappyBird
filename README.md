## Flappy Bird With DQN

DQN is a technology to realize reinforcement learning, first proposed by Deep Mind in NIPS13([paper in arxiv](https://arxiv.org/pdf/1312.5602v1.pdf)), whose input is raw pixels and whose output is a value function estimating future rewards. Using Experience Replay, they overcame the problem of network training.

This demo is about using DQN to train a convolutional neural network to play flappy bird game. It is a practice when I learned reinforcement learning and partly reused [songrotek's code](https://github.com/songrotek/DRL-FlappyBird), especially the game engine and basic idea. Thanks for sharing, thanks to the spirit and community of open source.

A video of the demo can be found on [YouTube](https://youtu.be/h4jEdF_roXU) or [优酷 Youku](http://v.youku.com/v_show/id_XMjcwOTcwMjYzMg==.html?spm=a2hzp.8253869.0.0&from=y1.7-2#paction) if you don't have access to YouTube.

### DQN implemented by PyTorch

PyTorch is an elegant framework published by Facebook. I implemented the neural network and training/testing procedure using PyTorch. So you need install PyTorch to run this demo. Besides, pygame package is needed by the game engine.

### How to run the demo

#### Play the game with pretrained model

At the beginning, you can play the game with a pretrained model by me. You can download the pretrained model from [Google Drive](https://drive.google.com/file/d/0B98MUaCGMMG0em1uQzkzYmt3U00/view?usp=sharing) (or [Baidu Netdisk](https://pan.baidu.com/s/1pKOpRqr) if Google Drive is not available) and use the following commands to play the game. Make sure that the pretrained model is in the root directory of this project.

```
chmod +x play.sh
./play.sh
```

For more detail infomation about the meaning of the arguments of the program, run `python main.py --help` or refer to the code in `main.py`.

#### Train DQN

You can use the following commands to train the model from scrach or finetuning(if pretrained weight file is provided).

```
chmod +x train.sh
./train.sh   # please see `main.py` for detail info about the variables
```

Some tips for training:

- Do not set `memory_size` too large(or too small). It depends on available memory in your computer.   

- It takes a long time to complete training. I finetuned the model several times and change `epsilon` of ϵ-greedy exploration manually every time.

- When choose action randomly in training, I prefer to 'DO Nothing' compared to 'UP'. I think it can accelarate convergence. See `get_action_randomly` method in `BrainDQN.py` for detail.

### Disclaimer

This work is based on the repo [songrotek/DRL-FlappyBird](https://github.com/songrotek/DRL-FlappyBird) and [yenchenlin1994/DeepLearningFlappyBird](https://github.com/yenchenlin1994/DeepLearningFlappyBird.git). Thanks two authors!
