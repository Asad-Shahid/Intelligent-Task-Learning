# Intelligent-Task-Learning
The Repository for the _Autonomous Robots 2022 Journal article:_

"[Continuous control actions learning and adaptation for robotic manipulation through reinforcement learning](https://link.springer.com/article/10.1007/s10514-022-10034-z)" 

This contains a python toolkit for learning a grasping task with Franka Emika Panda Robot. The robot can be trained to grasp the cube, avoid obstacles and learn to manage redundancy using modern Reinforcement Learning algorithms of [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) and [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905). It is powered by [MuJoCo physics engine](http://www.mujoco.org/) 


# How to cite
```
@article{shahid2022continuous,
  title={Continuous control actions learning and adaptation for robotic manipulation through reinforcement learning},
  author={Shahid, Asad Ali and Piga, Dario and Braghin, Francesco and Roveda, Loris},
  journal={Autonomous Robots},
  pages={1--16},
  year={2022},
  publisher={Springer}
}
```

# New Experiment (Dynamic Enviornment)
The adapted policy can grasp the moving cube in 30 mints of retraining.
![](adapted_policy.gif)

Before, the base grasping policy trained on a static cube is not able to grasp the moving cube.
![](base_policy.gif)

# Simulation Video
[![Video](https://img.youtube.com/vi/aX55Zc2XMTE/maxres3.jpg)](https://www.youtube.com/watch?v=aX55Zc2XMTE)

# Experimental validation
[![Video](https://github.com/Asad-Shahid/Intelligent-Task-Learning/blob/master/exp_image.png)](https://drive.google.com/file/d/1zlS-_HIWMlIAvrxqGNGRyMbuDfQrws8z/view)


## Installation

To use this toolkit, it is required to first install [MuJoCo 200](https://www.roboti.us/index.html) and then [mujoco-py](https://github.com/openai/mujoco-py) from Open AI. mujoco-py allows using MuJoCo from python interface.
The installation requires python 3.6 or higher. It is recommended to install all the required packages under a conda virtual environment


## References
This toolit is mainly developed based on [Surreal Robotics Suite](https://github.com/StanfordVL/robosuite) and the Reinforcement learning part is referenced from
[this repo](https://github.com/clvrai/furniture)
