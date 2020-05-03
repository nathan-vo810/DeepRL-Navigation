[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Deep Reinforcement Learning Project: Navigation

## Project Details

This project aims to train an agent to walk in a virtual environment (Banana Unity environment) and collect as many yellow bananas as possible while avoiding purple ones.

The trained agent in this project is a Deep Q-Network (DQN) based agent.

![Trained Agent][image1]

### The environment

The environment chosen for the project is similar but not identical to the version of the [Banana 
Collector Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector)** from the Unity ML-Agents toolkit.

* A reward of +1 is provided for collecting a yellow banana and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. Note that whenever the agent collects a banana, a new banana is spawn at a random place in the planar environment.

* The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
    - **`0`** - move forward.
    - **`1`** - move backward.
    - **`2`** - turn left.
    - **`3`** - turn right.

* The task is episodic and the criteria solving the environment is to get **an average score of +13 over 100 consecutive episodes**.

## Getting Started

### Downloading the environment

Download the environment from one of the links below.  You need only select the environment that matches your operating system:

Platform | Link
-------- | -----
Linux             | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX           | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

Unzip (or decompress) the downloaded file and store the path of the executable as we will need the path to input on `Navigation.ipynb`. 

Note that this environment is compatible only with an older version of the ML-Agents toolkit (version 0.4.0), the next setup section will take care of this.

### Dependencies

Please follow the instructions on [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the required libraries.

## Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent or load the trained model and visualize the agent collecting the bananas!