# Code Release for Learning Force Control for Legged Manipulation


# Table of contents
1. [Overview](#overview)
2. [System Requirements](#requirements)
3. [Training a Model](#simulation)
    1. [Installation](#installation)
    2. [Environment and Model Configuration](#configuration)
    3. [Training and Logging](#training)
    4. [Analyzing the Policy](#analysis)

## Overview <a name="introduction"></a>

This repository provides an implementation of the paper:


<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://tif-twirl-13.github.io/learning-compliance.html" target="_blank">
      <b> Learning Force Control for Legged Manipulation </b>
      </a>
      <br>
      <a href="https://tif-twirl-13.github.io/Home.html" target="_blank">Tifanny Portela</a> and <a href="https://gmargo11.github.io/" target="_blank">Gabriel B. Margolis</a> and <a href="https://yandongji.github.io/" target="_blank">Yandong Ji</a> and <a href="https://people.csail.mit.edu/pulkitag" target="_blank">Pulkit Agrawal</a>
      <br>
      <em>International Conference on Robotics and Automation</em>, 2024
      <br>
      <a href="">paper</a> /
      <a href="https://tif-twirl-13.github.io/learning-compliance.html" target="_blank">project page</a>
    <br>
</td>

<br>

This environment builds on the [legged gym environment](https://leggedrobotics.github.io/legged_gym/) by Nikita
Rudin, Robotic Systems Lab, ETH Zurich (Paper: https://arxiv.org/abs/2109.11978) and the Isaac Gym simulator from 
NVIDIA (Paper: https://arxiv.org/abs/2108.10470). Training code builds on the 
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) repository, also by Nikita
Rudin, Robotic Systems Lab, ETH Zurich. All redistributed code retains its
original [license](LICENSES/legged_gym/LICENSE).

Our initial release provides the following features:
* Train a force control and locomotion policy for the Unitree B1 with Z1 arm.

## System Requirements <a name="requirements"></a>

**Simulated Training and Evaluation**: Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 10GB of VRAM. The code can run on a smaller GPU if you decrease the number of parallel environments (`Cfg.env.num_envs`). However, training will be slower with fewer environments.

## Training a Model <a name="simulation"></a>

### Installation <a name="installation"></a>

#### Install pytorch 1.10 with cuda-11.3:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### Install Isaac Gym

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. now install the python package
    ```bash
    cd isaacgym/python && pip install -e .
    ```
4. Verify the installation by try running an example

    ```bash
    python examples/1080_balls_of_solitude.py
    ```
5. For troubleshooting check docs `isaacgym/docs/index.html`

#### Install the `b1_gym` package

In this repository, run `pip install -e .`

### Verifying the Installation

If everything is installed correctly, you should be able to run the test script with:

```bash
python scripts/test.py
```

You should see a GUI window with 10 B1+Z1 robots standing in place.

### Environment and Model Configuration <a name="configuration"></a>


**CODE STRUCTURE** The main environment for simulating a legged robot is
in [legged_robot.py](b1_gym/envs/base/legged_robot.py). The default configuration parameters including reward
weightings are defined in [legged_robot_config.py::Cfg](b1_gym/envs/base/legged_robot_config.py). Note that many 
are overridden in `train.py`.

There are three scripts in the [scripts](scripts/) directory:

```bash
scripts
├── __init__.py
├── play.py
├── test.py
└── train.py
```

You can run the `test.py` script to verify your environment setup. If it runs then you have installed the gym
environments correctly. To train an agent, run `train.py`. To evaluate a trained agent, run `play.py`. 


### Training and Logging <a name="training"></a>

To train the Go1 controller from [Walk these Ways](https://sites.google.com/view/gait-conditioned-rl/), run: 

```bash
python scripts/train.py
```

The script should print `Saving model at iteration 0`. It will log the training result to weights and biases. To visualize training progress, you can visit your weights and biases dashboard.

The GUI is off during training by default. To turn it on, set `headless=False` at the bottom of `train.py` i.e. `train_b1_z1_IK(headless=False)`

Training with the default configuration requires about 12GB of GPU memory. If you have less memory available, you can 
still train by reducing the number of parallel environments used in simulation (the default is `Cfg.env.num_envs = 4000`).

### Analyzing the Policy <a name="analysis"></a>

To evaluate a model, first edit the variable in `play.py` to point to its location in weights & biases:

```bash
RUN_PATH = [your_wandb_run]
```

e.g.

```bash
RUN_PATH = "robot-locomotion/b1-loco-z1-manip/runs/voq40aun" 
```

Then run:

```bash
python scripts/play.py
```

