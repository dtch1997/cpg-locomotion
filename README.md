# Perceptual Locomotion with Controllable Pace and Natural Gait Transitions Over Uneven Terrain

Codebase supporting the CoRL 2022 paper submission: `Perceptual Locomotion with Controllable Pace and Natural Gait Transitions Over Uneven Terrain`. 

## Setup

Our repository works for Python 3.8+. 
Install dependencies and source in developer mode:

```
python -m pip install -e .
```

## Training heightmap eutoencoder
First, generate the data used in training the heightmap autoencoder:
```
python blind_walking/examples/hover_robot.py -n 1800
```
A few notes on the script:
* `-n 1800` sets the number of timesteps collected to be 1800.
* `--record` to record the hovering robot's trajectory over different terrains.

Shift the generated `heightmap.npy` data file to a folder `./blind_walking/examples/data/heightmap.npy`. Then, train the heightmap autoencoder:
```
python train_autoencoder.py
```

## Loading an Existing Model

A pre-trained locomotion policy is provided in `saved_models/policy`.  
To run the policy in Pybullet simulation in headless mode: 

```
python scripts/enjoy_with_logging.py --algo ppo --env A1GymEnv-v0 -f saved_models/policy --no-render
```

The `--record` flag can optionally be added to record the episode as a video. 
Video parameters such as camera angles can be configured in `blind_walking.envs.locomotion_gym_config.SimulationParameters`. 

## Training a New Model

To reproduce the training settings from the paper:

```
python train.py --algo ppo --env A1GymEnv-v0 -f logs --n-timesteps 4000000
```

# Acknowledgements

This codebase draws inspiration from the following codebases: 
- [Stable Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Motion Imitation](https://github.com/erwincoumans/motion_imitation)
