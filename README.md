
## Prerequisites
> pip install gym gym_super_mario_bros ale-py gym[accept-rom-license]

## Training
> python trainer.py

To train with CPU
> python trainer.py --device cpu

You may find the AI playback videos in the following folder

```
checkpoints/[game-name]/video
```

## Switching a game
Find the following code snippet in trainer.py, change the parameters as you wish.
```
ppo = PPO(games.create_mario_profile, dict(world=1, stage=1), 4)
```
