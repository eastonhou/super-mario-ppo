from typing import Tuple
import gym, gym_super_mario_bros, cv2
import numpy as np
from gym.core import Env
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
from galois_common import gcutils

NORM_IMAGE_SIZE = (160, 160)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0
        for _ in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done: break
        return obs, total_reward, done, trunk, info

class CustomReward(gym.Wrapper):
    def __init__(self, env=None, world=None, stage=None):
        super().__init__(env)
        self.curr_score = 0
        self.current_x = 40
        self.status = 'small'
        self.world = world
        self.stage = stage

    def step(self, action):
        state, reward, done, trunk, info = self.env.step(action)
        reward += (info['score'] - self.curr_score) / 40
        self.curr_score = info['score']
        if self.status == 'small' and info['status'] == 'big': reward += 20
        elif self.status == 'big' and info['status'] == 'small': reward -= 20
        self.status = info['status']
        if done:
            if info['flag_get']:
                reward += 50
                info['state'] = 'success'
                del info['flag_get']
            else:
                reward -= 50
                info['state'] = 'fail'
        else:
            info['state'] = 'playing'
        reward += (info['x_pos'] - self.current_x) / 10
        self.current_x = info['x_pos']
        return state, reward / 10, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        self.status = 'small'
        return self.env.reset()

class Replay(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.memory = []

    def collect(self, state, action):
        next_state, reward, info = self.step(action)
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'info': info
        })
        return next_state, info

    def reset(self):
        self.memory = []
        return super().reset()

class FrameConverter(gym.Wrapper):
    def __init__(self, env: Env, shape):
        super().__init__(env)
        self.shape = shape

    def reset(self):
        state, _ = super().reset()
        return self._convert_frame(state)

    def step(self, action):
        state, reward, info = super().step(action)
        state = self._convert_frame(state)
        return state, reward, info

    def _convert_frame(self, state):
        state = np.array([cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), self.shape) for x in state])
        return state

class Recorder(gym.Wrapper):
    def __init__(self, env, saved_path):
        super().__init__(env)
        height, width, _ = self.observation_space.shape
        gcutils.ensure_folder(saved_path)
        self.video = cv2.VideoWriter(str(saved_path), 0, 24, (width, height))

    def record(self, image):
        self.video.write(image[..., ::-1])

    def step(self, action):
        state, *others = super().step(action)
        self.record(state)
        return state, *others

action_list = [['NOOP'], ['left'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B']]
def _create_base_env(world, stage):
    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', render_mode='rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, action_list)
    return env

def create_train_env(world, stage, skip=4):
    env = _create_base_env(world, stage)
    env = FrameStack(env, num_stack=skip)
    env = SkipFrame(env, skip=skip)
    env = CustomReward(env, world, stage)
    env = FrameConverter(env, NORM_IMAGE_SIZE)
    env = Replay(env)
    return env

def create_evaluate_env(world, stage, monitor_path, skip=4):
    env = _create_base_env(world, stage)
    env = Recorder(env, monitor_path)
    env = FrameStack(env, num_stack=skip)
    env = SkipFrame(env, skip=skip)
    env = CustomReward(env, world, stage)
    env = FrameConverter(env, NORM_IMAGE_SIZE)
    return env
