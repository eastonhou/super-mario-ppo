import gym, cv2
import numpy as np
from gym.core import Env
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
        state, reward, _, _, info = super().step(action)
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

def create_train_env(env, skip=4):
    env = FrameStack(env, num_stack=skip)
    env = SkipFrame(env, skip=skip)
    env = FrameConverter(env, NORM_IMAGE_SIZE)
    env = Replay(env)
    return env

def create_evaluate_env(env, monitor_path, skip=4):
    env = Recorder(env, monitor_path)
    env = FrameStack(env, num_stack=skip)
    env = SkipFrame(env, skip=skip)
    env = FrameConverter(env, NORM_IMAGE_SIZE)
    return env
