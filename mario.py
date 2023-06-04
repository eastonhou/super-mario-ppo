import torch, random, datetime, os, time, copy, gym, gym_super_mario_bros
import numpy as np
import matplotlib.pyplot as plt
from gym.core import Env
from torch import nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import deque
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from galois_common import gcutils
import utils

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb', apply_api_compatibility=True)
env = JoypadSpace(env, [['right'], ['right', 'A']])
env.reset()

next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

class SkipFrame(gym.Wrapper):
    def __init__(self, env: Env, monitor, skip):
        super().__init__(env)
        self._skip = skip
        self.monitor = monitor

    def step(self, action):
        total_reward = 0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if evaluation: self.monitor.record(obs)
            if done: break
        return obs, total_reward, done, trunk, info

class CustomReward(gym.Wrapper):
    def __init__(self, env=None, world=None, stage=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage

    def step(self, action):
        state, reward, done, trunk, info = self.env.step(action)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
                reward -= 50
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True

        self.current_x = info["x_pos"]
        return state, reward / 10., done, trunk, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return self.env.reset()

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = transforms.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: Env, shape):
        super().__init__(env)
        if isinstance(shape, int): self.shape = (shape, shape)
        else: self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        _transforms = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        observation = _transforms(observation).squeeze(0)
        return observation

class Mario:
    def __init__(self, state_dim, action_dim, save_dir) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = 'cuda'
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.model_path = os.path.join(save_dir, 'model.ckpt')
        self.load_checkpoint()
        self.net = self.net.to(device=self.device)
        self.exploration_rate = 0.976
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

    def load_checkpoint(self):
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=lambda storage, location: storage)
            self.net.load_state_dict(ckpt)

    def save_checkpoint(self):
        ckpt = self.net.state_dict()
        torch.save(ckpt, self.model_path)
    
    def act(self, state, eval=False):
        if not eval and np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x): return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        pass

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        c, h, w = input_dim
        assert h == 84 and w == 84
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, output_dim))
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters(): p.requires_grad = False

    def forward(self, input, model):
        if model == 'online': return self.online(input)
        elif model == 'target': return self.target(input)

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir) -> None:
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir) -> None:
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.ckpt"
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir) -> None:
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def learn(self):
        if self.curr_step % self.sync_every == 0: self.sync_Q_target()
        if self.curr_step % self.save_every == 0: self.save()
        if self.curr_step < self.burnin: return None, None
        if self.curr_step % self.learn_every != 0: return None, None
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

save_dir = Path('checkpoints/0')
#gcutils.rmdir(save_dir)
gcutils.mkdir(save_dir)

monitor = utils.Monitor(256, 240, save_dir / 'video.avi')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = utils.MetricLogger(save_dir)
env = SkipFrame(env, monitor, skip=4)
env = CustomReward(env)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
evaluation = False

def play():
    global evaluation
    evaluation = True
    state = env.reset()
    for step in range(200):
        action = mario.act(state, True)
        next_state, reward, done, trunc, info = env.step(action)
        state = next_state
        if done or info['flag_get']: break

episodes = 2000000
for e in range(episodes):
    evaluation = False
    state = env.reset()
    while True:
        action = mario.act(state)
        next_state, reward, done, trunc, info = env.step(action)
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        if done or info['flag_get']: break
    logger.log_episode()
    if e % 5 == 0: logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
    if e % 20 == 0:
        play()
        mario.save_checkpoint()
