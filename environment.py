import gym, cv2, ray, os, time
import numpy as np
from gym.core import Env
from gym.wrappers import FrameStack
from common import gcutils

NORM_IMAGE_SIZE = (84, 84)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        for _ in range(self._skip):
            obs, score, done, trunk, info = self.env.step(action)
            if done: break
        return obs, score, done, trunk, info

class Replay(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.memory = []

    def collect(self, state, action):
        next_state, score, info = self.step(action)
        self.memory.append({
            'prev_state': state,
            'prev_score': self.prev_score,
            'action': action,
            'score': score,
            'state': next_state,
            'count': len(self.memory) + 1,
            'info': info
        })
        self.prev_score = score
        return next_state, info

    def reset(self):
        self.memory = []
        self.prev_score = 0
        return super().reset()

class FrameConverter(gym.Wrapper):
    def __init__(self, env: Env, shape):
        super().__init__(env)
        self.shape = shape

    def reset(self):
        state, _ = super().reset()
        return self._convert_frame(state)

    def step(self, action):
        state, score, _, _, info = super().step(action)
        state = self._convert_frame(state)
        return state, score, info

    def _convert_frame(self, state):
        state = np.array([cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), self.shape) for x in state])
        return state

class Recorder(gym.Wrapper):
    def __init__(self, env, saved_path, render_callback=None):
        super().__init__(env)
        height, width, _ = self.observation_space.shape
        gcutils.ensure_folder(saved_path)
        self.video = cv2.VideoWriter(str(saved_path), 0, 24, (width, height))
        self.render_callback = render_callback

    def record(self, image, score):
        image = image[..., ::-1].copy()
        if self.render_callback is not None:
            self.render_callback(image, score)
        self.video.write(image)

    def step(self, action):
        state, score, *others = super().step(action)
        self.record(state, score)
        return state, score, *others

class FrameRenderer:
    def __init__(self) -> None:
        self.action_prob = None
        self.value = 0

    def update(self, action_prob, value):
        self.action_prob = action_prob
        self.value = value * 0.1 + self.value * 0.9

    def __call__(self, image, score):
        if self.action_prob is None or self.value is None: return
        x0, y0 = 5, 5
        self._render_action_probs(image, x0, y0)
        self._render_value(image, x0, y0 + 40, score)

    def _render_action_probs(self, image, x0, y0):
        slot_width = 16
        slot_height = 30
        pts = []
        for k, prob in enumerate(self.action_prob):
            x1, y1 = x0 + k * slot_width, y0
            x2, y2 = x1 + slot_width, y1 + slot_height
            pts.append([x1, y1, x2, y2])
            cv2.rectangle(image, (x1, y1 + int(slot_height * (1 - prob))), (x2, y2), (0, 128, 255), cv2.FILLED)
        for x1, y1, x2, y2 in pts:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    def _render_value(self, image, x0, y0, score):
        cv2.putText(image, f'value: {self.value:>.2F}/{score:>.2F}', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(255, 0, 0))

def create_train_env(env, stack, skip):
    env = FrameStack(env, num_stack=stack)
    env = SkipFrame(env, skip=skip)
    env = FrameConverter(env, NORM_IMAGE_SIZE)
    env = Replay(env)
    return env

def create_evaluate_env(env, monitor_path, stack, skip, render_callback=None):
    env = Recorder(env, monitor_path, render_callback)
    env = FrameStack(env, num_stack=stack)
    env = SkipFrame(env, skip=skip)
    env = FrameConverter(env, NORM_IMAGE_SIZE)
    return env

@ray.remote
class TrainEnv:
    def __init__(self, id, creator, kwargs, stack, skip) -> None:
        game = creator(**kwargs)
        self.env = create_train_env(game, stack, skip)
        self.id = id

    def step(self, state, action):
        return self.id, self.env.collect(state, action)

    def reset(self):
        return self.id, (self.env.reset(), {'state': 'reset'})

    def collect(self):
        return self.env.memory

class MultiTrainEnv:
    def __init__(self, game_creator, game_arguments, sampler, stack, skip, parallelism=None, max_size=100) -> None:
        if parallelism is None: parallelism = os.cpu_count() // 4
        self.envs = [TrainEnv.remote(id, game_creator, game_arguments, stack, skip) for id in range(parallelism)]
        self.sampler = sampler
        self.queue = []
        self.max_size = max_size
        self.score = 0
        self.thead = run_parallel(self.worker)

    def worker(self):
        gcutils.event_loop_init()
        tasks = [x.reset.remote() for x in self.envs]
        while True:
            ready, _ = gcutils.when_any(tasks)
            for id, (state, info) in ready:
                if info['state'] == 'done':
                    samples = ray.get(self.envs[id].collect.remote())
                    samples = self._random_size(samples)
                    self.put(samples)
                    tasks[id] = self.envs[id].reset.remote()
                    self.score = self.score * 0.99 + samples[-1]['score'] * 0.01
                else:
                    action = self.sampler(state)
                    tasks[id] = self.envs[id].step.remote(state, action)

    def get(self):
        while not self.queue: time.sleep(0.5)
        samples = self.queue.pop(0)
        return samples

    def put(self, item):
        while len(self.queue) == self.max_size: time.sleep(0.5)
        self.queue.append(item)

    def _random_size(self, samples, min_size=200, max_size=512):
        start = np.random.randint(0, max(len(samples) - min_size, 0) + 1)
        end = np.random.randint(min(start + min_size, len(samples)), len(samples) + 1)
        end = min(end, start + max_size)
        return samples[start:end]

def run_parallel(target, *args):
    import threading
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread
