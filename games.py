import gym, gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

class MarioReward(gym.Wrapper):
    def __init__(self, env=None, world=None, stage=None):
        super().__init__(env)
        self.world = world
        self.stage = stage

    def step(self, action):
        state, reward, done, trunk, info = self.env.step(action)
        score = self._compute(reward, done, info)
        return state, score, done, trunk, info

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def _compute(self, reward, done, info):
        score = info['score'] + info['coins'] * 50
        score += info['x_pos'] / 10
        if info['status'] == 'big': score += 200
        if done:
            if info['flag_get']: score += 1000
            else: score -= 200
            info['state'] = 'done'
        else:
            info['state'] = 'playing'
        return score / 200

class BreakoutReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def step(self, action):
        state, reward, done, trunk, info = self.env.step(action)
        if done: info['state'] = 'done'
        else: info['state'] = 'playing'
        self.score += 1 if reward > 0 else 0
        score = self.score * 100 + info['lives'] * 50 + 1
        self.ttl += 1
        return state, score / 100, done, trunk, info

    def reset(self):
        state, info = super().reset()
        self.score = 0
        self.ttl = 0
        return state, info
    
def create_mario_profile(world, stage):
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    #action_list = [['NOOP'], ['A', 'B'], ['left', 'B'], ['left', 'A', 'B'], ['right', 'B'], ['right', 'A', 'B']]
    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', render_mode='rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = MarioReward(env, world, stage)
    return env

def create_breakout():
    #action_space = [['NOOP'], ['LEFT'], ['RIGHT'], ['FIRE']]
    env = gym.make('ALE/Breakout-v5')
    env = BreakoutReward(env)
    #env = JoypadSpace(env, action_space)
    return env
