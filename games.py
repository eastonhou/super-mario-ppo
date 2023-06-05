import gym, gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

class MarioReward(gym.Wrapper):
    def __init__(self, env=None, world=None, stage=None):
        super().__init__(env)
        self.curr_score = 0
        self.current_x = 40
        self.status = 'small'
        self.world = world
        self.stage = stage

    def step(self, action):
        state, reward, done, trunk, info = self.env.step(action)
        reward = self._compute_reward(reward, done, info)
        return state, reward, done, trunk, info

    def reset(self):
        state, info = self.env.reset()
        self.curr_score = 0
        self.cur_time = 400
        self.current_x = 40
        self.status = 'small'
        return state, info

    def _compute_reward(self, reward, done, info):
        reward += info['score'] - self.curr_score
        if self.status == 'small' and info['status'] == 'big': reward += 2000
        elif self.status == 'big' and info['status'] == 'small': reward -= 2000
        if done:
            if info['flag_get']:
                reward += 5000
                info['state'] = 'done'
                del info['flag_get']
            else:
                reward -= 5000
                info['state'] = 'done'
        else:
            info['state'] = 'playing'
        reward += (info['x_pos'] - self.current_x)
        #reward -= (self.cur_time - info['time']) / 20
        self.cur_time = info['time']
        self.current_x = info['x_pos']
        self.curr_score = info['score']
        self.status = info['status']
        return reward

class BreakoutReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, done, trunk, info = self.env.step(action)
        if done: info['state'] = 'done'
        else: info['state'] = 'playing'
        return state, reward, done, trunk, info

def create_mario_profile(world, stage):
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    #action_list = [['NOOP'], ['A', 'B'], ['left', 'B'], ['left', 'A', 'B'], ['right', 'B'], ['right', 'A', 'B']]
    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', render_mode='rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioReward(env, world, stage)
    return env

def create_breakout():
    #action_space = [['NOOP'], ['LEFT'], ['RIGHT'], ['FIRE']]
    env = gym.make('ALE/Breakout-v5')
    env = BreakoutReward(env)
    #env = JoypadSpace(env, action_space)
    return env