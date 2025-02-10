import ppo, grpo, games
from common import options

if __name__ == '__main__':
    opts = options.make_options(rl='grpo', device='cuda')
    creators = {
        'ppo': ppo.Trainer,
        'grpo': grpo.Trainer
    }
    rl = creators[opts.rl](games.create_mario_profile, dict(world=1, stage=1), 8, 4)
    #ppo = PPO(games.create_breakout, {}, 8, 1)
    #ppo = PPO(games.create_flappy_bird, {}, 4)
    rl.train(opts.device)
