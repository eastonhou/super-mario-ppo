import torch, os, tqdm, functools, collections
import numpy as np
import environment, models, utils, games
from torch import nn
from common import gcutils, options
from torch.utils.tensorboard import SummaryWriter
class PPO:
    def __init__(self, game_creator, game_arguments, skip) -> None:
        self.skip = skip
        self.game = game_creator(**game_arguments)
        self.folder = gcutils.join('checkpoints', self.game.spec.name)
        self.game_creator = game_creator
        self.game_arguments = game_arguments
        self.action_space = self.game.action_space.n
        gcutils.mkdir(self.folder)
        self.model = models.GameModel(self.skip, self.action_space)
        self.model.share_memory()
        self.criterion = Loss()
        self.logger = utils.MetricLogger(gcutils.join(self.folder, 'log.txt'))
        self.tb_writer = SummaryWriter(self.folder)
        self.renderer = environment.FrameRenderer()
        self.loss_dict = collections.defaultdict(lambda: 0)
        self._load()

    def train(self, device):
        self.model.to(device)
        self.criterion.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                state[k] = v.to(device)
        env = environment.MultiTrainEnv(self.game_creator, self.game_arguments, functools.partial(self._sample, device=device), self.skip)
        while True:
            self.epoch += 1
            self.logger.new_epoch()
            self._train_epoch(env, 10, 8, device)
            video_path = self._eval_epoch(device)
            self.logger.end_epoch(self.epoch)
            message = ' '.join(f'{key}={value:>.2F}' for key, value in self.loss_dict.items())
            score_msg = f'score={env.score:>.2F}'
            if self.logger.score > self.model.rewards:
                self._save()
                print(f'MODEL SAVED [{self.model.rewards:>.2F}] {message} {score_msg}')
                gcutils.copy_file(video_path, gcutils.join(os.path.dirname(video_path), f'{self.epoch}.avi'))
            else:
                #self._save('model.current.ckpt')
                print(f'MODEL CONTINUE [{self.logger.score:>.2F}/{self.model.rewards:>.2F}] {message} {score_msg}')
            self.tb_writer.add_scalar('training/score', env.score, self.epoch)
            for k, v in self.loss_dict.items():
                self.tb_writer.add_scalar(f'training/{k}', v, self.epoch)

    def _train_epoch(self, env, rounds, batch_size, device):
        self.model.train()
        tq = tqdm.tqdm(range(rounds))
        for _ in tq:
            #ref_model = self._create_reference_model(device)
            batch = [self._precompute_step(env, device) for _ in range(batch_size)]
            loss = self._offline(batch)
            tq.desc = f'[{loss:>.2F}]'
            self.logger.train_step(loss)

    def _precompute_step(self, env, device):
        samples = env.get()
        states = torch.tensor(np.stack([x['prev_state'] for x in samples] + [samples[-1]['state']], 0), device=device)
        scores = torch.tensor([x['score'] for x in samples], dtype=torch.float32, device=device)
        actions = [x['action'] for x in samples]
        rewards = torch.tensor([x['score'] - x['prev_score'] for x in samples], dtype=torch.float32, device=device)
        with torch.no_grad():
            ref_logits, logv = self.model(states)
            V, advantages, ref_log_probs = self.criterion.precomute(ref_logits, logv, actions, rewards, scores)
            return {
                'states': states[:-1],
                'actions': actions,
                'rewards': rewards,
                'V': V,
                'advantages': advantages,
                'ref_log_probs': ref_log_probs
            }

    def _offline(self, batch):
        total_loss = 0
        for _ in range(10):
            loss = 0
            for pack in batch:
                logits, values = self.model(pack['states'])
                loss_dict = self.criterion(pack, logits, values)
                loss += loss_dict['loss']
                for k, v in loss_dict.items():
                    self.loss_dict[k] = self.loss_dict[k] * 0.95 + v * 0.05
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _create_reference_model(self, device):
        model = models.GameModel(self.skip, self.action_space).to(device)
        model.load_state_dict(self.model.state_dict())
        model.eval()
        return model

    @torch.inference_mode()
    def _eval_epoch(self, device):
        self.model.eval()
        video_path = gcutils.join(self.folder, 'video', 'playing.avi')
        env = environment.create_evaluate_env(self.game, video_path, self.skip, self.renderer)
        state = env.reset()
        for _ in range(1000):
            action = self._sample(state, device)
            state, score, info = env.step(action)
            self.logger.eval_step(score)
            if self._game_finished(info): break
        return video_path

    def _game_finished(self, info):
        return info['state'] == 'done'

    @torch.inference_mode()
    def _sample(self, state, device=None):
        if self.model.training and np.random.ranf() < 0.1:
            action = np.random.choice(self.action_space)
        state = torch.tensor(state[None, ...], device=device)
        logits, value = self.model(state)
        logits = logits[0].softmax(-1)
        action = torch.multinomial(logits, 1, True)
        #action = logits.argmax(-1)
        action = action.item()
        self.renderer.update(logits.cpu().numpy(), value.item())
        return action

    def _load(self):
        self.epoch = 0
        path = gcutils.join(self.folder, 'model.ckpt')
        if not os.path.isfile(path):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            return
        ckpt = self.model.load(path)
        self.epoch = ckpt['epoch']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def _save(self, name='model.ckpt'):
        path = gcutils.join(self.folder, name)
        self.model.save(path, epoch=self.epoch, rewards=self.logger.score, optimizer=self.optimizer.state_dict())

class Loss(nn.Module):
    def __init__(self, gamma=1, lmbda=0.975, epsilon=0.2, beta=0.02) -> None:
        super().__init__()
        self.gamma = gamma
        self.lmbda = lmbda
        self.beta = beta
        self.epsilon = epsilon
        coef = torch.pow(gamma * lmbda, torch.cumsum(torch.triu(torch.ones(512, 512), 1), -1))
        self.register_buffer('coef', torch.triu(coef, 0))

    def forward(self, pack, logits, values):
        n = pack['ref_log_probs'].shape[0]
        new_log_probs = logits.log_softmax(-1)
        ratio = (new_log_probs[torch.arange(n), pack['actions']] - pack['ref_log_probs']).exp()
        lb = torch.full_like(ratio, fill_value=0)
        ub = torch.full_like(ratio, fill_value=10000)
        lb[pack['advantages'] < 0] = 1 - self.epsilon
        ub[pack['advantages'] > 0] = 1 + self.epsilon
        actor_loss = -torch.min(
            ratio.mul(pack['advantages']),
            torch.clamp(ratio, lb, ub).mul(pack['advantages'])).mean()
        #critic_loss = (R.sub(values[:n]) ** 2).mean()
        #critic_loss = nn.functional.l1_loss(values, pack['V'], reduction='mean')
        critic_loss = nn.functional.l1_loss(values, pack['V'], reduction='mean')
        entropy_loss = -new_log_probs.exp().mul(new_log_probs).sum(-1).mean()
        loss = actor_loss + critic_loss - self.beta * entropy_loss
        return {
            'loss': loss,
            'actor': actor_loss.item(),
            'critic': critic_loss.item(),
            'entropy': entropy_loss.item(),
            'steps': n
        }

    def precomute(self, ref_logits, ref_values, actions, rewards, scores):
        n = ref_values.shape[0] - 1
        #delta = rewards + self.gamma * ref_values[1:] - ref_values[:-1]
        delta = rewards
        advantages = delta[None, :].mul(self.coef[:n, :n]).sum(-1)
        ref_log_prob = ref_logits.log_softmax(-1)[torch.arange(n), actions]
        V = scores.float()
        return V, advantages, ref_log_prob
    '''
    def precomute(self, ref_logits, ref_values, actions, rewards, scores):
        n = ref_values.shape[0] - 1
        delta = rewards + self.gamma * ref_values[1:] - ref_values[:-1]
        R = delta[None, :].mul(self.coef[:n, :n]).sum(-1)
        advantages = R - ref_values[:-1]
        ref_log_prob = ref_logits.log_softmax(-1)[torch.arange(n), actions]
        return R, advantages, ref_log_prob
    '''

if __name__ == '__main__':
    opts = options.make_options(device='cuda')
    #ppo = PPO(games.create_mario_profile, dict(world=1, stage=1), 4)
    ppo = PPO(games.create_breakout, {}, 2)
    #ppo = PPO(games.create_flappy_bird, {}, 4)
    ppo.train(opts.device)
