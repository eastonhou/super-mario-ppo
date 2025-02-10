import os, torch, functools, collections, tqdm
import environment, models, utils, games
import numpy as np
from common import gcutils
from torch.utils.tensorboard import SummaryWriter

class Reinforcement:
    def __init__(self, name, game_creator, game_arguments, stack, skip):
        self.skip = skip
        self.stack = stack
        self.game = game_creator(**game_arguments)
        self.folder = gcutils.join(f'checkpoints/{name}', self.game.spec.name)
        self.game_creator = game_creator
        self.game_arguments = game_arguments
        self.action_space = self.game.action_space.n
        gcutils.mkdir(self.folder)
        self.model = models.GameModel(stack, self.action_space)
        self.model.share_memory()
        #self.criterion = Loss()
        self.logger = utils.MetricLogger(gcutils.join(self.folder, 'log.txt'))
        self.tb_writer = SummaryWriter(self.folder)
        self.renderer = environment.FrameRenderer()
        self.loss_dict = collections.defaultdict(lambda: 0)
        self.lr = 2.5e-4
        self._load()

    def train(self, device):
        self.model.to(device)
        self.criterion.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                state[k] = v.to(device)
        env = environment.MultiTrainEnv(self.game_creator, self.game_arguments, functools.partial(self._sample, device=device), self.stack, self.skip)
        while True:
            self.epoch += 1
            self.logger.new_epoch()
            self._train_epoch(env, 10, 8, device)
            video_path = self._eval_epoch(device)
            self.logger.end_epoch(self.epoch)
            kvs = {
                'lr': self.lr,
                **self.loss_dict,
                'score': env.score
            }
            message = ' '.join(f'{key}={_format_value(value)}' for key, value in kvs.items())
            if self.logger.score > self.model.rewards:
                self._save()
                print(f'[{self.epoch}] SAVED [{self.model.rewards:>.2F}] {message}')
                gcutils.copy_file(video_path, gcutils.join(os.path.dirname(video_path), f'{self.epoch}.avi'))
            else:
                print(f'[{self.epoch}] [{self.logger.score:>.2F}/{self.model.rewards:>.2F}] {message}')
            self.tb_writer.add_scalar('training/score', env.score, self.epoch)
            for k, v in self.loss_dict.items():
                self.tb_writer.add_scalar(f'training/{k}', v, self.epoch)
            self._change_lr()

    def _train_epoch(self, env, rounds, batch_size, device):
        self.model.train()
        tq = tqdm.tqdm(range(rounds))
        for _ in tq:
            loss = self.train_step(env, batch_size, device)
            tq.desc = f'[{loss:>.2F}]'
            self.logger.train_step(loss)

    def train_step(self, env, batch_size, device):
        raise NotImplementedError()

    def _change_lr(self):
        self.lr = max(self.lr * 0.99, 2.5e-6)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def _load(self):
        self.epoch = 0
        path = gcutils.join(self.folder, 'model.ckpt')
        if os.path.isfile(path):
            ckpt = self.model.load(path)
            self.epoch = ckpt['epoch']
        else:
            ckpt = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if ckpt is not None: self.optimizer.load_state_dict(ckpt['optimizer'])

    def _save(self, name='model.ckpt'):
        path = gcutils.join(self.folder, name)
        self.model.save(path, epoch=self.epoch, rewards=self.logger.score, optimizer=self.optimizer.state_dict())

    @torch.inference_mode()
    def _eval_epoch(self, device):
        self.model.eval()
        video_path = gcutils.join(self.folder, 'video', 'playing.avi')
        env = environment.create_evaluate_env(self.game, video_path, self.stack, self.skip, self.renderer)
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
    def _sample(self, state, device=None, explore=0):
        if self.model.training and np.random.ranf() < 0.1:
            action = np.random.choice(self.action_space)
        state = torch.tensor(state[None, ...], device=device)
        logits, value = self.model(state)
        logits = logits[0].exp().clamp(explore / logits.numel(), None)
        logits.div_(logits.sum())
        action = torch.multinomial(logits, 1, True)
        #action = logits.argmax(-1)
        action = action.item()
        self.renderer.update(logits.cpu().numpy(), value.item())
        return action

def _format_value(value):
    if isinstance(value, str): return value
    elif torch.is_tensor(value): return f'{value.item():>.2F}'
    elif value >= 0.01: return f'{value:>.2F}'
    else: return f'{value:>.4g}'
