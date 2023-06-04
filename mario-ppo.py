import torch, os, tqdm
import numpy as np
import environment, models, utils
from torch import nn
from galois_common import gcutils

class PPO:
    def __init__(self, world, stage) -> None:
        self.skip = 4
        self.world, self.stage = world, stage
        self.folder = gcutils.join('checkpoints', f'general')
        gcutils.mkdir(self.folder)
        self.train_env = environment.create_train_env(world, stage, self.skip)
        self.model = models.MarioNet(self.skip, self.train_env.action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = Loss()
        self.logger = utils.MetricLogger(gcutils.join(self.folder, 'log.txt'))
        self._load()

    def train(self, device):
        self.model.to(device)
        self.criterion.to(device)
        while True:
            self.logger.new_epoch()
            self._train_epoch(100, device)
            self._eval_epoch(device)
            self._save()
            self.logger.end_epoch(self.epoch)
            self.epoch += 1

    def _train_epoch(self, rounds, device):
        self.model.train()
        tq = tqdm.tqdm(range(rounds))
        for _ in tq:
            ref_model = self._create_reference_model(device)
            samples = self.sample_game(self.train_env, device)
            states = torch.tensor(np.stack([x['state'] for x in samples], 0), device=device)
            actions = [x['action'] for x in samples]
            rewards = torch.tensor([x['reward'] for x in samples], device=device)
            with torch.no_grad():
                ref_logits, ref_values = ref_model(states)
                R, advantages, ref_log_probs = self.criterion.precomute(ref_logits, ref_values, actions, rewards)
            loss = self._offline(states, R, advantages, ref_log_probs, actions, rewards)
            tq.desc = f'[{loss:>.2F}]'
            self.logger.train_step(loss)

    def _offline(self, states, R, advantages, ref_log_probs, actions, rewards):
        total_loss = 0
        for _ in range(4):
            logits, values = self.model(states)
            loss = self.criterion(R, advantages, ref_log_probs, logits, values, actions, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def sample_game(self, env, device, min_length=10, max_length=200):
        state = env.reset()
        while True:
            action = self._sample(env, state, device)
            state, info = env.collect(state, action)
            if self._game_finished(info): break
        start = np.random.randint(0, max(len(env.memory) - min_length, 0) + 1)
        return env.memory[start:start+max_length]

    def _create_reference_model(self, device):
        model = models.MarioNet(self.skip, self.train_env.action_space.n).to(device)
        model.load_state_dict(self.model.state_dict())
        model.eval()
        return model

    @torch.inference_mode()
    def _eval_epoch(self, device):
        self.model.eval()
        video_path = gcutils.join(self.folder, 'video', f'{self.epoch}.avi')
        env = environment.create_evaluate_env(self.world, self.stage, video_path, self.skip)
        state = env.reset()
        while True:
            action = self._sample(env, state, device)
            state, reward, info = env.step(action)
            self.logger.eval_step(reward)
            if self._game_finished(info): break

    def _game_finished(self, info):
        return info['state'] in ['success', 'fail']

    def _sample(self, env, state, device=None):
        if self.model.training and np.random.ranf() < max(0.97 ** self.epoch, 0.1):
            action = np.random.choice(env.action_space.n)
        else:
            state = torch.tensor(state[None, ...], device=device)
            logits, _ = self.model(state)
            logits = logits[0].softmax(-1)
            action = torch.multinomial(logits, 1, True)
            action = action.item()
        return action

    def _load(self):
        self.epoch = 0
        path = gcutils.join(self.folder, 'model.ckpt')
        if not os.path.isfile(path): return
        ckpt = self.model.load(path)
        self.epoch = ckpt['epoch'] + 1

    def _save(self):
        path = gcutils.join(self.folder, 'model.ckpt')
        self.model.save(path, epoch=self.epoch)

class Loss(nn.Module):
    def __init__(self, gamma=1, lmbda=0.95, epsilon=0.2, beta=0.01) -> None:
        super().__init__()
        self.gamma = gamma
        self.lmbda = lmbda
        self.beta = beta
        self.epsilon = epsilon
        coef = torch.pow(gamma * lmbda, torch.cumsum(torch.triu(torch.ones(512, 512), 1), -1))
        self.register_buffer('coef', torch.triu(coef, 0))

    def forward(self, R, advantages, ref_log_probs, logits, values, actions, rewards):
        n = ref_log_probs.shape[0]
        new_log_probs = logits.log_softmax(-1)
        ratio = (new_log_probs[torch.arange(n), actions[:n]] - ref_log_probs).exp()
        actor_loss = -torch.min(
            ratio.mul(advantages),
            torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon).mul(advantages)).mean()
        #critic_loss = (R.sub(values[:n]) ** 2).mean()
        critic_loss = nn.functional.smooth_l1_loss(R, values[:n], reduction='mean')
        entropy_loss = -new_log_probs.exp().mul(new_log_probs).sum(-1).mean()
        loss = actor_loss + critic_loss - self.beta * entropy_loss
        return loss

    def precomute(self, ref_logits, ref_values, actions, rewards):
        n = ref_values.shape[0] - 1
        delta = rewards[:-1] + self.gamma * ref_values[1:] - ref_values[:-1]
        R = delta[None, :].mul(self.coef[:n, :n]).sum(-1)
        advantages = R - ref_values[:-1]
        ref_log_prob = ref_logits.log_softmax(-1)[torch.arange(n), actions[:n]]
        return R, advantages, ref_log_prob

if __name__ == '__main__':
    ppo = PPO(1, 1)
    ppo.train('cuda')