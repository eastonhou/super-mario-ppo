import torch
import numpy as np
import models, trainer_base
from torch import nn

class Trainer(trainer_base.Reinforcement):
    def __init__(self, game_creator, game_arguments, stack, skip) -> None:
        super().__init__('ppo', game_creator, game_arguments, stack, skip)
        self.criterion = Loss()

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

    def train_step(self, env, batch_size, device):
        batch = [self._precompute_step(env, device) for _ in range(batch_size)]
        loss = self._offline(batch)
        return loss

    def _change_lr(self):
        super()._change_lr()
        self.criterion.c2 *= 0.99

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

class Loss(nn.Module):
    def __init__(self, gamma=1, lmbda=0.975, epsilon=0.2, c2=0.1) -> None:
        super().__init__()
        self.gamma = gamma
        self.lmbda = lmbda
        self.c2 = c2
        self.epsilon = epsilon
        coef = torch.pow(gamma * lmbda, torch.cumsum(torch.triu(torch.ones(512, 512), 1), -1))
        self.register_buffer('coef', torch.triu(coef, 0))

    def forward(self, pack, new_log_probs, values):
        # 一共有n步
        n = pack['ref_log_probs'].shape[0]
        # 计算每个操作的对数概率
        # 计算当前模型相对参考模型对于每个动作的概率比
        ratio = (new_log_probs[torch.arange(n), pack['actions']] - pack['ref_log_probs']).exp()
        # 剪切概率比，当激励为正时，概率比不大于$1+\epsilon$，当激励函数为负时，概率比不小于$1-\epsilon$
        lb = torch.full_like(ratio, fill_value=0)
        ub = torch.full_like(ratio, fill_value=10000)
        lb[pack['advantages'] < 0] = 1 - self.epsilon
        ub[pack['advantages'] > 0] = 1 + self.epsilon
        # 计算策略收益
        actor_loss = -torch.min(
            ratio.mul(pack['advantages']),
            torch.clamp(ratio, lb, ub).mul(pack['advantages'])).mean()
        # 计算价函数函数误差
        critic_loss = nn.functional.l1_loss(values, pack['V'], reduction='mean')
        # 计算策略概率自身的熵
        entropy_loss = -new_log_probs.exp().mul(new_log_probs).sum(-1).mean()
        loss = actor_loss + critic_loss - self.c2 * entropy_loss
        return {
            'loss': loss,
            'actor': actor_loss.item(),
            'critic': critic_loss.item(),
            'entropy': entropy_loss.item(),
            'steps': n
        }

    def precomute(self, ref_logits, ref_values, actions, rewards, scores):
        n = ref_values.shape[0] - 1
        # 计算即时奖励$\delta_t$
        delta = rewards + self.gamma * ref_values[1:] - ref_values[:-1]
        # 每个即时奖励乘上时间系数
        advantages = delta[None, :].mul(self.coef[:n, :n]).sum(-1)
        ref_log_prob = ref_logits[torch.arange(n), actions]
        # 获得$V_t$的目标值，这里以局面分数作为局面评估函数的目标
        V = scores.float()
        return V, advantages, ref_log_prob
