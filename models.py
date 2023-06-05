import torch
import torch.nn as nn

class GameModel(nn.Module):
    def __init__(self, num_inputs, num_actions) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1152, 512), nn.ReLU(inplace=True))
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()
        self.rewards = -100000

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    def forward(self, input):
        hidden = self.layers(input.float().div(255))
        logits, critic = self.actor_linear(hidden), self.critic_linear(hidden)
        return logits, critic.view(-1)

    def load(self, path):
        ckpt = torch.load(path, map_location=lambda storage, location: storage)
        self.load_state_dict(ckpt['model'])
        self.rewards = ckpt['rewards']
        return ckpt

    def save(self, path, **kwargs):
        ckpt = {'model': self.state_dict()}
        ckpt.update(kwargs)
        torch.save(ckpt, path)
        self.rewards = kwargs['rewards']
