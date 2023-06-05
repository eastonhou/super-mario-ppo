import torch
import torch.nn as nn
import torch.nn.functional as F

class GameModel(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = x.float().div(255)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x).view(-1)

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
