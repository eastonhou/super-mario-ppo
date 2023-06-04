import torch
import torch.nn as nn

class MarioNet(nn.Module):
    def __init__(self, num_inputs, num_actions) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3200, 512), nn.LayerNorm(512))
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

    def forward(self, input):
        hidden = self.layers(input.float())
        logits, critic = self.actor_linear(hidden), self.critic_linear(hidden)
        return logits, critic.view(-1)

    def load(self, path):
        ckpt = torch.load(path, map_location=lambda storage, location: storage)
        self.load_state_dict(ckpt['model'])
        return ckpt

    def save(self, path, **kwargs):
        ckpt = {'model': self.state_dict()}
        ckpt.update(kwargs)
        torch.save(ckpt, path)
