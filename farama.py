import gym, torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

device = 'cuda:0'

num_cells = 256
lr = 3e-4
max_grad_norm = 1.0
frame_skip = 1
frames_per_batch = 1000 // frame_skip
total_frames = 30000 // frame_skip

sub_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

base_env = GymEnv('InvertedDoublePendulum-v4', device=device, frame_skip=frame_skip)
env = TransformedEnv(base_env, Compose(
    ObservationNorm(in_keys=['observation']),
    DoubleToFloat(in_keys=['observation']),
    StepCounter()))
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

check_env_specs(env)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device), NormalParamExtractor())

policy_module = TensorDictModule(actor_net, in_keys=['observation'], out_keys=['loc', 'scale'])

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=['loc', 'scale'],
    distribution_class=TanhNormal,
    distribution_kwargs={'min': env.action_spec.space.minimum, 'max': env.action_spec.space.maximum},
    return_log_prob=True)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(num_cells, device=device), nn.Tanh(),
    nn.LazyLinear(1, device=device))

value_module=ValueOperator(module=value_net, in_keys=['observation'])

print('running poplicy:', policy_module(env.reset()))
print('running poplicy:', value_module(env.reset()))

collector = SyncDataCollector(env, policy_module,
                              frames_per_batch=frames_per_batch,
                              total_frames=total_frames,
                              split_trajs=False,
                              device=device)

replay_buffer = ReplayBuffer(storage=LazyTensorStorage(frames_per_batch), sampler=SamplerWithoutReplacement())

advantage_module = GAE(gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True)

loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    advantage_key='advantage',
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=gamma,
    loss_critic_type='smooth_l1')

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)

logs = defaultdict(list)
pbar = tqdm(total=total_frames * frame_skip)
evar_str = ''
from torchrl.record import VideoRecorder
rec = VideoRecorder()
rec()
for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):
        with torch.inference_mode():
            advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = loss_vals['loss_objective'] + loss_vals['loss_critic'] + loss_vals['loss_entropy']
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()
    logs['reward'].append(tensordict_data['next', 'reward'].mean().item())
    pbar.update(tensordict_data.numel() * frame_skip)
    cum_reward_str = f"average reward={logs['reward'][-1]:4.4f} (init={logs['reward'][0]:4.4f})"
    logs['step_count'].append(tensordict_data['step_count'].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs['lr'].append(optim.param_groups[0]['lr'])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    with set_exploration_type(ExplorationType.MEAN), torch.inference_mode():
        eval_rollout = env.rollout(1000, policy_module)
        logs['eval reward (mean)'].append(eval_rollout['next', 'reward'].mean().item())
        logs['eval reward (sum)'].append(eval_rollout['next', 'reward'].sum().item())
        logs['eval step_count'].append(eval_rollout['step_count'].max().item())
        eval_str = (
            f"eval cumculative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][-1]: 4.4f}) "
            f"eval step_count: {logs['eval step_count'][-1]}")
        del eval_rollout
    pbar.set_description(', '.join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    scheduler.step()

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()
