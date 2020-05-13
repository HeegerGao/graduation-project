'''
    GAIL
    5/10：  发现ppo和ddpg的actor还不能通用，一个是分布采样，一个是确定性的输出
            先用现有的ppo测试一下gail效果，然后再用ddpg测试gail效果。
            然后再用ppo作为强化学习测试效果
            update 记录alogp和其他
    5/11:   发现D的输出很快收敛，导致G没有reward，一直为0
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import Env, TrajectoryLoader
import numpy as np
import time
import os
import sys
from utils import Env
from mpi4py import MPI
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n-epochs', type=int, default=1000, help='the number of epochs to train the agent')
    parser.add_argument('--d-updates', type=int, default=5, help='the times to update d')
    parser.add_argument('--ppo-updates', type=int, default=5, help='the times to update ppo')

    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_param', type=float, default=0.1)
    
    parser.add_argument('--logdir', type=str, default='./gaillog')
    parser.add_argument('--save-dir', type=str, default='ddpg_models/', help='the path to save the models')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args

def compute_reward(cube_pos, goal_pos):
    length = cube_pos.shape[0]
    r = np.zeros(length,)
    for i in range(length):
        for num in range(4):
            dist = np.linalg.norm(cube_pos[(num*3):((num+1)*3)] - goal_pos[(num*3):((num+1)*3)])
            if dist < 0.04:
                r[i] += 1
    
    return r.reshape((length, 1))

def get_env_params(): 
    params = {'obs': 11,
    'goal': 12,
    'action': 8,
    'action_min': -1,
    'action_max': 1,
    'max_timesteps': 300,
    'reward_func': compute_reward
    }

    return params

def sample_demonstrations(demo_loader, sample_num):
    demos_no_reward = demo_loader.sample_trajectories_no_shuffle(sample_num)
    assert demos_no_reward.shape[1] == 43
    demo_transitions = {}
    demo_transitions['obs'] = demos_no_reward[:-1, 0:11]
    demo_transitions['ag'] = demos_no_reward[:-1, 11:23]
    demo_transitions['g'] = demos_no_reward[:-1, 23:35]
    demo_transitions['actions'] = demos_no_reward[:-1, 35:43]
    demo_transitions['obs_next'] = demos_no_reward[1:, 0:11]
    demo_transitions['ag_next'] = demos_no_reward[1:, 11:23]
    demo_transitions['r'] = compute_reward(demo_transitions['ag_next'], demo_transitions['g'])

    return demo_transitions

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.apply(self._weights_init)

    def forward(self, state, action):
        prob = self.net(torch.cat([state, action], 1))
        return prob

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)


class NetForPPO(nn.Module):
    """
    Actor-Critic Network for PPO, using Beta Distribution
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.pre_process = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.v = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))   #输出critic
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(128, action_dim), nn.Softplus())  #输出0~∞
        self.beta_head = nn.Sequential(nn.Linear(128, action_dim), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        pre = self.pre_process(state)
        
        v = self.v(pre)
        after = self.fc(pre)
        alpha = self.alpha_head(after) + 1  #1~∞，适合作为beta分布的参数
        beta = self.beta_head(after) + 1

        return (alpha, beta), v


class PPOPolicy():
    """
    Agent for training
    """
    def __init__(self, state_dim, action_dim, clip_param, gamma, lr, device):
        self.net = NetForPPO(state_dim, action_dim).float().to(device)
        self.clip_param = clip_param    # epsilon in clipped loss
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.99))
        self.gamma = gamma
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.detach().cpu().numpy()
        a_logp = a_logp.item()

        return action, a_logp

    def update(self, state, action, reward, a_logp, state_):
        s = torch.tensor(state, device=self.device, dtype=torch.float32)
        a = torch.tensor(action, device=self.device, dtype=torch.float32)
        r = torch.tensor(reward, device=self.device, dtype=torch.float32)
        s_ = torch.tensor(state_, device=self.device, dtype=torch.float32)
        old_a_logp = torch.tensor(a_logp, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            target_v = r + self.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)


        alpha, beta = self.net(s)[0]
        dist = Beta(alpha, beta)
        a_logp = dist.log_prob(a).sum(dim=1, keepdim=True)
        ratio = torch.exp(a_logp - old_a_logp)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.smooth_l1_loss(self.net(s)[1], target_v)
        loss = action_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.mean().item()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    #env
    args=get_args()
    env_params = get_env_params()
    writer = SummaryWriter(args.logdir)
    train_env = Env(scene_file='single.ttt', render=False)

    # seed
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # demos
    loader = TrajectoryLoader('single_panda_cube_expert_trajectorys.csv')
    expert_traj = loader.expert_trajectories[0:276]
    expert_traj = np.delete(expert_traj, 0, axis=1)
    expert_traj = np.delete(expert_traj, 0, axis=1)

    # define D
    discriminator = Discriminator(env_params['obs'] + 2 * env_params['goal'], env_params['action']).to(args.device)
    d_op = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_criterion = nn.BCELoss()  # 二分类交叉熵(Binary Classification Entropy)，正好适合GAN网络的loss：logD+log(1-D)

    # define G
    policy = PPOPolicy(env_params['obs'] + 2 * env_params['goal'], env_params['action'], args.clip_param, args.gamma, args.lr, device=args.device)

    # logging variables
    running_reward = 0
    timestep = 0

    # trainning loop
    discriminator.train()
    policy.net.train()

    d_loss = torch.Tensor([0.])
    g_loss = 0
    expert_D = torch.Tensor([0,0,0])
    agent_D = torch.Tensor([0,0,0])

    start_time = time.time()

    #用来记录一次batch的
    state = []
    state_ = []
    action = []
    action_log_p = []
    reward = []

    for i_epoch in range(1, args.n_epochs + 1):
        s = train_env.reset_from_demo(expert_traj[0, 0:35])
        running_reward = 0
        for t in range(env_params['max_timesteps']):
            print(t)
            timestep += 1

            state.append(s[0])
            
            a, a_logp = policy.select_action(s)
            s_, r, done = train_env.step(a[0])
            
            action.append(a[0])
            action_log_p.append(a_logp)
            state_.append(s_[0])

            # update if its time
            if timestep % args.batch_size == 0: #PPO是on policy的
                print('updating')
                #expert trajectories
                demo_tras = sample_demonstrations(loader, args.batch_size)
                expert_states = np.concatenate((demo_tras['obs'], demo_tras['ag'], demo_tras['g']), axis=1)
                print(expert_states.shape)
                expert_actions = demo_tras['actions']
                print(expert_actions.shape)

                #agent trajectories
                state = np.array(state)
                action = np.array(action)
                action_log_p = np.array(action_log_p).reshape(state.shape[0], 1)
                state_ = np.array(state_)

                # update discriminator
                for _ in range(args.d_updates):
                    # compute D(s,a) from expert trajectories
                    expert_D = discriminator(torch.from_numpy(expert_states).to(torch.float32).to(args.device), torch.from_numpy(expert_actions).to(torch.float32).to(args.device))
                    agent_D = discriminator(torch.from_numpy(state).to(torch.float32).to(args.device).detach(), torch.from_numpy(action).to(torch.float32).to(args.device).detach())

                    d_op.zero_grad()
                    # 这次变一下，让D为0的时候表示专家
                    d_loss = d_criterion(expert_D, torch.zeros((expert_D.shape[0], 1), device=args.device)) + d_criterion(
                        agent_D, torch.ones((agent_D.shape[0], 1), device=args.device))
                    d_loss.backward()
                    d_op.step()
                    torch.cuda.empty_cache()

                # 由于更新了D，重新计算reward
                agent_D = discriminator(torch.from_numpy(state).to(torch.float32).to(args.device).detach(),
                                        torch.from_numpy(action).to(torch.float32).to(args.device).detach())

                # GAIL
                gail_reward = -torch.log(agent_D)   # 由于0是专家，所以越小越好，所以负的越大越好

                reward = gail_reward

                for _ in range(args.ppo_updates):
                    # update agent using ppo
                    g_loss = policy.update(state, action, reward, action_log_p, state_)
                    
                state = []
                action = []
                action_log_p = []
                reward = []
                state_ = []

            running_reward += r

            s = s_

            if done:
                break

        # logging
        time_length = int(time.time() - start_time)
        print('#####################################################')
        print(
            'Time: {:02}:{:02}:{:02}\tEpisode {} \t reward in one episode: {} \nd_loss: {} \t g_loss: {} \nexpert_D: {} \t agent_D:{}'.format(
                time_length//3600, time_length%3600//60, time_length%60, i_epoch, running_reward, d_loss.mean().item(), g_loss, expert_D.mean().item(),
                agent_D.mean().item()))
        print('#####################################################')

        writer.add_scalar('Loss/discriminator_loss', d_loss.mean().item(), global_step=timestep)
        writer.add_scalar('Loss/generator_loss', g_loss, global_step=timestep)
        writer.add_scalar('D out/generator', agent_D.mean().item(), global_step=timestep)
        writer.add_scalar('D out/expert', expert_D.mean().item(), global_step=timestep)
        writer.add_scalar('reward', running_reward, global_step=timestep)

        torch.save(policy.net.state_dict(), './GAIL_policy_{}.pth'.format('single_panda_cube'))
        torch.save(discriminator.state_dict(), './GAIL_D_{}.pth'.format('single_panda_cube'))


