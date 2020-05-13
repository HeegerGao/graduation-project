'''
    mpirun -np 8 python -u single.py

    5/10：  真神奇，竟然能用了。注意他是在learn update的时候同步一下网络参数和网络梯度
            修改了一下test的重置，为真正随机
            修改timestep为300
    5/10：  开始尝试，仅在第一个环境下是否能学出来。
            1. 先学不reset from demo的
            2. 同时训练一个gail，也在第一个demo下
            3.发现巨大错误！ag_next忘了用！
    5/11：  思考：现在跑偏了
                把action末端的输出变为0~1
                1. 跑偏的时候，bc怎么说，现在的actor模型，同样也能在bc的专家的示教state处做出相同action
                2.  her的话，her算新奖励的时候，是ag和state作计算，而state里面是各个方块的state，不是末端位置
                    应该设为末端位置之类的，如末端位置的和方块的距离等等

                    尝试：reset不是从各阶段reset而是真实的随机reset，这样就有接触动力学了
                    尝试：修改了reset的env函数，好像成功了，能抓起来
                3.  同样的，如果gail的话，
            
            先训练拿一个100步的
            挂了，100步的都挂了。。。明天调一个只有一个方块的环境，然后训练一下Pick and place，如果还不行就完蛋了。。。

'''
import numpy as np
from mpi4py import MPI
import os, sys
from utils import sync_networks, sync_grads, Env, Actor, Critic, TrajectoryLoader, HERSampler, ReplayBuffer
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import time

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--replay-k', type=int, default=4, help='ratio for her to be replace')

    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--num-rollouts-per-cycle', type=int, default=2, help='the rollouts per update')
    parser.add_argument('--n-updates', type=int, default=40, help='the times to update the network per time')

    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--demo-batch-size', type=int, default=256, help='the demo sample batch size')
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)

    parser.add_argument('--logdir', type=str, default='./log')
    parser.add_argument('--save-dir', type=str, default='ddpg_models/', help='the path to save the models')

    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
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
    'max_timesteps': 100,
    'reward_func': compute_reward
    }

    return params


class DDPGPolicy(nn.Module):
    def __init__(self, actor, actor_optim, critic, critic_optim,
                 tau=0.005, gamma=0.99, exploration_noise=0.1,
                 action_range=None, **kwargs):
        super().__init__(**kwargs)
        if actor is not None:
            self.actor, self.actor_old = actor, deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim = actor_optim
        if critic is not None:
            self.critic, self.critic_old = critic, deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim = critic_optim
        
        sync_networks(self.actor)
        sync_networks(self.critic)
        assert 0 <= tau <= 1, 'tau should in [0, 1]'
        self._tau = tau
        assert 0 <= gamma <= 1, 'gamma should in [0, 1]'
        self._gamma = gamma
        assert 0 <= exploration_noise, 'noise should not be negative'
        self._eps = exploration_noise
        assert action_range is not None
        self._range = action_range
        self._action_bias = (action_range[0] + action_range[1]) / 2
        self._action_scale = (action_range[1] - action_range[0]) / 2

    def set_eps(self, eps):
        """Set the eps for exploration."""
        self._eps = eps

    def train(self):
        """Set the module in training mode, except for the target network."""
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        """Set the module in evaluation mode, except for the target network."""
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def sync_weight(self):
        """Soft-update the weight for the target network."""
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic_old.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def choose_action(self, state, actor, use_eps=True):
        logits = actor(state)
        logits += self._action_bias
        logits[0][7] = (logits[0][7] + 1) / 2.0   #末端执行器的动作范围为[0~1]
        
        if use_eps:
            if self._eps > 0:
                logits += torch.randn(
                    size=logits.shape, device=logits.device) * self._eps
        logits = logits.clamp(self._range[0], self._range[1])
        # print(logits)
        return logits

    def learn(self, transitions, demo_tras):
        # use demos
        obs = np.concatenate((transitions['obs'], demo_tras['obs']))
        ag = np.concatenate((transitions['ag'], demo_tras['ag']))
        ag_next = np.concatenate((transitions['ag_next'], demo_tras['ag_next']))
        obs_next = np.concatenate((transitions['obs_next'], demo_tras['obs_next']))
        rew = np.concatenate((transitions['r'], demo_tras['r']))
        act = np.concatenate((transitions['actions'], demo_tras['actions']))
        g = np.concatenate((transitions['g'], demo_tras['g']))

        obs = torch.tensor(obs, device=args.device, dtype=torch.float32)
        ag = torch.tensor(ag, device=args.device, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, device=args.device, dtype=torch.float32)
        ag_next = torch.tensor(ag_next, device=args.device, dtype=torch.float32)
        rew = torch.tensor(rew, device=args.device, dtype=torch.float32)
        act = torch.tensor(act, device=args.device, dtype=torch.float32)
        g = torch.tensor(g, device=args.device, dtype=torch.float32)
        
        demo_obs = torch.tensor(demo_tras['obs'], device=args.device, dtype=torch.float32)
        demo_ag = torch.tensor(demo_tras['ag'], device=args.device, dtype=torch.float32)
        demo_g = torch.tensor(demo_tras['g'], device=args.device, dtype=torch.float32)
        demo_a = torch.tensor(demo_tras['actions'], device=args.device, dtype=torch.float32)
        demo_state = torch.cat((demo_obs, demo_ag, demo_g), 1)
        imitator_a = self.choose_action(demo_state, self.actor)

        state = torch.cat((obs, ag, g), 1)
        state_next = torch.cat((obs_next, ag_next, g), 1)   #因为obs和obs_next的goal是一样的

        with torch.no_grad():
            target_q = self.critic_old(state_next, self.choose_action(state_next, self.actor_old))
            target_q = rew + self._gamma * target_q

        current_q = self.critic(state, act)
        
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()
        
        #bc loss
        bc_loss = 10 * F.mse_loss(imitator_a, demo_a)   #一般来说数量级差十倍
        
        actions_real = self.choose_action(state, self.actor)
        actor_loss = - self.critic(state, actions_real).mean()
        actor_loss += (actions_real).pow(2).mean()  #l2 正则化，注意是动作输出的正则化而不是网络参数的正则化
        actor_loss += bc_loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()
        print('bc loss: ', bc_loss.item(), 'critic loss: ', critic_loss.item(), 'actor loss: ', actor_loss.item() - bc_loss.item())

        self.sync_weight()
        return {
            'loss/actor': actor_loss.item() - bc_loss.item(),
            'loss/critic': critic_loss.item(),
            'loss/bc': bc_loss.item()
        }


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

def evaluation_function(policy, test_envs, d_state):
    policy.eval()
    # observation = test_envs.reset()
    observation = test_envs.reset_from_demo(d_state)
    obs = np.array([observation[0][0:11]])
    ag = np.array([observation[0][11:23]])
    g = np.array([observation[0][23:35]])

    r = 0
    # start to collect samples
    
    for t in range(env_params['max_timesteps']):    #注意这个循环不收集reward，只在her sample batch的时候计算
        with torch.no_grad():
            state = np.concatenate([obs, ag, g], axis=1)
            state = torch.tensor(state, device=args.device, dtype=torch.float32)
            action = policy.choose_action(state, policy.actor, use_eps=False)   #测试时不需要random

        # feed the actions into the environment
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        observation_new, rew, _= test_envs.step(action[0])
        
        obs_new = np.array([observation_new[0][0:11]])
        ag_new = np.array([observation_new[0][11:23]])
        # re-assign the observation
        obs = obs_new
        ag = ag_new

        r += rew

    return r #表示该回合内的总奖励

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # # 绘动态图
    # plt.ion()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(4,1,1)
    # ax2 = fig.add_subplot(4,1,2)
    # ax3 = fig.add_subplot(4,1,3)
    # ax4 = fig.add_subplot(4,1,4)
    
    # iter_re = []
    # loss_actor = []
    # loss_critic = []
    # loss_bc = []
    # test_rew = []

    #env
    args=get_args()
    env_params = get_env_params()
    writer = SummaryWriter(args.logdir)
    train_env = Env(scene_file='single.ttt', render=False)
    # seed
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # model
    actor = Actor(env_params['obs'] + 2 * env_params['goal'], env_params['action'],
        env_params['action_max']).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    actor.load_state_dict(torch.load('./BC_{}.pth'.format('single_panda_block')))
    # actor.load_state_dict(torch.load(args.save_dir + 'actor.pth'))

    critic = Critic(env_params['obs'] + 2 * env_params['goal'], env_params['action']
        ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    # critic.load_state_dict(torch.load(args.save_dir + 'critic.pth'))

    #policy
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        args.tau, args.gamma, args.exploration_noise,
        [env_params['action_min'], env_params['action_max']])

    # her sampler
    her_module = HERSampler(args.replay_k, env_params['reward_func'])   #不用HER
    # create the replay buffer
    buffer = ReplayBuffer(env_params, args.buffer_size, her_module.sample_her_transitions)
    
    # demos
    loader = TrajectoryLoader('single_panda_cube_expert_trajectorys.csv')
    expert_traj = loader.expert_trajectories[0:97]
    expert_traj = np.delete(expert_traj, 0, axis=1)
    #get the index of step state
    reset_index = expert_traj[:, 0]
    expert_traj = np.delete(expert_traj, 0, axis=1)
    target_index = []   #表示reset的下标
    
    for i in range(reset_index.shape[0] - 1):
        if reset_index[i] != reset_index[i+1]:
            target_index.append(i+1)
    target_index.append(0)  #把初始状态也加上

    # train
    start_time = time.time()
    timestep_iter = 0
    for epoch in range(args.n_epochs):
        for cycle in range(args.n_cycles):
            policy.train()
            '''
            三维，第一维是第几条轨迹，第二维是timesetps，第三维是每个数据的维数
            obs 和 ag都要第二维+1，是因为最后存了next
            '''
            mb_obs = np.empty([args.num_rollouts_per_cycle, env_params['max_timesteps'] + 1, env_params['obs']])
            mb_ag = np.empty([args.num_rollouts_per_cycle, env_params['max_timesteps'] + 1, env_params['goal']])
            mb_g = np.empty([args.num_rollouts_per_cycle, env_params['max_timesteps'], env_params['goal']])
            mb_actions = np.empty([args.num_rollouts_per_cycle, env_params['max_timesteps'], env_params['action']])
            for roll_num in range(args.num_rollouts_per_cycle):
                # reset the rollouts
                ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []    #按照单个transition存储
                # reset the environment
                d_state = expert_traj[np.random.randint(0, 60), 0:35] #随机选一个step的demo作为reset
                # d_state = expert_traj[0, 0:35] #随机选一个step的demo作为reset
                observation = train_env.reset_from_demo(d_state)

                
                # observation = train_env.reset()
                '''
                    'observation' = 0:11
                    'achieved_goal' = 11:23
                    'desired_goal' = 23:35
                '''
                obs = np.array([observation[0][0:11]])
                ag = np.array([observation[0][11:23]])
                g = np.array([observation[0][23:35]])
                # start to collect samples
                for t in range(env_params['max_timesteps']):    #注意这个循环不收集reward，只在her sample batch的时候计算
                    print(t)
                    # time.sleep(1)
                    timestep_iter += 1
                    with torch.no_grad():
                        state = np.concatenate([obs, ag, g], axis=1)
                        state = torch.tensor(state, device=args.device, dtype=torch.float32)
                        action = policy.choose_action(state, policy.actor)
                    # feed the actions into the environment
                    if isinstance(action, torch.Tensor):
                        action = action.detach().cpu().numpy()
                    # writer.add_scalar('action mean', action.mean(), global_step=timestep_iter)
                    observation_new, _, _ = train_env.step(action[0])
                    obs_new = np.array([observation_new[0][0:11]])
                    ag_new = np.array([observation_new[0][11:23]])
                    # append rollouts
                    ep_obs.append(obs[0].copy())
                    ep_ag.append(ag[0].copy())
                    ep_g.append(g[0].copy())
                    ep_actions.append(action[0].copy())
                    # re-assign the observation
                    obs = obs_new
                    ag = ag_new
                ep_obs.append(obs[0].copy())
                ep_ag.append(ag[0].copy())
                
                mb_obs[roll_num, :, :] = np.array(ep_obs)
                mb_ag[roll_num, :, :] = np.array(ep_ag)
                mb_g[roll_num, :, :] = np.array(ep_g)
                mb_actions[roll_num, :, :] = np.array(ep_actions)
                
            # store the episodes
            buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            
            #update
            for _ in range(args.n_updates):
                #HER samples are already in transitions, and transitions are a list structure, not episode structure
                transitions = buffer.sample(args.batch_size)
                demo_tras = sample_demonstrations(loader, args.demo_batch_size)

                losses = policy.learn(transitions, demo_tras)

            # one cycle, one evaluation
            
            tr = evaluation_function(policy, train_env, expert_traj[0, 0:35])
            time_length = time.time() - start_time
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('#############################################################################\n########################################################################')
                print('Time: {:02}:{:02}:{:02}'.format(time_length//3600, time_length%3600//60, time_length%60), 
                'epoch:', epoch, 'cycle:', cycle, '\nloss/actor:', losses['loss/actor'], 'loss/critic:', losses['loss/critic'], 'test_rew:', tr)
                print('#############################################################################\n########################################################################')

                writer.add_scalar('Loss/critic_loss', losses['loss/critic'], global_step=timestep_iter)
                writer.add_scalar('Loss/actor_loss', losses['loss/actor'], global_step=timestep_iter)
                writer.add_scalar('Loss/bc_loss', losses['loss/bc'], global_step=timestep_iter)
                writer.add_scalar('reward', tr, global_step=timestep_iter)
            
            # iter_re.append(timestep_iter)
            # loss_actor.append(losses['loss/actor'])
            # loss_critic.append(losses['loss/critic'])
            # loss_bc.append(losses['loss/bc'])
            # test_rew.append(tr)

            # ax1.cla()
            # ax2.cla()
            # ax3.cla()
            # ax4.cla()

            # ax1.plot(iter_re, loss_actor, 'b',label='loss of actor')
            # ax2.plot(iter_re, loss_critic, 'r',label='loss of critic')
            # ax3.plot(iter_re, loss_bc, 'g',label='loss of bc')
            # ax4.plot(iter_re, test_rew, 'g',label='test reward')
            # ax1.legend()
            # ax2.legend()
            # ax3.legend()
            # ax4.legend()

            # plt.draw()
            # plt.draw()
            # plt.draw()
            # plt.draw()
            # plt.pause(0.01)

                torch.save(policy.actor.state_dict(), args.save_dir + 'actor.pth')
                torch.save(policy.critic.state_dict(), args.save_dir + 'critic.pth')

