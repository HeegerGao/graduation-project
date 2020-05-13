'''
    single panda 4 blocks using:
    1) demonstrations
    2) reset from demonstrations
    3) ddpg + her
    4) bc loss + rl loss

    log:
    5/7:
        1) √，一会改一下demo的输入位置，要算demo的Q和loss，所以应该在learn里面？ √
        2) √
    5/8:
        3) √
        4) √
        一堆nan又出来了！！！！！！！！草！！！！！！！
        排查：  不是demo数据的问题
                原来没加bcloss 的时候是没有nan的
                进展：当loss有nan的时候，就return，放弃这个learning
                但是，训练一段时间之后所有的都会变成跳过放弃

                不是reset from demo的问题，虽然一开始有错（没写成set joint position 而是 set joint target position）
                
    5/10：保留，因为有多个并行环境的各种transpose
'''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from copy import deepcopy
from utils import Actor, Critic, Env, ParallelEnv, ReplayBuffer, HERSampler, TrajectoryLoader
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='FetchReach-v1')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--replay-k', type=int, default=4, help='ratio for her to be replace')

    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--num-rollouts-per-cycle', type=int, default=2, help='the rollouts per update')
    parser.add_argument('--n-updates', type=int, default=40, help='the times to update the network per time')
    
    parser.add_argument('--train-num', type=int, default=5)
    parser.add_argument('--test-num', type=int, default=10)

    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=4096, help='the sample batch size')
    parser.add_argument('--demo-batch-size', type=int, default=1024, help='the demo sample batch size')
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)

    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--layer-num', type=int, default=4)
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio, mmp jing ran hai yao clip')
    
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0., help='0 for no render, 0.001 for render')

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
    'action_min': -2.1,
    'action_max': 2.1,
    'max_timesteps': 350,
    'reward_func': compute_reward
    }

    return params

def evaluation_function(policy, test_envs):
    # print('testing')
    # evaluation using test_envs
    policy.eval()
    observation = test_envs.reset().reshape((args.test_num, env_params['obs'] + 2 * env_params['goal']))  
    obs = np.array([observation[i][0:11] for i in range(args.test_num)])
    ag = np.array([observation[i][11:23] for i in range(args.test_num)])
    g = np.array([observation[i][23:35] for i in range(args.test_num)])

    av_r = 0
    # start to collect samples
    
    for t in range(env_params['max_timesteps']):    #注意这个循环不收集reward，只在her sample batch的时候计算
        with torch.no_grad():
            state = np.concatenate([obs, ag, g], axis=1)
            state = torch.tensor(state, dtype=torch.float32)
            if args.device == 'cuda':
                state = state.cuda()
            action = policy.choose_action(state, policy.actor)

        # feed the actions into the environment
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        # action[np.isnan(action)] = 0.

        observation_new, rew, _= test_envs.step(action)
        observation_new = observation_new.reshape((args.test_num, env_params['obs'] + 2 * env_params['goal']))
        # print('testing is ok')
        
        obs_new = np.array([observation_new[i][0:11] for i in range(args.test_num)])
        ag_new = np.array([observation_new[i][11:23] for i in range(args.test_num)])
        # re-assign the observation
        obs = obs_new
        ag = ag_new
        rew = np.array(rew)
        
        av_r += rew.mean()

    return av_r

def sample_demonstrations(demo_loader, sample_num):
    demos_no_reward = demo_loader.sample_trajectories_no_shuffle(sample_num)
    assert demos_no_reward.shape[1] == 43
    demo_transitions = {}
    if np.isnan(demos_no_reward).any():
        print('demo data nan !!!!!!!!!!!!!!!!')
        exit(0)
    # print('demo data max:', np.max(demos_no_reward))
    demo_transitions['obs'] = demos_no_reward[:-1, 0:11]
    demo_transitions['ag'] = demos_no_reward[:-1, 11:23]
    demo_transitions['g'] = demos_no_reward[:-1, 23:35]
    demo_transitions['actions'] = demos_no_reward[:-1, 35:43]
    demo_transitions['obs_next'] = demos_no_reward[1:, 0:11]
    demo_transitions['ag_next'] = demos_no_reward[1:, 11:23]
    demo_transitions['r'] = compute_reward(demo_transitions['ag_next'], demo_transitions['g'])

    return demo_transitions

def merge_transition(trans, demo_tras):
    # 合并两个trans
    for key in trans.keys():
        tras[key] = np.concatenate((trans[key], demo_tras[key])) 
    return tras

class DDPGPolicy(nn.Module):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic
        network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param action_range: the action range (minimum, maximum).
    :type action_range: [float, float]
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor, actor_optim, critic, critic_optim,
                 tau=0.005, gamma=0.99, exploration_noise=0.1,
                 action_range=None, reward_normalization=False,
                 ignore_done=False, **kwargs):
        super().__init__(**kwargs)
        if actor is not None:
            self.actor, self.actor_old = actor, deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim = actor_optim
        if critic is not None:
            self.critic, self.critic_old = critic, deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim = critic_optim
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
        # it is only a little difference to use rand_normal
        # self.noise = OUNoise()
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()

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

    # def process_fn(self, batch, buffer, indice):
    #     if self._rew_norm:
    #         bfr = buffer.rew[:min(len(buffer), 1000)]  # avoid large buffer
    #         mean, std = bfr.mean(), bfr.std()
    #         if std > self.__eps:
    #             batch.rew = (batch.rew - mean) / std
    #     if self._rm_done:
    #         batch.done = batch.done * 0.
    #     return batch

    def choose_action(self, state, actor):
        # print(state.shape)
        logits = actor(state)
        logits += self._action_bias
         
        if self._eps > 0:
            logits += torch.randn(
                size=logits.shape, device=logits.device) * self._eps
        logits = logits.clamp(self._range[0], self._range[1])
        return logits

    def learn(self, transitions, demo_tras):
        # use demos
        obs = np.concatenate((transitions['obs'], demo_tras['obs']))
        ag = np.concatenate((transitions['ag'], demo_tras['ag']))
        obs_next = np.concatenate((transitions['obs_next'], demo_tras['obs_next']))
        rew = np.concatenate((transitions['r'], demo_tras['r']))
        act = np.concatenate((transitions['actions'], demo_tras['actions']))
        g = np.concatenate((transitions['g'], demo_tras['g']))
        # print('rew:', rew)

        obs = torch.tensor(obs, device=args.device, dtype=torch.float32)
        ag = torch.tensor(ag, device=args.device, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, device=args.device, dtype=torch.float32)
        rew = torch.tensor(rew, device=args.device, dtype=torch.float32)
        act = torch.tensor(act, device=args.device, dtype=torch.float32)
        g = torch.tensor(g, device=args.device, dtype=torch.float32)

        demo_obs = torch.tensor(demo_tras['obs'], device=args.device, dtype=torch.float32)
        demo_ag = torch.tensor(demo_tras['ag'], device=args.device, dtype=torch.float32)
        demo_g = torch.tensor(demo_tras['g'], device=args.device, dtype=torch.float32)
        demo_a = torch.tensor(demo_tras['actions'], device=args.device, dtype=torch.float32)
        demo_state = torch.cat((demo_obs, demo_ag, demo_g), 1)

        # print('demo_state max: ', torch.max(demo_state))
        
        state = torch.cat((obs, ag, g), 1)
        state_next = torch.cat((obs_next, ag, g), 1)   #因为obs和obs_next的goal是一样的

        # if torch.any(state != state) or torch.any(state_next != state_next):
        #     print('state is nan!!!!!!!!!!!!!!!!!!!')
        #     print('state is : ', state)
        #     print((state != state).nonzero())
        #     print('!!!!!!jump!!!!!!!!!!!')
        #     return {
        #         'loss/actor': 0,
        #         'loss/critic': 0,
        #     }
            # exit(0)


        # print('state before choose: ', state)
        # print('state next before choose: ', state_next)
        with torch.no_grad():
            target_q = self.critic_old(state_next, self.choose_action(state_next, self.actor_old))
            target_q = rew + self._gamma * target_q

        # if (torch.abs(target_q) > 1e10).any():
        #     print('target q too large!!!!!!!!!!!!!!!')
        #     print('state: ', state, 'max in state: ', torch.max(state), 'location: ', (state == torch.max(state)).nonzero())
        #     print('state_next: ', state_next, 'max in state_next: ', torch.max(state_next), 'location: ', (state_next == torch.max(state_next)).nonzero())
        #     print('next action:', self.choose_action(state_next, self.actor_old))
        #     tq1 = self.critic_old(state_next, self.choose_action(state_next, self.actor_old))
        #     print('tq1: ', tq1, 'max in tq1: ', torch.max(tq1))
        #     print('tq1 shape:', self.critic_old(state_next, self.choose_action(state_next, self.actor_old)).shape)
        #     print('rew:', rew, 'rew shape: ', rew.shape)
        #     tq2 = rew + self._gamma * target_q
        #     print('tq2: ', tq2, 'tq2 shape:', tq2.shape, 'max in tq2:', torch.max(tq2))
            # exit(0)
        current_q = self.critic(state, act)
        # if (torch.abs(current_q) > 1e10).any():
        #     print('current q too large!!!!!!!!!!!!!!!')
        #     print('state: ', state, 'max in state: ', torch.max(state), 'location: ', (state == torch.max(state)).nonzero())
        #     print('act: ', act, 'max in act: ', torch.max(act), 'location: ', (act == torch.max(act)).nonzero())
        #     print('cq:', self.critic(state, act))
            # exit(0)
            
        # print('current_q:', current_q)
        # print('target_q:', target_q)

        # q filter
        imitator_a = self.choose_action(demo_state, self.actor)
        with torch.no_grad():
            demo_q = self.critic(demo_state, demo_a)
            imitator_q = self.critic(demo_state, imitator_a)
            # if :
                # print('demo_q is nan!!!!!!!!!!!!!!!!!!!')
                # # exit(0)
            if torch.any(imitator_a != imitator_a) or torch.any(demo_q != demo_q) or torch.any(imitator_q != imitator_q):
                print('imitator_a or imitator_q or demo_q is nan!!!!!!!!!!!!!!!!!!!')
                return {
                    'loss/actor': 0,
                    'loss/critic': 0,
                }
                # exit(0)
            flags = demo_q > imitator_q
            flags = flags.nonzero()[:, 0] #索引
            # print('flags:', flags)

        # bc loss
        use_bc_flag = True
        if epoch <= 10:  # 一开始全部都要bc
            bc_loss = F.mse_loss(imitator_a, demo_a)
        else:
            bc_loss = F.mse_loss(imitator_a[flags], demo_a[flags])
        # print('bc loss:', bc_loss)
        if torch.any(bc_loss != bc_loss):
            # print('bcloss is nan!!!!!!!!!!!!!!!!!!!')
            # print('demo_q , imitator_q: ', demo_q , imitator_q)
            use_bc_flag = False
        
        critic_loss = F.mse_loss(current_q, target_q)
        if critic_loss != critic_loss or torch.abs(critic_loss) > 20:
            print('!!!!!!jump!!!!!!!!!!!')
            return {
                'loss/actor': 0,
                'loss/critic': 0,
            }
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(state, self.choose_action(state, self.actor)).mean()
        if use_bc_flag == True:
            actor_loss += bc_loss   # 加入bc loss，没加权重
        # actor_loss = actor_loss.mean()
        if actor_loss != actor_loss or torch.abs(actor_loss) > 20:
            print('!!!!!!jump!!!!!!!!!!!')
            return {
                'loss/actor': 0,
                'loss/critic': 0,
            }
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        print('bc loss: ', bc_loss.item(), 'critic loss: ', critic_loss.item(), 'actor loss: ', actor_loss.item())

        self.sync_weight()
        return {
            'loss/actor': actor_loss.item(),
            'loss/critic': critic_loss.item(),
        }


if __name__ == '__main__':
    # 绘动态图
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    iter_re = []
    loss_actor = []
    loss_critic = []
    test_rew = []
    iter = 0

    #env
    args=get_args()
    env_params = get_env_params()
    train_envs = ParallelEnv(args.train_num)
    test_envs = ParallelEnv(args.test_num)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    actor = Actor(
        args.layer_num, env_params['obs'] + 2 * env_params['goal'], env_params['action'],
        env_params['action_max'], args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic = Critic(
        args.layer_num, env_params['obs'] + 2 * env_params['goal'], env_params['action'], args.device
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    #policy
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        args.tau, args.gamma, args.exploration_noise,
        [env_params['action_min'], env_params['action_max']],
        reward_normalization=False, ignore_done=False)

    # her sampler
    her_module = HERSampler(args.replay_k, env_params['reward_func'])
    # create the replay buffer
    buffer = ReplayBuffer(env_params, args.buffer_size, her_module.sample_her_transitions)
    
    # demos
    loader = TrajectoryLoader('single_panda_cube_expert_trajectorys.csv')
    expert_traj = loader.expert_trajectories
    expert_traj = np.delete(expert_traj, 0, axis=1)
    #get the index of step state
    reset_index = expert_traj[:, 0]
    expert_traj = np.delete(expert_traj, 0, axis=1)
    target_index = []   #表示reset的下标
    
    for i in range(reset_index.shape[0] - 1):
        if reset_index[i] != reset_index[i+1]:
            target_index.append(i+1)
    
    # train
    start_time = time.time()
    for epoch in range(args.n_epochs):
        for cycle in range(args.n_cycles):
            policy.train()
            mb_obs = np.empty([args.num_rollouts_per_cycle * args.train_num, env_params['max_timesteps'] + 1, env_params['obs']])
            mb_ag = np.empty([args.num_rollouts_per_cycle * args.train_num, env_params['max_timesteps'] + 1, env_params['goal']])
            mb_g = np.empty([args.num_rollouts_per_cycle * args.train_num, env_params['max_timesteps'], env_params['goal']])
            mb_actions = np.empty([args.num_rollouts_per_cycle * args.train_num, env_params['max_timesteps'], env_params['action']])
            for roll_num in range(args.num_rollouts_per_cycle):
                print('start a new roll!!!!!!!!')
                # reset the rollouts
                ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []    #按照单个transition存储
                # reset the environment
                demo_states = []
                d_state = expert_traj[target_index[np.random.randint(0, len(target_index))], 0:35] #随机选一个step的demo作为reset
                for _ in range(args.train_num):
                    demo_states.append(d_state)
                observation = train_envs.reset_from_demo(np.array(demo_states))
                # observation = train_envs.reset()
                observation = observation.reshape((args.train_num, env_params['obs'] + 2 * env_params['goal']))  
                # print('reset obs: ', observation)
                print('reset ok!!!!!!!!')
                # print('epoch:', epoch, ' cycle:', cycle, ' roll:', roll_num)
                # print(observation.shape)
                # print('here!!!', observation)
                '''
                    'observation' = 0:11
                    'achieved_goal' = 11:23
                    'desired_goal' = 23:35
                '''
                obs = np.array([observation[i][0:11] for i in range(args.train_num)])
                ag = np.array([observation[i][11:23] for i in range(args.train_num)])
                g = np.array([observation[i][23:35] for i in range(args.train_num)])

                # start to collect samples
                for t in range(env_params['max_timesteps']):    #注意这个循环不收集reward，只在her sample batch的时候计算
                    with torch.no_grad():
                        print('timestep: ', t)
                        state = np.concatenate([obs, ag, g], axis=1)
                        state = torch.tensor(state, dtype=torch.float32)
                        if args.device == 'cuda':
                            state = state.cuda()
                        # print('train')
                        action = policy.choose_action(state, policy.actor)
                        # print('training is ok')
                    # feed the actions into the environment
                    if isinstance(action, torch.Tensor):
                        action = action.detach().cpu().numpy()
                        # print(action)
                        assert action.shape[1] == 8
                    if np.isnan(action).any():
                        print('action is nan!!!!!!!!!!!!!!!!!!!')
                        exit(0)
                    # action[np.isnan(action)] = 0.
                    observation_new, _, _ = train_envs.step(action)
                    observation_new = observation_new.reshape((args.train_num, env_params['obs'] + 2 * env_params['goal']))  
                    obs_new = np.array(observation_new[0:args.train_num, 0:11])
                    ag_new = np.array(observation_new[0:args.train_num, 11:23])
                    # print('observation new: ', observation_new)
                    # print(action)
                    if np.isnan(observation_new).any():
                        print('obs new is nan!!!!!!!!!!!!!!!!!!')
                        exit(0)
                    if np.isinf(observation_new).any():
                        print('obs new is inf!!!!!!!!!!!!!!!!!!')
                        exit(0)

                    # append rollouts
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_g.append(g.copy())
                    ep_actions.append(action.copy())
                    # re-assign the observation
                    obs = obs_new
                    ag = ag_new
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                # print('ep_obs:', np.array(ep_obs).shape)
                ep_obs = np.array(ep_obs).transpose((1,0,2))
                ep_ag = np.array(ep_ag).transpose((1,0,2))
                ep_g = np.array(ep_g).transpose((1,0,2))
                ep_actions = np.array(ep_actions).transpose((1,0,2))
                # ep_actions[np.isnan(ep_actions)] = 0.
                
                mb_obs[roll_num:roll_num+args.train_num, :, :] = ep_obs
                mb_ag[roll_num:roll_num+args.train_num, :, :] = ep_ag
                mb_g[roll_num:roll_num+args.train_num, :, :] = ep_g
                mb_actions[roll_num:roll_num+args.train_num, :, :] = ep_actions
                
            # store the episodes
            buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            
            #update
            print('updating!!!!!!!!')
            for _ in range(args.n_updates):
                #HER samples are already in transitions, and transitions are a list structure, not episode structure
                transitions = buffer.sample(args.batch_size)
                demo_tras = sample_demonstrations(loader, args.demo_batch_size)
                print('data loaded!!!!!!!!')
                # print('transitions max:', np.max(transitions))
                # print('demo max: ', np.max(demo_tras))
                losses = policy.learn(transitions, demo_tras)
                print('learning ok!!!!!!!!')

            # one cycle, one evaluation
            iter += 1
            tr = evaluation_function(policy, test_envs)
            time_length = time.time() - start_time
            print('#############################################################################\n########################################################################')
            print('Time: {:02}:{:02}:{:02}'.format(time_length//3600, time_length%3600//60, time_length%60), 
            'epoch:', epoch, 'cycle:', cycle, '\nloss/actor:', losses['loss/actor'], 'loss/critic:', losses['loss/critic'], 'test_rew:', tr)
            print('#############################################################################\n########################################################################')
            iter_re.append(iter)
            loss_actor.append(losses['loss/actor'])
            loss_critic.append(losses['loss/critic'])
            test_rew.append(tr)
            
            ax1.cla()
            ax2.cla()
            ax3.cla()

            ax1.plot(iter_re, loss_actor, 'b',label='loss of actor')
            ax2.plot(iter_re, loss_critic, 'r',label='loss of critic')
            ax3.plot(iter_re, test_rew, 'r',label='test reward')
            ax1.legend()
            ax2.legend()
            ax3.legend()

            plt.draw()
            plt.draw()
            plt.draw()
            plt.pause(0.01)

        torch.save(policy.actor.state_dict(), args.save_dir + 'actor.pth')
        torch.save(policy.critic.state_dict(), args.save_dir + 'critic.pth')

        record = np.concatenate((iter_re, loss_actor, loss_critic, test_rew))
        data = pd.DataFrame(expert_traj)
        data.to_csv('./{}_training_recording.csv'.format('single_panda_cube'))
    train_envs.close()
    test_envs.close()