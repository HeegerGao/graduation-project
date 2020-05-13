'''
    单木块pick and place

    mpirun -np 8 python -u singlesingle.py

    5/12：先试试行不，如果不行，用Hind的代码换环境跑，无论如何跑出Pick and place
        不行，直接改他的吧。。。感觉世界观崩塌
    5/13：状态里面加入本体的关节速度
'''
import numpy as np
from mpi4py import MPI
import os, sys
sys.path.append("..") 
from utils import sync_networks, sync_grads, Actor, Critic, TrajectoryLoader, HERSampler, ReplayBuffer
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import time
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.const import PrimitiveShape

class Env():
    """
    Environment wrapper for Vrep
    """

    def __init__(self, scene_file=None, render=False):
        self.pr = PyRep()
        self.scene_file = scene_file
        self.render = render
        self.goal_range_x = [0.5, 0.7]
        self.goal_range_y = [-0.2, 0.2]
        self._launch()

    def _launch(self):
        self._headless = not self.render
        self.pr.launch(self.scene_file, headless=self._headless)
        self.pr.start()

        self.arm = Panda(0)
        self.gripper = PandaGripper(0)
        
        self.arm.set_control_loop_enabled(False)  #应该只是在不专家示教的情况下才用
        self.arm.set_motor_locked_at_zero_velocity(True)

        #设置到一个合理的初始位置
        self.arm.set_joint_positions([2.19369307e-03 ,-2.27448225e-01, 2.27380134e-02, -3.04887581e+00, -3.27670062e-03, 2.99315810e+00, 8.13528419e-01])
        self.pr.step()

        self.init_joint_angles = self.arm.get_joint_positions()

        self.red_cube = Shape('block1')
        self.goal = self._generate_goal()
        self.done = False

        self.launched = 1

    def _get_state(self):
        proprioceptive_state = self.arm.get_tip().get_position()
        proprioceptive_state = np.concatenate((proprioceptive_state, self.arm.get_joint_positions(), self.arm.get_joint_velocities()))  #7+7=14
        proprioceptive_state = np.concatenate((proprioceptive_state, self.gripper.get_open_amount()))[0:-1]
        proprioceptive_state[-1] = 1.0 if proprioceptive_state[-1] > 0.6 else 0.0
        
        return np.concatenate((proprioceptive_state, 
                                self.red_cube.get_position(),
                                self.goal)).reshape(1, -1)

    def reset(self):
        if self.launched == 0:
            self.launch()

        #防止关节转多了甩出去
        self.pr.stop()
        self.pr.start()

        #重新获得handle
        self.arm = Panda(0)
        self.gripper = PandaGripper(0)
        self.arm.set_control_loop_enabled(False)  #应该只是在不专家示教的情况下才用
        self.arm.set_motor_locked_at_zero_velocity(True)
        
        #设置到一个合理的初始位置
        self.arm.set_joint_positions([2.19369307e-03 ,-2.27448225e-01, 2.27380134e-02, -3.04887581e+00, -3.27670062e-03, 2.99315810e+00, 8.13528419e-01])
        self.pr.step()

        self.init_joint_angles = self.arm.get_joint_positions()

        self.red_cube = Shape('block1')

        self.goal = self._generate_goal()
        self.done = False

        self.red_cube.set_position([0.5 + 0.2 * np.random.random(), -0.2 + 0.4 * np.random.random(), 0.76999867])
        self.pr.step()

        while not self.gripper.actuate(1.0, 0.4):
            self.pr.step()

        self.goal = self._generate_goal()

        # 设置目标点位显示
        target = Shape.create(type=PrimitiveShape.SPHERE,
                size=[0.02, 0.02, 0.02],
                color=[1.0, 0.1, 0.1],
                static=True, respondable=False)
        target.set_position(self.goal)

        return self._get_state()

    def reset_from_demo(self, state):
        # 重置到某个指定的位置
        #防止关节转多了甩出去
        self.pr.stop()
        self.pr.start()

        #重新获得handle
        self.arm = Panda(0)
        self.gripper = PandaGripper(0)
        self.arm.set_control_loop_enabled(False)  #应该只是在不专家示教的情况下才用
        self.arm.set_motor_locked_at_zero_velocity(True)
            
        self.init_joint_angles = self.arm.get_joint_positions()

        self.red_cube = Shape('block1')
        self.goal = state[21:24]
        self.done = False

        self.red_cube.set_position(state[18:21])
        self.arm.set_joint_positions(state[3:10])
        self.pr.step()

        command = 1.0 if state[17] > 0.6 else 0.0

        #试试这样重置可以不
        while not self.gripper.actuate(command, 0.4):
            #一直和木块抓
            self.red_cube.set_position(state[18:21])

            if command == 0:
                self.gripper.grasp(self.red_cube)
            elif command == 1:
                self.gripper.release()
            self.pr.step()
        
        # 设置目标点位显示

        target = Shape.create(type=PrimitiveShape.SPHERE,
                size=[0.02, 0.02, 0.02],
                color=[1.0, 0.1, 0.1],
                static=True, respondable=False)
        target.set_position(self.goal)

        return self._get_state()

    def step(self, action):
        #要这么写才能防止pr多step
        command = 1.0 if action[-1] > 0.6 else 0.0
        self.gripper.actuate(command, 0.4)
        self.arm.set_joint_target_velocities(action[0:7])  # Execute action on arm
        self.pr.step()
        # self.arm.set_joint_target_velocities([0, 0, 0, 0, 0, 0, 0])

        if command == 0:
            self.gripper.grasp(self.red_cube)
        
        elif command == 1:
            self.gripper.release()

        current_state = self._get_state()
        cube_pos = current_state[0][18:21]
        goal_pos = self.goal
        r = self._compute_reward(cube_pos, goal_pos)
        
        if r == 1:
            self.done = True

        return current_state, r, self.done

    def _generate_goal(self):
        #自动生成目标点位置
        base_goal = [self.goal_range_x[0] + (self.goal_range_x[1] - self.goal_range_x[0]) * np.random.random(), self.goal_range_y[0] + (self.goal_range_y[1] - self.goal_range_y[0]) * np.random.random()]

        red_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + np.random.randint(0, 4) * 0.04])
        
        return red_goal

    def _compute_reward(self, cube_pos, goal_pos):
        r = 0
        dist = np.linalg.norm(cube_pos - goal_pos)
        if dist < 0.04:
            r += 1
        
        return r

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--replay-k', type=int, default=4, help='ratio for her to be replace')

    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--num-rollouts-per-cycle', type=int, default=2, help='the rollouts per update')
    parser.add_argument('--n-updates', type=int, default=40, help='the times to update the network per time')

    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=1024, help='the sample batch size')
    parser.add_argument('--demo-batch-size', type=int, default=256, help='the demo sample batch size')
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.2)

    parser.add_argument('--logdir', type=str, default='./log')
    parser.add_argument('--save-dir', type=str, default='./ddpg_models/', help='the path to save the models')

    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args

def compute_reward(cube_pos, goal_pos):
    length = cube_pos.shape[0]
    r = np.zeros(length,)
    for i in range(length):
        dist = np.linalg.norm(cube_pos[i] - goal_pos[i])
        if dist < 0.04:
            r[i] += 1
    
    return r.reshape((length, 1))

def get_env_params(): 
    params = {'obs': 18,
    'goal': 3,
    'action': 8,
    'action_min': -1,
    'action_max': 1,
    'max_timesteps': 120,
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

    def learn(self, transitions, demo_tras=None):
        if demo_tras == None:
            # no demos
            obs = torch.tensor(transitions['obs'], device=args.device, dtype=torch.float32)
            ag = torch.tensor(transitions['ag'], device=args.device, dtype=torch.float32)
            obs_next = torch.tensor(transitions['obs_next'], device=args.device, dtype=torch.float32)
            ag_next = torch.tensor(transitions['ag_next'], device=args.device, dtype=torch.float32)
            rew = torch.tensor(transitions['r'], device=args.device, dtype=torch.float32)
            act = torch.tensor(transitions['actions'], device=args.device, dtype=torch.float32)
            g = torch.tensor(transitions['g'], device=args.device, dtype=torch.float32)

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

            actions_real = self.choose_action(state, self.actor)
            actor_loss = - self.critic(state, actions_real).mean()
            actor_loss += (actions_real).pow(2).mean()  #l2 正则化，注意是动作输出的正则化而不是网络参数的正则化
            self.actor_optim.zero_grad()
            actor_loss.backward()
            sync_grads(self.actor)
            self.actor_optim.step()
            print('critic loss: ', critic_loss.item(), 'actor loss: ', actor_loss.item())

            self.sync_weight()
            return {
                'loss/actor': actor_loss.item(),
                'loss/critic': critic_loss.item(),
            }

        else:
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
            bc_loss = F.mse_loss(imitator_a, demo_a)   #一般来说数量级差十倍
            
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

def evaluation_function(policy, test_envs, d_state=None):
    policy.eval()
    if d_state == None:
        observation = test_envs.reset()
    else:
        observation = test_envs.reset_from_demo(d_state)
    obs = np.array([observation[0][0:18]])
    ag = np.array([observation[0][18:21]])
    g = np.array([observation[0][21:24]])

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
        
        obs_new = np.array([observation_new[0][0:18]])
        ag_new = np.array([observation_new[0][18:21]])
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

    #env
    args=get_args()
    env_params = get_env_params()
    writer = SummaryWriter(args.logdir)
    train_env = Env(scene_file='singlesingle.ttt', render=False)
    # seed
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # model
    actor = Actor(env_params['obs'] + 2 * env_params['goal'], env_params['action'],
        env_params['action_max']).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    # actor.load_state_dict(torch.load('./BC_{}.pth'.format('single_panda_block')))
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
    # loader = TrajectoryLoader('single_panda_cube_expert_trajectorys.csv')
    # expert_traj = loader.expert_trajectories[0:97]
    # expert_traj = np.delete(expert_traj, 0, axis=1)
    # #get the index of step state
    # reset_index = expert_traj[:, 0]
    # expert_traj = np.delete(expert_traj, 0, axis=1)
    # target_index = []   #表示reset的下标
    
    # for i in range(reset_index.shape[0] - 1):
    #     if reset_index[i] != reset_index[i+1]:
    #         target_index.append(i+1)
    # target_index.append(0)  #把初始状态也加上

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
                # d_state = expert_traj[np.random.randint(0, 60), 0:35] #随机选一个step的demo作为reset
                # d_state = expert_traj[0, 0:35] #随机选一个step的demo作为reset
                # observation = train_env.reset_from_demo(d_state)

                
                observation = train_env.reset()
                '''
                    'observation' = 0:18
                    'achieved_goal' = 18:21
                    'desired_goal' = 21:24
                '''
                obs = np.array([observation[0][0:18]])
                ag = np.array([observation[0][18:21]])
                g = np.array([observation[0][21:24]])
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
                    obs_new = np.array([observation_new[0][0:18]])
                    ag_new = np.array([observation_new[0][18:21]])
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
                # demo_tras = sample_demonstrations(loader, args.demo_batch_size)

                losses = policy.learn(transitions)

            # one cycle, one evaluation
            
            tr = evaluation_function(policy, train_env)
            time_length = time.time() - start_time
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('#############################################################################\n########################################################################')
                print('Time: {:02}:{:02}:{:02}'.format(time_length//3600, time_length%3600//60, time_length%60), 
                'epoch:', epoch, 'cycle:', cycle, '\nloss/actor:', losses['loss/actor'], 'loss/critic:', losses['loss/critic'], 'test_rew:', tr)
                print('#############################################################################\n########################################################################')

                writer.add_scalar('Loss/critic_loss', losses['loss/critic'], global_step=timestep_iter)
                writer.add_scalar('Loss/actor_loss', losses['loss/actor'], global_step=timestep_iter)
                # writer.add_scalar('Loss/bc_loss', losses['loss/bc'], global_step=timestep_iter)
                writer.add_scalar('reward', tr, global_step=timestep_iter)

                torch.save(policy.actor.state_dict(), args.save_dir + 'actor.pth')
                torch.save(policy.critic.state_dict(), args.save_dir + 'critic.pth')

