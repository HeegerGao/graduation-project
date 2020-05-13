'''
    1. 读取专家轨迹
    2. AC网络
    3. ReplayBuffer
    4. HERSampler
    5. VectorEnv
    6. Env
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mpi4py import MPI
from torch.distributions import Beta
from copy import deepcopy
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.const import PrimitiveShape
import torch.multiprocessing as mp
import threading


class TrajectoryLoader(object):
    def __init__(self, filepath):
        super(object, self).__init__()
        self.filepath = filepath
        self.expert_trajectories = np.array(pd.read_csv(self.filepath))
        self.expert_trajectories = np.delete(self.expert_trajectories, 0, axis=0) #去掉第一行
        self.expert_trajectories =np.delete(self.expert_trajectories, 0, axis=1) #去掉第一列，没用的第一列，不是episod序号

    def sample_trajectories(self, batch_size):
        row_rand_array = np.arange(self.expert_trajectories.shape[0])
        np.random.shuffle(row_rand_array)
        row_rand = self.expert_trajectories[row_rand_array[0:batch_size]]
        return row_rand

    def sample_trajectories_no_shuffle(self, batch_size):   #不打乱顺序的随机
        begin = np.random.randint(0, self.expert_trajectories[0:275].shape[0] - batch_size)
        print(begin)
        return self.expert_trajectories[begin:(begin + batch_size), 2:]


class Env():
    """
    Environment wrapper for Vrep
    """

    def __init__(self, scene_file=None, render=False):
        self.pr = PyRep()
        self.scene_file = scene_file
        self.render = render
        self.goal_range_x = [0.5, 0.7]
        self.goal_range_y = [-0.4, 0.4]
        self._launch()

    def _launch(self):
        self._headless = not self.render
        self.pr.launch(self.scene_file, headless=self._headless)
        self.pr.start()

        self.arm = Panda(0)
        self.gripper = PandaGripper(0)
        
        self.arm.set_control_loop_enabled(False)  #应该只是在不专家示教的情况下才用
        self.arm.set_motor_locked_at_zero_velocity(True)

        self.init_joint_angles = self.arm.get_joint_positions()

        self.red_cube = Shape('block1')
        self.blue_cube = Shape('block2')
        self.green_cube = Shape('block3')
        self.yellow_cube = Shape('block4')
        self.blocks = [self.red_cube, self.blue_cube, self.green_cube, self.yellow_cube]
        self.goal = self._generate_goal()
        self.done = False

        self.launched = 1

    def _get_state(self):
        proprioceptive_state = self.arm.get_tip().get_position()
        proprioceptive_state = np.concatenate((proprioceptive_state, self.arm.get_joint_positions()))  #7
        proprioceptive_state = np.concatenate((proprioceptive_state, self.gripper.get_open_amount()))[0:-1]
        proprioceptive_state[-1] = 1.0 if proprioceptive_state[-1] > 0.6 else 0.0
        
        return np.concatenate((proprioceptive_state, 
                                self.red_cube.get_position(),
                                self.blue_cube.get_position(),
                                self.green_cube.get_position(),
                                self.yellow_cube.get_position(),
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
            
        self.init_joint_angles = self.arm.get_joint_positions()

        self.red_cube = Shape('block1')
        self.blue_cube = Shape('block2')
        self.green_cube = Shape('block3')
        self.yellow_cube = Shape('block4')
        self.blocks = [self.red_cube, self.blue_cube, self.green_cube, self.yellow_cube]
        self.goal = self._generate_goal()
        self.done = False

        self.red_cube.set_position([0.5 + 0.1 * np.random.random(), -0.4 + 0.4 * np.random.random(), 0.76999867])
        self.blue_cube.set_position([0.5 + 0.1 * np.random.random(), 0.0 + 0.4 * np.random.random(), 0.76999867])
        self.green_cube.set_position([0.6 + 0.1 * np.random.random(), -0.4 + 0.4 * np.random.random(), 0.76999867])
        self.yellow_cube.set_position([0.6 + 0.1 * np.random.random(), 0.0 + 0.4 * np.random.random(), 0.76999867])
        self.pr.step()

        while not self.gripper.actuate(1.0, 0.4):
            self.pr.step()

        self.goal = self._generate_goal()

        # 设置目标点位显示
        tmp_g = self.goal.reshape(4, 3)
        for g in tmp_g:
            target = Shape.create(type=PrimitiveShape.SPHERE,
                    size=[0.02, 0.02, 0.02],
                    color=[1.0, 0.1, 0.1],
                    static=True, respondable=False)
            target.set_position(g)

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
        self.blue_cube = Shape('block2')
        self.green_cube = Shape('block3')
        self.yellow_cube = Shape('block4')
        self.blocks = [self.red_cube, self.blue_cube, self.green_cube, self.yellow_cube]
        self.goal = state[23:35]
        self.done = False

        self.red_cube.set_position(state[11:14])
        self.blue_cube.set_position(state[14:17])
        self.green_cube.set_position(state[17:20])
        self.yellow_cube.set_position(state[20:23])
        self.arm.set_joint_positions(state[3:10])
        self.pr.step()

        command = 1.0 if state[10] > 0.6 else 0.0

        #试试这样重置可以不
        while not self.gripper.actuate(command, 0.4):
            #一直和木块抓
            self.red_cube.set_position(state[11:14])
            self.blue_cube.set_position(state[14:17])
            self.green_cube.set_position(state[17:20])
            self.yellow_cube.set_position(state[20:23])
            if command == 0:
                for cube in self.blocks:
                    self.gripper.grasp(cube)
            elif command == 1:
                self.gripper.release()
            self.pr.step()
        
        # 设置目标点位显示
        tmp_g = self.goal.reshape(4, 3)
        for g in tmp_g:
            target = Shape.create(type=PrimitiveShape.SPHERE,
                    size=[0.02, 0.02, 0.02],
                    color=[1.0, 0.1, 0.1],
                    static=True, respondable=False)
            target.set_position(g)

        return self._get_state()

    def step(self, action):
        #要这么写才能防止pr多step
        command = 1.0 if action[-1] > 0.6 else 0.0
        self.gripper.actuate(command, 0.4)
        self.arm.set_joint_target_velocities(action[0:7])  # Execute action on arm
        self.pr.step()
        # self.arm.set_joint_target_velocities([0, 0, 0, 0, 0, 0, 0])

        if command == 0:
            for cube in self.blocks:
                self.gripper.grasp(cube)
        
        elif command == 1:
            self.gripper.release()

        current_state = self._get_state()
        cube_pos = current_state[0][11:23].reshape(4,3)
        goal_pos = self.goal.reshape(4,3)
        r = self._compute_reward(cube_pos, goal_pos)
        
        if r == 4:
            self.done = True

        return current_state, r, self.done

    def _generate_goal(self):
        #自动生成目标点位置
        base_goal = [self.goal_range_x[0] + (self.goal_range_x[1] - self.goal_range_x[0]) * np.random.random(), self.goal_range_y[0] + (self.goal_range_y[1] - self.goal_range_y[0]) * np.random.random()]
        index = np.array([0, 1, 2, 3])
        np.random.shuffle(index)    #注意shuffle是在原地生成的

        red_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[0]])
        blue_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[1]])
        green_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[2]])
        yellow_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[3]])
        
        return np.concatenate((red_goal, blue_goal, green_goal, yellow_goal))

    def _compute_reward(self, cube_pos, goal_pos):
        r = 0
        for num in range(4):
            dist = np.linalg.norm(cube_pos[num] - goal_pos[num])
            if dist < 0.04:
                r += 1
        
        return r

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Actor(nn.Module):
    def __init__(self, state_shape, action_shape, max_action):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256 ,action_shape),
            nn.Tanh()
        )
        self._max = max_action
        self.apply(self._weights_init)

    def forward(self, s, **kwargs):
        logits = self.model(s)
        logits = self._max * logits
        return logits

    @staticmethod
    def _weights_init(m):   #正交初始化
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)


class Critic(nn.Module):
    def __init__(self, state_shape, action_shape=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape + action_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256 ,1),
        )
        self.apply(self._weights_init)

    def forward(self, s, a=None):
        logits = self.model(torch.cat([s, a], dim=1))
        return logits
    

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)


class ReplayBuffer:
    '''
        这种reply buffer适合做her，因为从self.buffers的结构可以看出来，它是三维的：
            第一维是episode，表示这是第几个episode的数据
            第二维是一个episode的长度，表示一个这个环境的episode有多长
            第三维才是数据
    '''
    def __init__(self, env_params, buffer_size, sample_func=None):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        if self.sample_func != None:
            transitions = self.sample_func(temp_buffers, batch_size)
        else:   #普通的sample batch
            pass
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx


class HERSampler:
    '''
        Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    '''
    def __init__(self, replay_k, reward_func=None, replay_strategy='future'):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        '''
            这个her sample很好的代码
        '''
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = self.reward_func(transitions['ag_next'], transitions['g'])
        # transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g']), 1)
        # transitions['r'][np.isnan(transitions['r'])] = 0.
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions


def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    
    env = Env(scene_file='single.ttt', render=False)

    # 没有info
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done = env.step(data)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        
        elif cmd == 'reset_from_demo':
            ob = env.reset_from_demo(data)
            worker_end.send(ob)

        elif cmd == 'close':
            env.shutdown()
            worker_end.close()
            break
        # elif cmd == 'get_spaces':
        #     worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class ParallelEnv:
    def __init__(self, processes_number):
        '''
        multiprocessing env for pyrep

        :param n_train_processes:
        :param env: 自定义输入的环境是啥

        reference:
        https://github.com/seungeunrho/minimalRL/blob/master/a2c.py
        https://github.com/stepjam/PyRep
        
        Warning:使用此方法必须在程序最后加上env.close()!!!!!!!!!!!!!!!!!!!!!
        '''
        self.process_num = processes_number
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.process_num)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

    def reset_from_demo(self, demo_states):
        for master_end, state in zip(self.master_ends, demo_states):
            master_end.send(('reset_from_demo', state))

        return np.stack([master_end.recv() for master_end in self.master_ends])

    def render(self):   #不能render
        raise NotImplementedError


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')

def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, global_grads, mode='grads')

# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])

def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
    
    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)



