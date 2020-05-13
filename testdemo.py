'''
    看看训练好的模型咋样
'''

import numpy as np
import time
from utils import TrajectoryLoader, Actor, Critic, Env
import torch

iter = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# demos
loader = TrajectoryLoader('single_panda_cube_expert_trajectorys.csv')
expert_traj = loader.expert_trajectories
expert_traj = np.delete(expert_traj, 0, axis=1)
#get the index of step state
reset_index = expert_traj[:, 0]
expert_traj = np.delete(expert_traj, 0, axis=1)
target_index = []   #表示reset的下标

# for i in range(reset_index.shape[0] - 1):
#     if reset_index[i] != reset_index[i+1]:
#         target_index.append(i+1)
# target_index.append(0)  #把初始状态也加上
env_params = {'obs': 11,
    'goal': 12,
    'action': 8,
    'action_min': -1,
    'action_max': 1,
    'max_timesteps': 300,
    }

actor = Actor(env_params['obs'] + 2 * env_params['goal'], env_params['action'],
        env_params['action_max']).to(device)

actor.load_state_dict(torch.load('./ddpg_models/actor.pth'))
# actor.load_state_dict(torch.load('./BC_{}.pth'.format('single_panda_block')))
env = Env(scene_file='single.ttt', render=True)

for i in range(1, iter + 1):
    # s = env.reset()
    s = env.reset_from_demo(expert_traj[0][0:35])   #bc test
    av_r = 0
    # d_state = expert_traj[target_index[np.random.randint(0, len(target_index))], 0:35] #随机选一个step的demo作为reset
    # s = env.reset_from_demo(d_state)
    for t_step in range(300):
        # s = [expert_traj[100+t_step][0:35]]
        s = torch.tensor(s, device=device, dtype=torch.float32)
        actions = actor(s)
        print(actions)
        s_, r, done = env.step(actions.detach().cpu().numpy()[0])
        s = s_

        av_r += r
        if done:
            break

    print('iter ', i, ' reward: ', np.mean(av_r))
env.shutdown()