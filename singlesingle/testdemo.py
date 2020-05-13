'''
    看看训练好的模型咋样
'''

import numpy as np
import time
import sys
sys.path.append("..") 
from utils import Actor, Critic
from singlesingle import Env
import torch

iter = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_params = {'obs': 11,
    'goal': 3,
    'action': 8,
    'action_min': -1,
    'action_max': 1,
    'max_timesteps': 120,
    }

actor = Actor(env_params['obs'] + 2 * env_params['goal'], env_params['action'],
        env_params['action_max']).to(device)

actor.load_state_dict(torch.load('./ddpg_models/actor.pth'))
# actor.load_state_dict(torch.load('./BC_{}.pth'.format('single_panda_block')))
env = Env(scene_file='singlesingle.ttt', render=True)

for i in range(1, iter + 1):
    s = env.reset()
    av_r = 0
    for t_step in range(120):
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