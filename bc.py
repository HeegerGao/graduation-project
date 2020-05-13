'''
    behavior cloning in single panda stack

    Author: Heeger

    state: 3 + 7 + 1 + 12 + 12 = 35
    action: 7 + 1 = 8

    4/27：加入gripper的pos
    问题：回放的时候有时候会碰到桌子，然后就歪了
    可能解决：1. state中加入各关节pos；2. 回放到每一个方块垒好的四个阶段

    4/28：放弃bc，感觉应该直接ddpg+her+reset
        还没放弃！发现是state不一样导致的！如果用expert traj的state就可以，回去看看为啥。明天先写报告 （×）
    5/3：
        莫名其妙，一开始单个state拟合发现到一定的地方就自动不动了？然后改了改网络，改成了简单的写法，就能行了？
        -0.2619 0.7381
        另外，不能在forward里面对某一个特定位置的tensor做动作

        bc基本宣告成功了，至少算法没问题。下来做单机械臂的ddpg+her
    5/10：开始全力先模仿学习！
        1. 先把所有的文件合并一下
        2. 开始在一个单一的初始情况下进行ddpg her
        整合的结果：
            在 study/graduation project里面：
            1. bc 2. sample traj 3. utils 4. singletest 5. singleddpgher
            6. mpi utils 7. normalizor 8. test.py 9. testdemo.py 10.train.py
        
        log：根据bc的结果，如果我们只在专家的state上进行测试，是完美的，但是一旦不用
            专家的state，自己跑的时候有一点点偏差，就跑了

'''
import numpy as np
import time
from utils import TrajectoryLoader, Actor, Critic, Env
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def behavior_cloning(loader, agent):
    
    ############## Hyperparameters ##############
    max_epochs = 10000
    LR = 1e-5
    ############## Hyperparameters ##############

    policy = agent
    policy.train()
    #衰减学习率
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    criterion = nn.MSELoss()

    start_time = time.time()
    running_reward = 0
    loss = torch.Tensor([0])

    #expert trajectories
    # expert_traj = loader.sample_trajectories(loader.expert_trajectories.shape[0])
    expert_traj = loader.expert_trajectories[0:97] #只模仿第一条轨迹
    expert_traj = np.delete(expert_traj, 0, axis=1) #去掉第一列episode index
    expert_traj = np.delete(expert_traj, 0, axis=1) #去掉第二列reset index

    expert_traj = torch.from_numpy(expert_traj).float().to(device)

    start_time = time.time()
    


    for epoch in range(1, max_epochs + 1):
        running_reward = 0
        
        states = expert_traj[:,0:35]
        e_actions = expert_traj[:,35:]
        assert e_actions.shape[1] == 8

        # Running policy:
        actions = policy(states)
        # print('e_a:', e_actions)
        # print('a:', actions)
        #update
        opt.zero_grad()
        loss = criterion(actions, e_actions)
        loss.backward()
        opt.step()


    # 看着环境一起来的，注意底下的env.shutdown()也要取消注释
    # env = Env(scene_file='single.ttt', render=True)
    # for epoch in range(1, max_epochs + 1):
        
    #     # running_reward = 0
    #     # s = env.reset()
    #     s = env.reset_from_demo(expert_traj[0, 0:35].detach().cpu().numpy())
    #     # print('################################')
    #     # print(s)
    #     # env.red_cube.set_position(expert_traj[0][11:14])
    #     # env.blue_cube.set_position(expert_traj[0][14:17])
    #     # env.green_cube.set_position(expert_traj[0][17:20])
    #     # env.yellow_cube.set_position(expert_traj[0][20:23])
    #     for t_step in range(276):
    #         # time.sleep(2)
    #         states = expert_traj[t_step,0:35]
    #         states = torch.unsqueeze(states, 0)
    #         # print('states:', states)
    #         e_actions = expert_traj[t_step,35:]
    #         assert e_actions.shape[0] == 8
    #         # print('e_actions:', e_actions)
    #         actions = policy(s)
    #         # print('a:', actions)
    #         s_, r, done = env.step(actions.detach().cpu().numpy()[0])
    #         # s_, r, done = env.step(expert_traj[t_step][35:])
    #         # print(actions[0])
    #         #update
    #         opt.zero_grad()
    #         loss = criterion(actions, e_actions)
    #         loss.backward()
    #         opt.step()
    #         s = s_
            
        # logging
        time_length = int(time.time() - start_time)
        # print('#####################################################')
        print(
            'Time: {:02}:{:02}:{:02}\tEpoch {} \t  loss: {}'.format(
                time_length//3600, time_length%3600//60, time_length%60,epoch, loss.item()))

        if epoch % 50 == 0:
            torch.save(policy.state_dict(), './BC_{}.pth'.format('single_panda_block'))
    # env.shutdown()

if __name__ == '__main__':
    # loader = TrajectoryLoader('single_panda_cube_expert_trajectorys.csv')
    # env_params = params = {'obs': 11,
    #     'goal': 12,
    #     'action': 8,
    #     'action_min': -1,
    #     'action_max': 1,
    #     'max_timesteps': 100,
    # }
    # actor = Actor(env_params['obs'] + 2 * env_params['goal'], env_params['action'],
    #     env_params['action_max']).to(device)
    # actor.load_state_dict(torch.load('./BC_{}.pth'.format('single_panda_block')))

    # behavior_cloning(loader, actor)






    env_params = params = {'obs': 11,
        'goal': 3,
        'action': 8,
        'action_min': -1,
        'action_max': 1,
        'max_timesteps': 120,
    }
    actor = Actor(env_params['obs'] + 2 * env_params['goal'], env_params['action'],
        env_params['action_max']).to(device)
    actor.load_state_dict(torch.load('./singlesingle/BC_{}.pth'.format('single_panda_block')))

    behavior_cloning(loader, actor)

