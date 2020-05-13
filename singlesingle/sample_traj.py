'''
    4/27：
        1. 加入gripper的vel，失败；变成获取张开角度直接设置，成功
        2. 加入joint 的 angle 作为state
'''

import numpy as np
import copy
import random
import math
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
import pandas as pd


# SCENE_FILE = 'single.ttt'
SCENE_FILE = './singlesingle/singlesingle.ttt'
goal_range_x = [0.5, 0.7]
goal_range_y = [-0.4, 0.4]
EPISODES = 100

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

def generate_goal():
    #自动生成目标点位置，按照red blue green yellow顺序返回
    base_goal = [goal_range_x[0] + (goal_range_x[1] - goal_range_x[0]) * np.random.random(), goal_range_y[0] + (goal_range_y[1] - goal_range_y[0]) * np.random.random()]
    index = np.array([0, 1, 2, 3])
    np.random.shuffle(index)    #注意shuffle是在原地生成的

    red_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[0]])
    blue_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[1]])
    green_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[2]])
    yellow_goal = np.array([base_goal[0], base_goal[1], 0.76999867 + 0.04 * index[3]])
    
    return np.vstack((red_goal, blue_goal, green_goal, yellow_goal))

def get_state():
    proprioceptive_state = agent.get_tip().get_position()   #3
    proprioceptive_state = np.concatenate((proprioceptive_state, agent.get_joint_positions()))  #7
    proprioceptive_state = np.concatenate((proprioceptive_state, gripper.get_open_amount()))[0:-1]  #1
    proprioceptive_state[-1] = 1.0 if proprioceptive_state[-1] > 0.6 else 0.0

    return np.concatenate((proprioceptive_state, #11
                            red_cube.get_position(),    #3
                            blue_cube.get_position(),   #3
                            green_cube.get_position(),  #3
                            yellow_cube.get_position(), #3
                            goals.reshape(1, -1)[0])).reshape(1, -1)    #12

def compute_reward(cube_pos, goal_pos):
        r = 0
        for num in range(4):
            dist = np.linalg.norm(cube_pos[num] - goal_pos[num])
            if dist < 0.04:
                r += 1
        
        return r

def panda_grasp(command):
    if command == 0:
        while not gripper.actuate(0.1, 0.4):
            pr.step()
    elif command == 1:
        while not gripper.actuate(0.9, 0.4):
            pr.step()

def pick_and_place(cube, goal):
    global tmp_es, tmp_ea
    target = Shape.create(type=PrimitiveShape.SPHERE,
                size=[0.02, 0.02, 0.02],
                color=[1.0, 0.1, 0.1],
                static=True, respondable=False)
    target.set_position(goal)

    pos = cube.get_position()
    ori = agent.get_tip().get_orientation()

    path = agent.get_path(pos, euler = ori)
    done = False
    while not done:
        done = path.step()
        pr.step()
        tmp_es = np.concatenate((tmp_es, get_state()))
        tmp_ea = np.concatenate((tmp_ea, np.concatenate((agent.get_joint_velocities(), [tmp_es[-1][10]])).reshape(1, -1)))

    panda_grasp(0)
    gripper.grasp(cube)

    new_goal = goal.copy()
    new_goal[2] = goal[2] + 0.1
    path = agent.get_path(new_goal, euler = ori, ignore_collisions=True)
    done = False
    while not done:
        done = path.step()
        pr.step()
        tmp_es = np.concatenate((tmp_es, get_state()))
        tmp_ea = np.concatenate((tmp_ea, np.concatenate((agent.get_joint_velocities(), [tmp_es[-1][10]])).reshape(1, -1)))

    goal[2] += 0.04
    path = agent.get_path(goal, euler = ori, ignore_collisions=True)
    done = False
    while not done:
        done = path.step()
        pr.step()
        tmp_es = np.concatenate((tmp_es, get_state()))
        tmp_ea = np.concatenate((tmp_ea, np.concatenate((agent.get_joint_velocities(), [tmp_es[-1][10]])).reshape(1, -1)))
    
    panda_grasp(1)
    gripper.release()

    goal[2] += 0.08
    path = agent.get_path(goal, euler = ori, ignore_collisions=True)
    done = False
    while not done:
        done = path.step()
        pr.step()
        tmp_es = np.concatenate((tmp_es, get_state()))
        tmp_ea = np.concatenate((tmp_ea, np.concatenate((agent.get_joint_velocities(), [tmp_es[-1][10]])).reshape(1, -1)))

    goal[2] -= 0.12 #把上面的拉高操作改回来

    # 返回初始位置
    # path = agent.get_path(initial_pos, euler = ori, ignore_collisions=True)
    # done = False
    # while not done:
    #     done = path.step()
    #     pr.step()
    
    print('pick and place ok!')

if __name__ == '__main__':
    i_ep = 0
    ep_index = []   #第几条轨迹
    reset_index_tmp = []
    reset_i = 0
    reset_index = []    #回放的编号，每抓成功一个cube一次就是一个新的reset节点
    expert_states = []
    expert_actions = []

    while(i_ep < EPISODES):
        print('episode: ', i_ep)
        #-------------------------------reset-----------------------------------
        pr.stop()
        pr.start()
        agent = Panda(0)
        print(agent.get_joint_upper_velocity_limits())
        gripper = PandaGripper(0)
        red_cube = Shape('block1')  #方块的边长为4cm
        blue_cube = Shape('block2')
        green_cube = Shape('block3')
        yellow_cube = Shape('block4')
        initial_pos = agent.get_tip().get_position()
        ori = agent.get_tip().get_orientation()
        goals = generate_goal()
        blocks = [red_cube, blue_cube, green_cube, yellow_cube]
        indexes = np.argsort(goals[:, 2])

        red_cube.set_position([0.5 + 0.1 * np.random.random(), -0.4 + 0.4 * np.random.random(), 0.76999867])
        blue_cube.set_position([0.5 + 0.1 * np.random.random(), 0.0 + 0.4 * np.random.random(), 0.76999867])
        green_cube.set_position([0.6 + 0.1 * np.random.random(), -0.4 + 0.4 * np.random.random(), 0.76999867])
        yellow_cube.set_position([0.6 + 0.1 * np.random.random(), 0.0 + 0.4 * np.random.random(), 0.76999867])
        pr.step()
        #-------------------------------reset finished------------------------------
        
        tmp_es = get_state()
        # print(tmp_es)
        tmp_ea = np.concatenate((agent.get_joint_velocities(), [tmp_es[-1][10]])).reshape(1, -1)
        # print(tmp_ea)
        reset_index_tmp = []
        
        old_r = 0

        try:
            for index in indexes:
                cube, goal = blocks[index], goals[index]
                pick_and_place(cube, goal)

                cube_pos = get_state()[0][11:23].reshape(4,3)
                goal_pos = goals.reshape(4,3)
                r = compute_reward(cube_pos, goal_pos)

                if r != old_r:
                    reset_index_tmp.append(tmp_ea.shape[0]) #把下标记住
                    old_r = r




        except ConfigurationPathError as e:
            print('Could not find path')
            continue

        cube_pos = get_state()[0][11:23].reshape(4,3)
        goal_pos = goals.reshape(4,3)
        r = compute_reward(cube_pos, goal_pos)
        
        if r == 4:
            print('success!')
            for _ in range(tmp_ea.shape[0]):
                ep_index.append(i_ep)

            #把存在reset_index_tmp中的下标位置拿出来，分割index
            for i in range(4):
                if i == 0:
                    nums = reset_index_tmp[i]
                elif i > 0:
                    nums = reset_index_tmp[i] - reset_index_tmp[i-1]
                for _ in range(nums):
                    reset_index.append(reset_i)
                reset_i += 1

            if i_ep == 0:
                expert_states = tmp_es
                expert_actions = tmp_ea
            else:
                expert_states = np.concatenate((expert_states, tmp_es))
                expert_actions = np.concatenate((expert_actions, tmp_ea))
            
            i_ep += 1

        else:
            print('fail!')

    print(expert_states.shape, expert_actions.shape)
    assert len(ep_index) == len(reset_index)
    ep_index = np.array(ep_index).reshape(-1, 1)
    reset_index = np.array(reset_index).reshape(-1, 1)
    expert_traj = np.concatenate((ep_index, reset_index, expert_states, expert_actions), axis=1)
    print(expert_traj.shape)

    data = pd.DataFrame(expert_traj)
    data.to_csv('./{}_expert_trajectorys.csv'.format('single_panda_cube'))
    #为了给BC和GAIL不同的专家轨迹
    # data.to_csv('./{}_expert_trajectorys_for_BC.csv'.format('jaco_reach_target'))

    pr.stop()
    pr.shutdown()

