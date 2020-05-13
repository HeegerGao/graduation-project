'''
    1. goal在中间：两边分开放。学：能够从必须回去； 进化：到不用必须回去，同时进行等
    2. goal可以在两边随机，另一侧的机械臂把够不到的挪到能够够到的地方。学：挪；进化：挪到不同的地方，同时进行等
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


SCENE_FILE = 'double.ttt'
goal_range_x = [0.5, 0.7]
goal_range_y = [-0.4, 0.4]
EPISODES = 20

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

    red_goal = np.array([0.6, 0, 0.76999867 + 0.04 * index[0]])
    blue_goal = np.array([0.6, 0, 0.76999867 + 0.04 * index[1]])
    green_goal = np.array([0.6, 0, 0.76999867 + 0.04 * index[2]])
    yellow_goal = np.array([0.6, 0, 0.76999867 + 0.04 * index[3]])
    
    return np.vstack((red_goal, blue_goal, green_goal, yellow_goal))

def get_state():
    proprioceptive_state = arm[0].get_tip().get_position()
    proprioceptive_state = np.concatenate((proprioceptive_state, gripper[0].get_open_amount()))[0:4]
    proprioceptive_state[3] = 1.0 if proprioceptive_state[3] > 0.9 else 0.0

    return np.concatenate((proprioceptive_state, 
                            red_cube.get_position(),
                            blue_cube.get_position(),
                            green_cube.get_position(),
                            yellow_cube.get_position(),
                            goals.reshape(1, -1)[0])).reshape(1, -1)

def panda_grasp(command):
    if command == 0:
        while not gripper.actuate(0.1, 0.4):
            pr.step()
    elif command == 1:
        while not gripper.actuate(0.9, 0.4):
            pr.step()

def pick_and_place(cube, goal, arm, gripper):
    # global tmp_es, tmp_ea
    target = Shape.create(type=PrimitiveShape.SPHERE,
                size=[0.02, 0.02, 0.02],
                color=[1.0, 0.1, 0.1],
                static=True, respondable=False)
    target.set_position(goal)

    tip_pos = [arm[0].get_tip().get_position(), arm[1].get_tip().get_position()]
    cube_pos = cube.get_position()
    arm_num = 0
    if np.linalg.norm(cube_pos - tip_pos[0]) <= np.linalg.norm(cube_pos - tip_pos[1]):
        arm_num = 0
    else:
        arm_num = 1

    ori = arm[arm_num].get_tip().get_orientation()

    path = arm[arm_num].get_path(cube_pos, euler = ori)
    done = False
    while not done:
        done = path.step()
        pr.step()
        # tmp_es = np.concatenate((tmp_es, get_state()))
        # tmp_ea = np.concatenate((tmp_ea, np.array(agent.get_joint_velocities()).reshape(1, -1)))

    while not gripper[arm_num].actuate(0.1, 0.4):
        pr.step()
    gripper[arm_num].grasp(cube)

    new_goal = goal.copy()
    new_goal[2] = goal[2] + 0.1
    path = arm[arm_num].get_path(new_goal, euler = ori, ignore_collisions=True)
    done = False
    while not done:
        done = path.step()
        pr.step()
        # tmp_es = np.concatenate((tmp_es, get_state()))
        # tmp_ea = np.concatenate((tmp_ea, np.array(agent.get_joint_velocities()).reshape(1, -1)))

    goal[2] += 0.04
    path = arm[arm_num].get_path(goal, euler = ori, ignore_collisions=True)
    done = False
    while not done:
        done = path.step()
        pr.step()
        # tmp_es = np.concatenate((tmp_es, get_state()))
        # tmp_ea = np.concatenate((tmp_ea, np.array(agent.get_joint_velocities()).reshape(1, -1)))
    
    while not gripper[arm_num].actuate(0.9, 0.4):
        pr.step()
    gripper[arm_num].release()

    goal[2] += 0.08
    path = arm[arm_num].get_path(goal, euler = ori, ignore_collisions=True)
    done = False
    while not done:
        done = path.step()
        pr.step()
        # tmp_es = np.concatenate((tmp_es, get_state()))
        # tmp_ea = np.concatenate((tmp_ea, np.array(agent.get_joint_velocities()).reshape(1, -1)))

    goal[2] -= 0.12 #把上面的拉高操作改回来

    path = arm[arm_num].get_path(initial_pos[arm_num], euler = ori, ignore_collisions=True)
    done = False
    while not done:
        done = path.step()
        pr.step()
    # arm[arm_num].set_joint_target_positions(initial_joint_positions[arm_num])
    # pr.step()
    print('pick and place ok!')

if __name__ == '__main__':
    i_ep = 0
    ep_index = []
    expert_states = []
    expert_actions = []

    while(i_ep < EPISODES):
        print('episode: ', i_ep)
        #-------------------------------reset-----------------------------------
        pr.stop()
        pr.start()
        arm = [Panda(0), Panda(1)]
        gripper = [PandaGripper(0), PandaGripper(1)]

        red_cube = Shape('block1')  #方块的边长为4cm
        blue_cube = Shape('block2')
        green_cube = Shape('block3')
        yellow_cube = Shape('block4')
        initial_pos = [arm[0].get_tip().get_position(), arm[1].get_tip().get_position()]
        initial_joint_positions = [arm[0].get_joint_positions(), arm[1].get_joint_positions()]
        goals = generate_goal()
        blocks = [red_cube, blue_cube, green_cube, yellow_cube]
        indexes = np.argsort(goals[:, 2])

        red_cube.set_position([0.5 + 0.1 * np.random.random(), -0.4 + 0.4 * np.random.random(), 0.76999867])
        blue_cube.set_position([0.5 + 0.1 * np.random.random(), 0.0 + 0.4 * np.random.random(), 0.76999867])
        green_cube.set_position([0.6 + 0.1 * np.random.random(), -0.4 + 0.4 * np.random.random(), 0.76999867])
        yellow_cube.set_position([0.6 + 0.1 * np.random.random(), 0.0 + 0.4 * np.random.random(), 0.76999867])
        pr.step()
        #-------------------------------reset finished------------------------------
        
        # tmp_es = get_state()
        # tmp_ea = np.array(agent.get_joint_velocities()).reshape(1, -1)
        
        try:
            for index in indexes:
                cube, goal = blocks[index], goals[index]
                pick_and_place(cube, goal, arm, gripper)

        except ConfigurationPathError as e:
            print('Could not find path')
            continue

        cube_pos = get_state()[0][4:16].reshape(4,3)
        goal_pos = goals.reshape(4,3)
        r = 0

        for num in range(4):
            dist = np.linalg.norm(cube_pos[num] - goal_pos[num])
            if dist < 0.04:
                r += 1
        
        if r == 4:
            print('success!')
            # for _ in range(tmp_ea.shape[0]):
            #     ep_index.append(i_ep)

            # if i_ep == 0:
            #     expert_states = tmp_es
            #     expert_actions = tmp_ea
            # else:
            #     expert_states = np.concatenate((expert_states, tmp_es))
            #     expert_actions = np.concatenate((expert_actions, tmp_ea))
            
            # i_ep += 1

        else:
            print('fail!')

    # print(expert_states.shape, expert_actions.shape)
    # ep_index = np.array(ep_index).reshape(-1, 1)
    # expert_traj = np.concatenate((ep_index, expert_states, expert_actions), axis=1)
    # print(expert_traj.shape)

    # data = pd.DataFrame(expert_traj)
    # data.to_csv('./{}_expert_trajectorys.csv'.format('single_panda_cube'))
    # #为了给BC和GAIL不同的专家轨迹
    # # data.to_csv('./{}_expert_trajectorys_for_BC.csv'.format('jaco_reach_target'))

    pr.stop()
    pr.shutdown()

