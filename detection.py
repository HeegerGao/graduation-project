'''
    4/18: 
        发现各种队列堵塞，改串行的吧
'''
import os
import sys
import numpy as np
import time
import cv2
cv2.__version__
import matplotlib.pyplot as plt
import random
from multiprocessing import Process, Queue
import signal
import math
from pyrep import PyRep
from pyrep.robots.arms.jaco import Jaco
from pyrep.robots.end_effectors.jaco_gripper import JacoGripper
from pyrep.robots.mobiles.youbot import YouBot
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.joint import Joint


# ~~~~~~~~~~~~~~~~~~~~COPPELIASIM~~~~~~~~~~~~~~~~~~~
# -----Setting-----------
SCENE_FILE = 'detection.ttt'
# -----Start-------------
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
print('\033[96m=====================\033[0m')
# -----Objects----------- 
Jaco = Jaco()
Jaco_gripper= JacoGripper(0)
Jaco_gripper_joint = Joint('Jaco_joint6')
agent = Jaco
youbot = YouBot()
vision_sensor = VisionSensor('Vision_sensor')
vision_sensor_Jaco = VisionSensor('Vision_sensor_Jaco')


# ~~~~~~~~~~~~~~~~~~~~MULTI PROCESSING~~~~~~~~~~~~~~
# picture_queue = Queue(maxsize=1)
# youbot_queue = Queue(maxsize=1)
# exit_queue = Queue(maxsize=1)

def youBot_move(command):
    if command == -1:
        empty_list = [0, -10, 0]
    elif command == 1:
        empty_list = [0, 10, 0]
    elif command == 0:
        empty_list = [-10, 0, 0]
    
    youbot.set_base_angular_velocites(empty_list)
    for _ in range(3):
        pr.step()
    youbot.set_base_angular_velocites([0, 0, 0])
    pr.step()



plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
plt.subplot(2,2,1), plt.title('youbot rgb')
plt.subplot(2,2,2), plt.title('jaco rgb')
plt.subplot(2,2,3), plt.title('youbot depth')
plt.subplot(2,2,4), plt.title('jaco depth')

# 设置jaco关节角度，给予一个合适的初始化姿态：朝下看
starting_joint_positions = agent.get_joint_positions()
starting_joint_positions[2] = math.radians(110)
starting_joint_positions[3] = math.radians(-90)
agent.set_joint_positions(starting_joint_positions)
pr.step()

while True:
    print('ok')
    youbot_image = vision_sensor.capture_rgb()
    youbot_image_depth =  vision_sensor.capture_depth(in_meters=True)
    jaco_image = vision_sensor_Jaco.capture_rgb()
    jaco_depth = vision_sensor_Jaco.capture_depth(in_meters=True)

    plt.clf()
    # plt.subplot(2,2,1), plt.imshow(youbot_image), plt.axis('off')
    # plt.subplot(2,2,2), plt.imshow(jaco_image), plt.axis('off')
    
    youbot_image = np.array(youbot_image * 255, dtype=np.uint8) #变成bgr
    # youbot_image_depth = np.array(map_to_show[1] * 255, dtype=np.uint8)   #深度图
    youbot_image_depth = np.array(youbot_image_depth, dtype=np.float32)   #深度图
    # print(youbot_image_depth)
    youbot_image_hsv = cv2.cvtColor(youbot_image, cv2.COLOR_RGB2HSV)
    plt.subplot(2,2,3), plt.imshow(youbot_image_hsv), plt.axis('off')

    # Threshold for blue
    lower_blue = np.array([100,43,46])
    upper_blue = np.array([124,255,255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(youbot_image_hsv, lower_blue, upper_blue)    #白色是255
    plt.subplot(2,2,2), plt.imshow(mask, cmap ='gray'), plt.axis('off') 
    # print(np.max(mask))
    rx, ry = vision_sensor.get_resolution()
    # print(mask==255)
    # print(np.any(mask==255))
    # print(np.where(mask==255))
    if np.any(mask==255):   #看到了方块
        print('have block')
        object_x = np.mean(np.where(mask==255)[0]).astype(np.uint8)
        object_y = np.mean(np.where(mask==255)[1]).astype(np.uint8) #注意y才是横向的坐标

        if object_y <= rx * 0.45:  #在偏左的位置
            youBot_move(1)
        elif object_y >= rx * 0.55:  #偏右
            youBot_move(-1)
        else:   #往前走
            youBot_move(0)
        

    plt.subplot(2,2,1), plt.imshow(youbot_image), plt.axis('off')
    # print(youbot_image_depth)
    # cv2.imwrite('test.png', youbot_image_depth([np.where(mask==255)[0], np.where(mask==255)[1]]))
    # if mask.ndim == 1:
    #显示灰度图要特别设置一下
    # plt.subplot(2,2,2), plt.imshow(mask, cmap ='gray'), plt.axis('off') 
    plt.draw()
    plt.pause(0.0000000001)




















# def show_img():
#     while not picture_queue.empty():
#         print('image out 1')
#         picture_queue.get(timeout=0.001)
#     picture_queue.put([vision_sensor.capture_rgb(), vision_sensor.capture_depth(in_meters=True), vision_sensor_Jaco.capture_rgb(), vision_sensor_Jaco.capture_depth(in_meters=True)], timeout=0.001)
#     print('image in 1')


# def youBot_move(command):
#     if command == -1:
#         empty_list = [0, -10, 0]
#     elif command == 1:
#         empty_list = [0, 10, 0]
#     elif command == 0:
#         empty_list = [-10, 0, 0]
    
#     youbot.set_base_angular_velocites(empty_list)
#     for _ in range(2):
#         # print('pr go step')
#         pr.step()
#         # print('pr done step')
#         show_img()
#         # print('image done show')
#     youbot.set_base_angular_velocites([0, 0, 0])
#     # print('pr go step')
#     pr.step()
#     # print('pr done step')
#     show_img()
#     # print('image done show')

# def sigint_handler(signum, frame):
#     print ('\n\033[96mCTRL+C Detected\033[0m')
#     print('\033[91mhandling\033[0m')
#     exit_queue.put('end', timeout=0.001)
#     print('\033[91mwaiting\033[0m')
#     p.join()
#     time.sleep(0.5)
#     plt.close()
#     pr.stop()  # Stop the simulation
#     pr.shutdown()
#     cv2.destroyAllWindows()
#     print('\033[91mfinished\033[0m')
#     sys.exit(0)

# def run_proc(picture_queue):
#     print('child process started')
#     while True:
#         if not(exit_queue.empty()):
#             exit_queue.get(timeout=0.001)
#             break
        
#         elif not picture_queue.empty():
#             map_to_show = picture_queue.get(timeout=0.001)
#             print('image out 1')
#             # youbot_image = vision_sensor.capture_rgb()
#             # youbot_image_depth =  vision_sensor.capture_depth(in_meters=True)
#             # jaco_image = vision_sensor_Jaco.capture_rgb()
#             # jaco_depth = vision_sensor_Jaco.capture_depth(in_meters=True)

#             youbot_image = map_to_show[0]
#             youbot_image_depth = map_to_show[1]
#             jaco_image = map_to_show[2]
#             jaco_depth = map_to_show[3]

#             plt.clf()
#             # plt.subplot(2,2,1), plt.imshow(youbot_image), plt.axis('off')
#             # plt.subplot(2,2,2), plt.imshow(jaco_image), plt.axis('off')
            
#             youbot_image = np.array(youbot_image * 255, dtype=np.uint8) #变成bgr
#             # youbot_image_depth = np.array(map_to_show[1] * 255, dtype=np.uint8)   #深度图
#             youbot_image_depth = np.array(youbot_image_depth, dtype=np.float32)   #深度图
#             # print(youbot_image_depth)
#             youbot_image_hsv = cv2.cvtColor(youbot_image, cv2.COLOR_BGR2HSV)

#             # Threshold for blue
#             lower_blue = np.array([100,43,46])
#             upper_blue = np.array([124,255,255])

#             # Threshold the HSV image to get only green colors
#             mask = cv2.inRange(youbot_image_hsv, lower_blue, upper_blue)    #白色是255
#             # print(np.max(mask))
#             rx, ry = vision_sensor.get_resolution()
#             # print(mask==255)
#             # print(np.any(mask==255))
#             # print(np.where(mask==255))
#             if np.any(mask==255):   #看到了方块
#                 object_x = np.mean(np.where(mask==255)[0]).astype(np.uint8)
#                 object_y = np.mean(np.where(mask==255)[1]).astype(np.uint8) #注意y才是横向的坐标
                
#                 while not youbot_queue.empty():
#                     print('com out 1')
#                     youbot_queue.get(timeout=0.001)

#                 if object_y <= rx * 0.45:  #在偏左的位置
#                     youbot_queue.put(-1, timeout=0.001)
#                 elif object_y >= rx * 0.55:  #偏右
#                     youbot_queue.put(1, timeout=0.001)
#                 else:   #往前走
#                     youbot_queue.put(0, timeout=0.001)
#                 print('com in 1')
                

#             plt.subplot(2,2,1), plt.imshow(youbot_image), plt.axis('off')
#             # print(youbot_image_depth)
#             # cv2.imwrite('test.png', youbot_image_depth([np.where(mask==255)[0], np.where(mask==255)[1]]))
#             # if mask.ndim == 1:
#             #显示灰度图要特别设置一下
#             # plt.subplot(1,2,1), plt.imshow(mask, cmap ='gray'), plt.axis('off') 
#             plt.draw()
#             plt.pause(0.00001)


# # ~~~~~~~~~~~~~~~~~~~~SIMULATION~~~~~~~~~~~~~~~~~~~~
# # 设置jaco关节角度，给予一个合适的初始化姿态：朝下看
# starting_joint_positions = agent.get_joint_positions()
# starting_joint_positions[2] = math.radians(110)
# starting_joint_positions[3] = math.radians(-90)
# agent.set_joint_positions(starting_joint_positions)
# pr.step()
# show_img()

# p = Process(target=run_proc, args=([picture_queue]))
# p.start()
# signal.signal(signal.SIGINT, sigint_handler)
# # show_img()

# while True:
#     if not youbot_queue.empty():
#         command = youbot_queue.get(timeout=0.001)
#         print('com out 1')
#         youBot_move(command)
        
# # -----Gripper-----------
# print('\033[92mGripper \033[0m')
# print('Close')
# while not Jaco_gripper.actuate(0.1, 0.4):
#     pr.step()
#     show_img()
    
# print('Open')
# time.sleep(0.5)
# while not Jaco_gripper.actuate(0.9, 0.4):
#     pr.step()
#     show_img()

# -----Path Planning-----
# print('\033[92mPath planning \033[0m')


# pos, quat = agent.get_tip().get_position(), agent.get_tip().get_quaternion()
# print('res    ', pos)
# print('res    ', quat)
# pos[0] += 0.3
# pos[1] -= 0.1
# pos[2] -= 0.3
# path = agent.get_path(pos, euler = [0, math.radians(180), 0])
# path.visualize()
# print('Executing plan ...')
# done = False
# while not done:
#     done = path.step()
#     pr.step()
#     show_img()

# path.clear_visualization()


# # -----IK----------------
# print('\033[92mIK \033[0m')
# def rotate(delta):
#     joint_position = Jaco_gripper_joint.get_joint_position()
#     print(joint_position)
#     joint_position += delta
#     Jaco_gripper_joint.set_joint_target_position(joint_position)
#     pr.step()
#     show_img()

# def move(index, delta):
#     pos, quat = agent.get_tip().get_position(), agent.get_tip().get_quaternion()
#     pos[index] += delta
#     new_joint_angles = agent.solve_ik(pos, quaternion=quat)
#     agent.set_joint_target_positions(new_joint_angles)
#     pr.step()
#     show_img()

# print('Start')
# for _ in range(400):
#     rotate(-0.01)
# print('End')

# for _ in range(60):
#     move(2, -0.01)
#     # time.sleep(0.02)
# for _ in range(10):
#     move(1, +0.01)
#     # time.sleep(0.02)
# for _ in range(10):
#     move(1, -0.01)
#     # time.sleep(0.02)
# for _ in range(10):
#     move(2, +0.01)
#     # time.sleep(0.02)
# for _ in range(30):
#     move(0, +0.01)
#     # time.sleep(0.02)
# for _ in range(40):
#     move(0, -0.01)
#     # time.sleep(0.02)

# -----youBot path-------
# print('\033[92myoubot path \033[0m')
# starting_pose = youbot.get_2d_pose()
# youbot.set_2d_pose(starting_pose)
# pos = [starting_pose[0]+0.5, starting_pose[1], 0.1]
# path_youbot = youbot.get_linear_path(position=pos, angle=0)
# path_youbot.visualize()
# done = False
# while not done:
#     done = path_youbot.step()
#     pr.step()
#     show_img()

# path_youbot.clear_visualization()
# time.sleep(0.7)

# -----youBot velocity---
# print('\033[92myoubot velocity \033[0m')
# youbot.set_base_angular_velocites([1, 0, 0])
# for _ in range(50):
#     pr.step()
#     show_img()

# youbot.set_base_angular_velocites([-1, 0, 0])
# for _ in range(100):
#     pr.step()
#     show_img()

# youbot.set_base_angular_velocites([0, 1, 0])
# for _ in range(100):
#     pr.step()
#     show_img()
# time.sleep(0.5)

# youbot.set_base_angular_velocites([0, 0, 1])
# for _ in range(100):
#     pr.step()
#     show_img()

# youbot.set_base_angular_velocites([0, -1, 0])
# for _ in range(80):
#     pr.step()
#     show_img()


# print('\033[96m=====================\033[0m')
# print('Exit')
# pr.stop()  # Stop the simulation
# pr.shutdown()  # Close the application

# print('Closing subprocess')
# picture_queue.put('end', timeout=0.001)
# plt.close()
# cv2.destroyAllWindows()
# p.join()