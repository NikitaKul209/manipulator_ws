#!/usr/bin/env python3
import time
import math

import gym
import numpy as np
import rospy
import roslib
import sys
import rospkg
import random

from functools import partial
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Range
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import *
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.env_checker import check_env




# tensorboard --logdir=/home/nikita/manipulator_ws/src/my_robot_description/logs/logs_3d_manipulator/PPO/manipulator_PPO_1


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
        super(CustomEnv, self).__init__()

        rospy.Subscriber('my_robot/gazebo/link_states', LinkStates, self.callback_links)

        self.x_goal = -1
        self.y_goal = 1            # В какую точку должен попасть конец второго звена
        self.z_goal = 2
        self.goal=[self.x_goal,self.y_goal,self.z_goal]

        self.delta_x = 0.1
        self.delta_y = 0.1
        self.delta_z = 0.1

        self.angle1 = 0
        self.angle2 = 0
        self.angle3 = 0
        self.angles = [self.angle1,self.angle2,self.angle3]


        self.position = [1,2,3]
        self.possible_angles = []  # Возможные значения углов для поворота
        for i in range(-2, 4, 2):
            self.possible_angles.append(i)

        fi = 2


        self.commands = [(0, 0, 0), (0, 0, fi),(0, fi, 0),(0, fi, fi),(fi, 0, 0),(fi, 0, fi),(fi, fi, 0),
                         (fi, fi, fi), (0, fi, -fi),(fi, 0, -fi),(fi, -fi, 0),(fi, fi, -fi),(fi, -fi, -fi),(fi, -fi, fi),
                         (0, 0, -fi),(0, -fi, 0),(0, -fi, -fi),(-fi, 0, 0),(-fi, 0, -fi),(-fi, -fi, 0),(-fi, -fi, -fi),
                         (0, -fi, fi),(-fi, 0, fi),(-fi, fi, 0),(-fi, -fi, fi),(-fi, fi, fi),(-fi, fi, -fi),]

        self.reward = 0
        self.done = False

        self.error = 0
        self.iteration = 0
        self.episode_length = 0
        self.action_space = gym.spaces.Discrete(len(self.commands))
        self.observation_space = gym.spaces.Box(low=np.float32(np.array([-180,-90.0,-90.0,-20,-20,-20])),high= np.float32(np.array([180,90.0,90.0,20,20,20])),dtype = np.float32,shape=(6,))

    def callback_links(self,msg):
        ind = msg.name.index('my_robot::link_04')
        pos = msg.pose[ind]
        x = pos.position.x
        y = pos.position.y
        z = pos.position.z
        position = [x, y, z]
        env.position = position

    def step(self, action):

        self.iteration +=1
        self.episode_length +=1
        self.angles[0] += self.commands[action][0]
        self.angles[1] += self.commands[action][1]
        self.angles[2] += self.commands[action][2]

        if self.angles[0]>180:
            self.error = 1
            self.angles[0]=180
        if self.angles[0] < -180:
            self.error = 1
            self.angles[0] = -180

        if self.angles[1] > 90:
            self.error = 1
            self.angles[1] = 90
        if self.angles[1] < -90:
            self.error = 1
            self.angles[1] = -90

        if self.angles[2] > 90:
            self.error = 1
            self.angles[2] = 90
        if self.angles[2] < -90:
            self.error = 1
            self.angles[2] = -90

        self.move_manipulator()
        reward = self.reward_calc()
        if self.episode_length >=500:
            self.episode_length = 0
            self.done = True


        observation = np.float32(np.array([self.angles[0],self.angles[1],self.angles[2], self.position[0], self.position[1], self.position[2]]))
        done = self.done
        info = {}
        print("Итерация: {}\nТекущая длина эпизода: {}\nЦель: {}\nВыбрано действие: {}\nТекущий угол 1: {}\nТекущий угол 2: {}\nТекущий угол 3: {}\nПозиция по Х: {}\nПозиция по Y: {}\nПозиция по Z: {}\nНаграда: {}\n".format(self.iteration,self.episode_length,self.goal,self.commands[action],self.angles[0],self.angles[1],self.angles[2],self.position[0],self.position[1],self.position[2],self.reward))
        rospy.sleep(0.001)
        return observation, reward, done, info

    def reward_calc(self):

        d = math.sqrt(math.fabs(((self.goal[0] - self.position[0])**2)+(self.goal[1] - self.position[1])**2)+(self.goal[2] - self.position[2])**2)
        self.reward = math.exp(-d)

        if math.fabs(self.position[0] - self.x_goal) < self.delta_x and math.fabs(self.position[1]  - self.y_goal) < self.delta_y and math.fabs(self.position[2]  - self.z_goal) < self.delta_z \
                and (self.angles[1] and self.angles[2]) in range(-160, 165):
            self.done = 1
            self.reward =1000

        # if self.error == 1:
        #     self.done = 1
        #     self.reward = -100
        self.error = 0
        return self.reward

    def move_manipulator(self):
        # rate = rospy.Rate(50)  # 50hz
        angle_topics = ['/my_robot/joint1_position_controller/command','/my_robot/joint2_position_controller/command','/my_robot/joint3_position_controller/command']
        for i in range(0,3):
            pub_angle = rospy.Publisher(angle_topics[i], Float64, queue_size=10)
            pub_angle.publish(self.angles[i]/180*math.pi)
            # rate.sleep()

    def reset(self):
        self.done = False
        self.error = 0
        self.episode_length = 0
        a1 = random.randrange(-180, 180, 2)
        a2 = random.randrange(-90, 90, 2)
        a3 = random.randrange(-90, 90, 2)

        self.angles = [a1,a2,a3]
        self.move_manipulator()
        time.sleep(0.8)

        observation = np.float32(np.array([self.angles[0],self.angles[1],self.angles[2], self.position[0], self.position[1], self.position[2]]))
        return observation  # reward, done, info can't be included

def test(model_name,algorithm):
    if algorithm == 'PPO':
        model = PPO.load("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)
    if algorithm == 'DQN':
        model = DQN.load("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)
    obs = env.reset()
    i=0
    try:
        while not rospy.is_shutdown():
            print("Идёт тестирование")
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if rewards >= 100:
                time.sleep(3)
                env.reset()

    except rospy.ROSInterruptException:
        pass

def train(model_name, algorithm,num_timesteps):

        env.reset()
        try:
            if algorithm == 'DQN':
                model = DQN("MlpPolicy", env, device='cuda', verbose=1, learning_starts=20000,gamma=0.8,tensorboard_log="src/my_robot_description/logs/logs_3d_manipulator/DQN").learn(total_timesteps=num_timesteps, tb_log_name=model_name)
            if algorithm =="PPO":
                model = PPO('MlpPolicy', env, device='cuda',verbose=1,gamma=0.8,tensorboard_log="src/my_robot_description/logs/logs_3d_manipulator/PPO").learn(total_timesteps=num_timesteps,tb_log_name=model_name)
            print("Обучение завершено!")
            model.save("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)
        except rospy.ROSInterruptException:
            pass

def train_old_model(algorithm,model_name,num_timesteps):

    if algorithm == "DQN":
        model = DQN.load("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)
    if algorithm == "PPO":
        model = PPO.load("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)

    model.set_env(env)
    model.learn(total_timesteps=num_timesteps,tb_log_name=model_name)
    model.save("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)


if __name__ == "__main__":
    rospy.init_node("manipulator_control", anonymous=True)

    model_name = "manipulator_PPO"
    algorithm = "PPO"
    num_timesteps = 100000

    env = CustomEnv()
    env.x_goal = -1
    env.y_goal = 1
    env.z_goal = 2
    time.sleep(2)
    train(model_name,algorithm,num_timesteps)
    test(model_name,algorithm)
exit(0)


