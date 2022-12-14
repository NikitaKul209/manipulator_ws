import random
import time
from typing import Dict
from stable_baselines3 import PPO,SAC,DQN,HerReplayBuffer,HER
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import math
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import tensorflow as tf
import os


# tensorboard --logdir=/home/nikita/manipulator_ws/src/my_robot_description/logs/logs_2d_manipulator_GoalEnv/manipulator_DQN_HER/manipulator_DQN_HER_9


class CustomEnv(gym.GoalEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
        super(CustomEnv, self).__init__()

        self.reward = 0

        self.x0 = 3  # Начальное положение конца первого звена
        self.y0 = 4

        self.x00 = 7                # Начальное положение конца второго звена
        self.y00 = 1

        self.x_goal = 6
        self.y_goal = 0            # В какую точку должен попасть конец второго звена
        self.goal=[self.x_goal,self.y_goal]

        self.delta_x = 0.3
        self.delta_y = 0.3          # Допустимая разница между реальным положением конца звена и целью

        self.angle_1 = 0
        self.angle_2 = 0            # На какой угол повернуть каждому из звеньев
        self.angles = [self.angle_1,self.angle_2]

        self.possible_angles = []   # Возможные значения углов для поворота

        self.possible_angles2 = []

        self.real_angle_first = 0.0 # Реальные положения звеньев
        self.real_angle_second = 0.0

        self.real_angle_first_old = 0
        self.real_angle_second_old = 0

        self.reset_flag = 0
        self.done = False
        self.error = 0
        self.iteration = 0
        self.time_lim = 0


        for i in range(-160, 165, 5):
            self.possible_angles2.append(i)
        fi=5
        self.possible_angles = [(0,0),(0,fi),(fi,0),(fi,fi),(0,-fi),(-fi,0),(-fi,-fi),(-fi,fi),(fi,-fi)]

        self.action_space = gym.spaces.Discrete(len(self.possible_angles))
        self.observation_space = gym.spaces.Box(low=np.float32(np.array([-160.0, -160.0,-20,-20,-20,-20])),high= np.float32(np.array([160.0,160.0,20,20,20,20])),dtype = np.float32,shape=(6,))
        self.observation_space = gym.spaces.Dict(dict(
            observation=gym.spaces.Box(low=np.float32(np.array([-160.0, -160.0,-10,-10,-10,-10])),high= np.float32(np.array([160.0,160.0,10,10,10,10])),dtype = np.float32,shape=(6,)),
            achieved_goal=gym.spaces.Box(low=np.float32(np.array([-10.0,-10.0])),high=np.float32(np.array([10.0,10.0])), dtype=np.float32,shape=(2,)),
            desired_goal=gym.spaces.Box(low=np.float32(np.array([-10.0,-10.0])),high=np.float32(np.array([20.0,10.0])), dtype=np.float32, shape=(2,)),
        ))

        self.desired_goal = np.float32(self.goal)



    def calc(self,angle1, angle2):
        self.angle_1 = angle1
        self.angle_2 = angle2

        self.real_angle_first_old = self.real_angle_first
        self.real_angle_first += self.angle_1
        if self.real_angle_first > 160:                  #Если угол больше или меньше максимального значения, то угол равен максимальному значению
            self.real_angle_first = 160
            angle1 =  160 - self.real_angle_first_old
            self.done = True
            self.error = 1
        if self.real_angle_first < -160:
            self.real_angle_first = -160
            angle1 = -160 - self.real_angle_first_old
            self.done = True
            self.error = 1

        self.real_angle_second_old = self.real_angle_second
        self.real_angle_second += self.angle_2
        if self.real_angle_second > 160:
            self.real_angle_second = 160
            angle2 = 160 - self.real_angle_second_old
            self.done = True
            self.error = 1
        if self.real_angle_second < -160:
            self.real_angle_second = -160
            angle2 = -160 - self.real_angle_second_old
            self.done = True
            self.error = 1


        angle1 = angle1 * math.pi / 180
        angle2 = angle2 * math.pi / 180

        x1 = self.x0 * math.cos(angle1) - self.y0 * math.sin(angle1)
        y1 = self.x0 * math.sin(angle1) + self.y0 * math.cos(angle1)

        self.x0 = x1
        self.y0 = y1

        x01 = self.x00 * math.cos(angle1) - self.y00 * math.sin(angle1)
        y01 = self.x00 * math.sin(angle1) + self.y00 * math.cos(angle1)

        x11 = ((x01 - x1) * math.cos(angle2) - (y01 - y1) * math.sin(angle2)) + x1
        y11 = ((x01 - x1) * math.sin(angle2) + (y01 - y1) * math.cos(angle2)) + y1

        self.x00 = x11
        self.y00 = y11

        return x11,y11,x1,y1

    def reward_calc(self):
        d = math.sqrt(math.fabs(((self.x_goal - self.x00)**2)+(self.y_goal - self.y00)**2))
        self.reward = math.exp(-d)
        if math.fabs(self.x00 - self.x_goal) < self.delta_x and math.fabs(self.y00 - self.y_goal) < self.delta_y \
                and (self.real_angle_second and self.real_angle_first) in range(-160, 165):
            self.done = True
            self.reward = 100
        if self.error == True:
            self.reward = -10
            self.error = False
            self.done = True

        return self.reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward

    def _step(self, action):

        self.angles[0] = self.possible_angles[action][0]
        self.angles[1] = self.possible_angles[action][1]
        self.calc(self.angles[0],self.angles[1])
        reward = self.reward_calc()
        done = self.done
        info ={}
        observation = np.float32(np.array([self.real_angle_first, self.real_angle_second,self.x0,self.y0,self.x00,self.y00]))
        print(
            "__________________________________________________________________________________________________________")

        print(
            'Итерация: {}\nЦель: {}\nВыбраны углы: {}\nНаграда: {}\nТекущее положение по: X {}\nТекущее положение по: Y {}\n Предыдущий угол_1:{}\nПредыдущий угол_2:{}\n Текущий угол_1: {}\nТекущий угол_2: {}'.format(
                self.iteration,
                self.goal,
                self.angles,
                self.reward,
                self.x00,
                self.y00,
                self.real_angle_first_old,
                self.real_angle_second_old,
                self.real_angle_first,
                self.real_angle_second))
        print(
            "__________________________________________________________________________________________________________")

        return observation, reward, done, info


    def step(self,action):
        self.iteration += 1
        self.time_lim +=1
        if self.time_lim >=500:

            self.done =True
        observation,reward,done, info = self._step(action)
        self.compute_reward(np.array([observation[4],observation[5]]),self.desired_goal,info)
        observation = self.transform_observation(observation)
        print(observation)
        info ={}
        return observation, reward, done, info

    def reset(self):
        self.x0 = 3  # Начальное положение конца первого звена
        self.y0 = 4

        self.x00 = 7  # Начальное положение конца второго звена
        self.y00 = 1


        # if self.iteration >= 1000000:
        #     self.desired_goal[0], self.desired_goal[1] = 6,0
        #     self.goal[0], self.goal[1] = 6,0
        #     self.x_goal, self.y_goal = 6,0
        if self.iteration >= 2000000:
            self.desired_goal[0], self.desired_goal[1] = 6,-6
            self.goal[0], self.goal[1] = 6, -6
            self.x_goal, self.y_goal = 6, -6



        # self.desired_goal[0],self.desired_goal[1] = random.randint(-9,9),random.randint(-9,9)
        # self.goal[0],self.goal[1] = self.desired_goal[0],self.desired_goal[1]
        # self.x_goal,self.y_goal = self.desired_goal[0],self.desired_goal[1]


        self.real_angle_first_old = 0
        self.real_angle_second_old = 0
        self.real_angle_first = 0
        self.real_angle_second = 0

        a1 = random.randint(0, len(self.possible_angles2) - 1)
        a2 = random.randint(0, len(self.possible_angles2) - 1)

        self.angle_1 = self.possible_angles2[a1]
        self.angle_2 = self.possible_angles2[a2]
        self.calc(self.angle_1, self.angle_2)
        print("Выполнен сброс, выбраны углы: {} и {}".format(self.real_angle_first, self.real_angle_second))
        print("Координаты точек после сброса", self.x00, self.y00)

        self.reward = 0
        self.done = False
        self.time_lim = 0

        observation = np.float32(
            np.array([self.real_angle_first, self.real_angle_second, self.x0, self.y0, self.x00, self.y00]))
        observation = self.transform_observation(observation)
        print("Сброс, новое состояние,", observation)
        return observation  # reward, done, info can't be included

    def transform_observation(self, observation) -> Dict :
        self.desired_goal = np.float32([self.x_goal,self.y_goal])
        return {
            "observation": observation,
            "achieved_goal": np.array([observation[4],observation[5]]),
            "desired_goal": self.desired_goal
        }



    def render(self, mode='human'):
        print("__________________________________________________________________________________________________________")

        print('Итерация {}\nНаграда {}\nЦель {}\nТекущее положение по X {}\nТекущее положение по Y {}\nТекущий угол_1 {}\nТекущий угол_2 {}' .format(self.iteration,self.reward,
                                                                                                      self.goal,
                                                                                                      self.x00,
                                                                                                      self.y00,
                                                                                                      self.real_angle_first,
                                                                                                      self.real_angle_second))
        print("__________________________________________________________________________________________________________")

    def graf(self):
        plt.ion()
        plt.title("График поворота")
        plt.xlim([-15, 15])
        plt.ylim(-15, 15)
        plt.grid()
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')

        plt.plot(self.goal[0],self.goal[1],'-*')
        graf1_x = [0, 3, 7]
        graf1_y = [0, 4, 1]
        plt.plot(graf1_x, graf1_y, '-*', label ="До поворота" )
        graf2_x = [0, self.x0, self.x00]
        graf2_y = [0, self.y0, self.y00]
        plt.plot(graf2_x, graf2_y, '-*', label = 'После поворота первого звена на {} и второго звена на {}.\n '
                                                 '"-" - Вращение по часовой стрелке.\n'
                                                 '"+" - Вращение против часовой стрелки '
                                                 .format(self.real_angle_first_old,self.real_angle_second_old))
        plt.legend()
        plt.show()
        plt.pause(0.2)
        plt.clf()


    def close (self):
        pass

def test(model_name):
    model = DQN.load("../models/2d_manipulator_model_GoalEnv/"+model_name,env=env)
    obs = env.reset()
    i=0
    while True:
        i+=1
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.graf()
        if i == 60:
            env.reset()
            i=0



def train(model_name,num_timesteps):


    replay_buffer_class = HerReplayBuffer
    replay_buffer_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future", max_episode_length=4,
                                online_sampling=True)


    obs = env.reset()
    model = DQN(
        "MultiInputPolicy",
        env,
        tensorboard_log="../logs/logs_2d_manipulator_GoalEnv/"+model_name,
        replay_buffer_class = replay_buffer_class,
        # Parameters for HER
        replay_buffer_kwargs = replay_buffer_kwargs,
        verbose=1,
    ).learn(total_timesteps=num_timesteps,tb_log_name = model_name)
    model.save("../models/2d_manipulator_model_GoalEnv/"+model_name)


def train_old_model(model_name,num_timesteps):

    model = DQN.load("../models/2d_manipulator_model_GoalEnv/"+model_name,env=env)
    model.set_env(env)
    model.learn(total_timesteps=num_timesteps,tb_log_name=model_name)
    model.save("../models/2d_manipulator_model_GoalEnv/"+model_name)


if __name__ == '__main__':

    num_timesteps = 4000000
    model_name = "manipulator_DQN_HER"

    env = CustomEnv()
    check_env(env)
    # train(model_name,num_timesteps)
    # train_old_model(model_name,num_timesteps)
    test(model_name)


