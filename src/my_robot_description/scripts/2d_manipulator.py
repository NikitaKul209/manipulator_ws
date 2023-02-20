import random
import time
from stable_baselines3 import PPO,SAC,DQN
import math
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
# import tensorflow as tf

#tensorboard --logdir=/home/nikita/manipulator_ws/src/my_robot_description/logs/logs_2d_manipulator/DQN/manipulator_DQN_2


class CustomEnv(gym.Env):
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
        self.y_goal = 6            # В какую точку должен попасть конец второго звена
        self.goal=[self.x_goal,self.y_goal]

        self.delta_x = 0.2
        self.delta_y = 0.2          # Допустимая разница между реальным положением конца звена и целью

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
        self.done = 0
        self.error = 0


        for i in range(-160, 165, 5):
            self.possible_angles2.append(i)
        fi=5
        self.possible_angles = [(0,0),(0,fi),(fi,0),(fi,fi),(0,-fi),(-fi,0),(-fi,-fi),(-fi,fi),(fi,-fi)]

        self.action_space = gym.spaces.Discrete(len(self.possible_angles))
        self.observation_space = gym.spaces.Box(low=np.float32(np.array([-160.0, -160.0,-20,-20,-20,-20])),high= np.float32(np.array([160.0,160.0,20,20,20,20])),dtype = np.float32,shape=(6,))

    def calc(self,angle1, angle2):
        self.angle_1 = angle1
        self.angle_2 = angle2

        self.real_angle_first_old = self.real_angle_first
        self.real_angle_first += self.angle_1
        if self.real_angle_first > 160:
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
        d = math.sqrt(math.fabs(((self.goal[0] - self.x00)**2)+(self.goal[1] - self.y00)**2))
        self.reward = math.exp(-d)
        if math.fabs(self.x00 - self.x_goal) < self.delta_x and math.fabs(self.y00 - self.y_goal) < self.delta_y \
                and (self.real_angle_second and self.real_angle_first) in range(-160, 165):
            self.done = 1
            self.reward = 100
        if self.error == True:
            self.reward = -1
            self.error = False

        return self.reward



    def step(self, action):

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
            'Цель: {}\nВыбраны углы:{}\nНаграда: {}\nТекущее положение по: X {}\nТекущее положение по: Y {}\n Предыдущий угол_1:{}\nПредыдущий угол_2:{}\n Текущий угол_1: {}\nТекущий угол_2: {}'.format(
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

    def reset(self):


        self.x0 = 3  # Начальное положение конца первого звена
        self.y0 = 4

        self.x00 = 7                # Начальное положение конца второго звена
        self.y00 = 1
        self.real_angle_first_old =  0
        self.real_angle_second_old = 0
        self.real_angle_first =  0
        self.real_angle_second = 0

        a1 = random.randint(0, len(self.possible_angles2)-1)
        a2 = random.randint(0, len(self.possible_angles2)-1)

        self.angle_1 = self.possible_angles2[a1]
        self.angle_2 = self.possible_angles2[a2]
        self.calc(self.angle_1,self.angle_2)
        print("Выполнен сброс, выбраны углы: {} и {}".format(self.real_angle_first, self.real_angle_second))
        print("Координаты точек после сброса", self.x00,self.y00)

        self.reward = 0
        self.done = 0

        observation = np.float32(
            np.array([self.real_angle_first, self.real_angle_second, self.x0, self.y0, self.x00, self.y00]))
        print("Сброс, новое состояние,", observation)





        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        print("__________________________________________________________________________________________________________")

        print('Награда {}\nЦель {}\nТекущее положение по X {}\nТекущее положение по Y {}\nТекущий угол_1 {}\nТекущий угол_2 {}' .format(self.reward,
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



def test(model_name,algorithm):

    obs = env.reset()
    if algorithm == "DQN":
        model = DQN.load("../models/2d_manipulator_model/"+algorithm+"/"+model_name)
    if algorithm == "PPO":
        model = PPO.load("../models/2d_manipulator_model/"+algorithm+"/"+model_name)
    i=0
    while True:
        i+=1
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.graf()
        if i == 60:
            env.reset()
            i=0

def train(model_name,num_timesteps,algorithm):

    obs = env.reset()
    if algorithm == 'DQN':
        model = DQN("MlpPolicy", env, device='cuda', verbose=1, learning_starts=20000, tensorboard_log="../logs/logs_2d_manipulator/DQN").learn(total_timesteps=num_timesteps, tb_log_name=model_name)
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env,device='cuda',verbose=1, tensorboard_log = "../logs/logs_2d_manipulator/PPO",).learn(total_timesteps=num_timesteps, tb_log_name = model_name)
    model.save("../models/2d_manipulator_model/"+algorithm+"/"+model_name)




def train_old_model(model_name,num_timesteps,algorithm):

    if algorithm == "DQN":
        model = DQN.load("../models/2d_manipulator_model/"+algorithm+"/"+model_name)

    if algorithm == "PPO":
        model = PPO.load("../models/2d_manipulator_model/"+algorithm+"/"+model_name)

    model.set_env(env)
    model.learn(total_timesteps=num_timesteps,tb_log_name=model_name)
    model.save("../models/2d_manipulator_model/"+algorithm+"/"+model_name)


if __name__ == '__main__':

    # print(tf.config.list_physical_devices('GPU'))
    print(torch.cuda.get_device_name(0))
    CUDA = torch.cuda.is_available()
    env = CustomEnv()
    env.x_goal = 6
    env.y_goal = 6
    model_name = "manipulator_DQN"
    algorithm ="DQN"
    num_timesteps = 700000



    train(model_name,num_timesteps,algorithm)
    # train_old_model(model_name,num_timesteps,algorithm)
    test(model_name,algorithm)

