import random
import time
from typing import Dict
from stable_baselines3 import DQN,HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import math
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import tensorflow as tf
import os
from stable_baselines3.common.callbacks import CheckpointCallback

# tensorboard --logdir=/home/nikita/manipulator_ws/src/my_robot_description/logs/logs_2d_manipulator_GoalEnv/manipulator_DQN_HER/manipulator_DQN_HER_9


class CustomEnv(gym.GoalEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,

        reward_type="sparse",
        render_mode=None,
        start_graph = 3950000


        ):
        super(CustomEnv, self).__init__()

        self.start_graph = start_graph
        self.render_mode = render_mode
        self.reward_type = reward_type

        self.reward = 0

        self.x0 = 0.5  # Начальное положение конца первого звена
        self.y0 = -0.5

        self.x00 = 1                # Начальное положение конца второго звена
        self.y00 = -1

        self.x_goal = 6
        self.y_goal = 0            # В какую точку должен попасть конец второго звена
        self.goal=[self.x_goal,self.y_goal]

        self.distance_threshold = 0.1


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



        for i in range(-160, 165, 5):
            self.possible_angles2.append(i)
        fi=5
        self.possible_angles = [(0,0),(0,fi),(fi,0),(fi,fi),(0,-fi),(-fi,0),(-fi,-fi),(-fi,fi),(fi,-fi)]

        self.action_space = gym.spaces.Discrete(len(self.possible_angles))

        self.observation_space = gym.spaces.Dict(dict(
            observation=gym.spaces.Box(low=np.float32(np.array([-160.0, -160.0,-1,-1,-1,-1])),high= np.float32(np.array([160.0,160.0,1,1,1,1])),dtype = np.float32,shape=(6,)),
            achieved_goal=gym.spaces.Box(low=np.float32(np.array([-1.0,-1.0])),high=np.float32(np.array([1.0,1.0])), dtype=np.float32,shape=(2,)),
            desired_goal=gym.spaces.Box(low=np.float32(np.array([-1.0,-1.0])),high=np.float32(np.array([1.0,1.0])), dtype=np.float32, shape=(2,)),
        ))

        self.desired_goal = np.float32([self.x_goal,self.y_goal])



    def calc(self,angle1, angle2):
        self.angle_1 = angle1
        self.angle_2 = angle2

        self.real_angle_first_old = self.real_angle_first
        self.real_angle_first += self.angle_1
        if self.real_angle_first > 160:                  #Если угол больше или меньше максимального значения, то угол равен максимальному значению
            self.real_angle_first = 160

        if self.real_angle_first < -160:
            self.real_angle_first = -160

        self.real_angle_second_old = self.real_angle_second
        self.real_angle_second += self.angle_2
        if self.real_angle_second > 160:
            self.real_angle_second = 160

        if self.real_angle_second < -160:
            self.real_angle_second = -160


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

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, desired_goal)
        print(d)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def goal_distance(self,goal_a, goal_b):
        """
        Calculated distance between two goal poses (usually an achieved pose
        and a required pose).
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def get_obs(self):
        obs = np.float32(
            np.array([self.real_angle_first, self.real_angle_second, self.x0, self.y0, self.x00, self.y00]))
        achieved_goal = np.array([self.x00,self.y00])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def step(self,action):
        self.iteration += 1

        self.angles[0] = self.possible_angles[action][0]
        self.angles[1] = self.possible_angles[action][1]
        self.calc(self.angles[0], self.angles[1])


        info = {}

        if self.render_mode == True:
            self.render()
        # if self.iteration >= self.start_graph:
        #     self.graf()
        observation = self.get_obs()
        reward = self.compute_reward(observation["achieved_goal"],self.desired_goal,info)
        self.reward = reward
        # print(observation)
        info ={}
        done = False
        return observation, reward, done, info

    def reset(self):
        self.x0 = 0.5  # Начальное положение конца первого звена
        self.y0 = -0.5

        self.x00 = 1  # Начальное положение конца второго звена
        self.y00 = -1



        self.x_goal, self.y_goal  = random.uniform(-1,1),random.uniform(-1,1)
        self.goal[0], self.goal[1] = self.x_goal, self.y_goal
        self.desired_goal[0], self.desired_goal[1] =  self.x_goal, self.y_goal

        self.real_angle_first_old = 0
        self.real_angle_second_old = 0
        self.real_angle_first = 0
        self.real_angle_second = 0

        a1 = random.randint(0, len(self.possible_angles2) - 1)
        a2 = random.randint(0, len(self.possible_angles2) - 1)

        self.angle_1 = self.possible_angles2[a1]
        self.angle_2 = self.possible_angles2[a2]
        self.calc(self.angle_1, self.angle_2)

        self.done = False


        observation = self.get_obs()

        # print("Сброс, новое состояние,", observation)
        return observation  # reward, done, info can't be included





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
        plt.title("Manipulator and goal")
        plt.xlim([-1.2, 1.2])
        plt.ylim(-1.2, 1.2)
        plt.grid()
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')

        plt.plot(self.goal[0],self.goal[1],'-*', label = 'Goal pose',)

        # graf1_x = [0, 3, 7]
        # graf1_y = [0, 4, 1]
        # plt.plot(graf1_x, graf1_y, '-*', label ="До поворота" )
        graf2_x = [0, self.x0, self.x00]
        graf2_y = [0, self.y0, self.y00]
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 16})
        plt.plot(graf2_x, graf2_y, '-*',label = 'Current pose'
                                                 .format(self.real_angle_first_old,self.real_angle_second_old))
        plt.legend()
        plt.show()
        plt.pause(0.075)
        plt.clf()


    def close (self):
        pass

def test(model_name,env):
    env = CustomEnv()
    model = DQN.load("../models/2d_manipulator_model_GoalEnv/"+model_name+"/checkpoints/manipulator_DQN_HER_5000000_steps",env=env)
    print(model.policy.net_arch)

    print(list(model.policy.parameters()))
    obs = env.reset()
    iteration=0
    while True:
        iteration+=1
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.graf()
        if iteration == 60:
            env.reset()
            i=0



def train(model_name,num_timesteps,max_episode_len,env,MAX_EPISODE_LEN = 120):
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=0.001,  # 0.0001
        buffer_size=int(1e5),  # 1e6
        learning_starts=256,  # 2048
        batch_size=256,  # 2048
        tau=0.05,  # 1.0
        gamma=0.95,
        train_freq=(MAX_EPISODE_LEN, 'step'),
        gradient_steps=1,
        optimize_memory_usage=False,
        target_update_interval=1000,  # 10000
        exploration_fraction=0.1,  # 0.1
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        seed=None,
        device='auto',
        tensorboard_log="../logs/logs_2d_manipulator_GoalEnv/" + model_name,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            goal_selection_strategy="episode",
            n_sampled_goal=4,
            max_episode_length=MAX_EPISODE_LEN,
            online_sampling=False,
            handle_timeout_termination=True
        ),
        verbose=0,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e6),
        save_path="../models/2d_manipulator_model_GoalEnv/" + model_name + "/checkpoints/",
        name_prefix=model_name,

    )

    model.learn(
        total_timesteps=num_timesteps,
        tb_log_name=model_name,
        callback=checkpoint_callback

    )
    model.save("../models/2d_manipulator_model_GoalEnv/" + model_name)


def train_old_model(model_name,num_timesteps):

    model = DQN.load("../models/2d_manipulator_model_GoalEnv/"+model_name+"/checkpoints/manipulator_DQN_HER_5000000_steps",env=env)
    model.set_env(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e6),
        save_path="../models/2d_manipulator_model_GoalEnv/" + model_name + "/checkpoints/",
        name_prefix=model_name,

    )

    model.learn(total_timesteps=num_timesteps,
                tb_log_name=model_name,
                callback=checkpoint_callback
                )
    model.save("../models/2d_manipulator_model_GoalEnv/"+model_name)

def make_env(max_episode_len,render):
    env = CustomEnv(render_mode=render,reward_type='sparse')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)
    return  env

if __name__ == '__main__':
    max_episode_len = 200
    num_timesteps = 10_000_000
    model_name = "manipulator_DQN_HER"

    # check_env(env)
    env = make_env(max_episode_len,render = False)
    train(model_name,num_timesteps,max_episode_len,env)
    # train_old_model(model_name,num_timesteps)
    test(model_name,env)

