#!/usr/bin/env python
import time
import math

import gym
import numpy as np
import rospy
import roslib
import sys
import rospkg
import random
import itertools
from functools import partial
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Range
from gazebo_msgs.msg import LinkStates, LinkState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from control_msgs.msg import JointControllerState
from gazebo_msgs.srv import SetModelState, SetLinkState
from gazebo_msgs.srv import *
from stable_baselines3 import PPO, DQN, HerReplayBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from typing import Optional, Tuple, Any

import signal


# tensorboard —logdir=/home/nikita/manipulator_ws/src/my_robot_description/logs/logs_3d_manipulator/PPO/manipulator_PPO_1


class CustomEnv(gym.GoalEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
        super(CustomEnv, self).__init__()

        rospy.Subscriber('gazebo/link_states', LinkStates, self.callback_links)
        # rospy.Subscriber('goal/joint5_position_controller/state', JointControllerState
        #                  , self.callback_button_pose)

        self.goal = np.zeros(3)

        self.distance_threshold = 0.2
        self.angles = np.zeros(4)
        self.position = np.zeros(6)
        self.btn_pos = 0
        self.possible_angles = []  # Возможные значения углов для поворота
        for i in range(-2, 4, 2):
            self.possible_angles.append(i)

        angle_speed = 5

        self.commands = []
        for subset in itertools.product([angle_speed, -angle_speed, 0], repeat=4):
            self.commands.append(subset)

        self.reward = 0
        self.done = False

        self.error = 0
        self.iteration = 0
        self.episode_length = 0
        self.action_space = gym.spaces.Discrete(len(self.commands))

        self.observation_space = gym.spaces.Dict(dict(
            observation=gym.spaces.Box(low=np.float32(np.array([-180, -120, -120, -90, -3, -3, -3, -3, -3, -3])),
                                       high=np.float32(np.array([180, 0, 0, 90, 3, 3, 3, 3, 3, 3])),
                                       dtype=np.float32,
                                       shape=(10,)),
            achieved_goal=gym.spaces.Box(low=np.float32(np.array([-3.0, -3.0, -3.0])),
                                         high=np.float32(np.array([3.0, 3.0, 3.0])),
                                         dtype=np.float32, shape=(3,)),
            desired_goal=gym.spaces.Box(low=np.float32(np.array([-3.0, -3.0, -3.0])),
                                        high=np.float32(np.array([3.0, 3.0, 3.0])),
                                        dtype=np.float32, shape=(3,)),
        ))

        observation = np.float32(np.array(
            [self.angles[0], self.angles[1], self.angles[2], self.angles[3], self.position[0], self.position[1],
             self.position[2],
             self.position[3], self.position[4], self.position[5]]))
        achieved_goal = np.float32(np.array([self.position[0], self.position[1], self.position[2]]))

        obs = {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
        # self.timer = rospy.Timer(rospy.Duration(0.1), self.set_link)

    # obs = self._get_obs()

    def callback_links(self, msg):
        ind1 = msg.name.index('my_robot::link_06')
        ind2 = msg.name.index('my_robot::link_05')
        pos1 = msg.pose[ind1]
        pos2 = msg.pose[ind2]
        x1 = pos1.position.x
        y1 = pos1.position.y
        z1 = pos1.position.z
        x2 = pos2.position.x
        y2 = pos2.position.y
        z2 = pos2.position.z

        position = np.array([x1, y1, z1, x2, y2, z2])
        self.position = position

    def callback_button_pose(self, msg):
        self.btn_pos = msg.process_value


    def _get_obs(self):
        obs = np.concatenate([
            self.angles,
            self.position.ravel()
        ], dtype="float32")
        achieved_goal = np.array(self.position[0:3].ravel(), dtype="float32")

        return {
            "observation": obs.copy(),
            "achieved_goal":
                achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def step(self, action):

        self.iteration += 1
        self.episode_length += 1
        self.angles[0] += self.commands[action][0]
        self.angles[1] += self.commands[action][1]
        self.angles[2] += self.commands[action][2]
        self.angles[3] += self.commands[action][3]
        self.angles[0] = np.clip(self.angles[0], -180, 180)
        self.angles[1:3] = np.clip(self.angles[1:3], -120, 0)
        self.angles[4] = np.clip(self.angles[4], -90, 90)

        self.move_manipulator()

        observation = np.float32(np.array(
            [self.angles[0], self.angles[1], self.angles[2], self.angles[3], self.position[0], self.position[1],
             self.position[2],
             self.position[3], self.position[4], self.position[5]]))
        achieved_goal = np.float32(np.array([self.position[0], self.position[1], self.position[2]]))

        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }
        # obs = self._get_obs()
        info = {}
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        done = self.done

        # print("Итерация: {}\nТекущая длина эпизода: {}\nЦель: {}\nВыбрано действие: {}\nТекущий угол 1: {}\nТекущий угол 2: {}\nТекущий угол 3: {}\nТекущий угол 3: {}\nПозиция по Х: {}\nПозиция по Y: {}\nПозиция по Z: {}\nНаграда: {}\n".format(self.iteration,self.episode_length,self.goal,self.commands[action],self.angles[0],self.angles[1],self.angles[2],self.angles[3], self.position[0],self.position[1],self.position[2],reward))
        # rospy.sleep(0.001)
        return obs, reward, done, info

    def goal_distance(self, goal_a, goal_b):
        """
        Calculated distance between two goal poses (usually an achieved pose
        and a required pose).
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = self.goal_distance(achieved_goal, desired_goal)
        if d < self.distance_threshold:
            return 1
        # return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def move_manipulator(self):
        # rate = rospy.Rate(50) # 50hz
        angle_topics = ['/my_robot/joint1_position_controller/command', '/my_robot/joint2_position_controller/command',
                        '/my_robot/joint3_position_controller/command', '/my_robot/joint4_position_controller/command']
        for i in range(0, 4):
            pub_angle = rospy.Publisher(angle_topics[i], Float64, queue_size=10)
            pub_angle.publish(self.angles[i] / 180 * math.pi)
        button_pose = rospy.Publisher("/goal/joint5_position_controller/command", Float64, queue_size=10)
        button_pose.publish(0.05)
        # rate.sleep()

    def reset(self):
        self.done = False
        self.error = 0
        self.episode_length = 0
        a1 = random.randrange(-180, 180, 5)
        a2 = random.randrange(-120, 0, 5)
        a3 = random.randrange(-120, 0, 5)
        a4 = random.randrange(-90, 90, 5)

        self.angles = [a1, a2, a3, a4]
        self.move_manipulator()
        time.sleep(0.7)

        observation = np.float32(np.array(
            [self.angles[0], self.angles[1], self.angles[2], self.angles[3], self.position[0], self.position[1],
             self.position[2],
             self.position[3], self.position[4], self.position[5]]))
        achieved_goal = np.float32(np.array([self.position[0], self.position[1], self.position[2]]))

        self.goal = self.get_new_goal()
        # self.goal = np.array([-1,-1,0.5])
        self.set_goal()
        # self.set_link()
        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal,
        }
        # obs = self._get_obs()
        return obs  # reward, done, info can't be included

    def get_new_goal(self):
        x = random.choice([random.uniform(0.2, 1.2), random.uniform(-1.2, -0.2)])
        y = random.choice([random.uniform(0.2, 1.2), random.uniform(-1.2, -0.2)])
        z = random.uniform(0.5, 2.0)
        goal = np.array([x, y, z])
        return goal

    def set_goal(self):

        dx = -self.goal[0]
        dy = -self.goal[1]

        angle = math.atan2(dy, dx)

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
        objstate.model_state.model_name = "goal"
        objstate.model_state.pose.position.x = self.goal[0]
        objstate.model_state.pose.position.y = self.goal[1]
        objstate.model_state.pose.position.z = self.goal[2]
        objstate.model_state.pose.orientation.w = math.cos((angle / 2) + math.radians(-45))
        objstate.model_state.pose.orientation.x = 0
        objstate.model_state.pose.orientation.y = 0
        objstate.model_state.pose.orientation.z = math.sin((angle / 2) + math.radians(-45))
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0.0
        result = set_state_service(objstate)

    def set_link(self, e):
        dx = -self.goal[0]
        dy = -self.goal[1]

        angle = math.atan2(dy, dx)

        rospy.wait_for_service('/gazebo/set_link_state')
        set_state_service = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        objstate = LinkState()  # Create an object of type SetModelStateRequest
        objstate.link_name = "goal::base"
        objstate.pose.position.x = self.goal[0]
        objstate.pose.position.y = self.goal[1]
        objstate.pose.position.z = self.goal[2]
        objstate.pose.orientation.w = math.cos((angle / 2) + math.radians(-45))
        objstate.pose.orientation.x = 0
        objstate.pose.orientation.y = 0
        objstate.pose.orientation.z = math.sin((angle / 2) + math.radians(-45))
        objstate.twist.linear.x = 0.0
        objstate.twist.linear.y = 0.0
        objstate.twist.linear.z = 0.0
        objstate.twist.angular.x = 0.0
        objstate.twist.angular.y = 0.0
        objstate.twist.angular.z = 0.0
        result = set_state_service(objstate)


def test(model_name, algorithm):
    if algorithm == 'DQN':
        model = DQN.load("src/my_robot_description/models/3d_manipulator_model/" + algorithm + "/" + model_name)
        obs = env.reset()
        i = 0
        try:
            while not rospy.is_shutdown():
                print("Идёт тестирование")
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                i += 1
                if i >= 200:
                    env.reset()
                    i = 0
        except rospy.ROSInterruptException:
            pass


def train(model_name, algorithm, num_timesteps):
    env.reset()
    try:

            model = PPO("MultiInputPolicy",
            env,
            tensorboard_log="src/my_robot_description/logs/logs_3d_manipulator/PPO",

        # if algorithm == 'DQN':
        #     model = DQN("MultiInputPolicy",
        #                 env,
        #                 batch_size=128,
        #                 learning_rate=0.0001,
        #                 buffer_size=int(1e7),
        #                 tensorboard_log="src/my_robot_description/logs/logs_3d_manipulator/DQN",
        #                 exploration_fraction=0.02,
        #                 learning_starts=256,
        #                 target_update_interval=1000,
        #                 device="cuda",

                        # target_update_interval=1000, # 10000
                        # tau= 0.1,
                        # replay_buffer_class=HerReplayBuffer,
                        # replay_buffer_kwargs=dict(
                        # goal_selection_strategy="episode",
                        # n_sampled_goal=16,
                        # max_episode_length=2000,
                        # online_sampling=False,
                        # handle_timeout_termination=True,
                        #
                        # ),
                        # )

                        # model = DQN(
                        # "MultiInputPolicy",
                        # env,
                        # learning_rate=0.001, # 0.0001
                        # buffer_size=int(1e6), # 1e6
                        # learning_starts=256, # 2048
                        # batch_size=256, # 2048
                        # tau=0.1, # 1.0
                        # gamma=0.99,
                        # train_freq=(2000, 'step'),
                        # gradient_steps=1,
                        # optimize_memory_usage=False,
                        # target_update_interval=1000, # 10000
                        # exploration_fraction=0.1, # 0.1
                        # exploration_initial_eps=1.0,
                        # exploration_final_eps=0.05,
                        # max_grad_norm=10,
                        # seed=None,
                        # device='auto',
                        # tensorboard_log="src/my_robot_description/logs/logs_3d_manipulator_GoalEnv/" + model_name,

                        verbose=0,
                        )

            checkpoint_callback = CheckpointCallback(
                save_freq=int(1e5),
                save_path="src/my_robot_description/models/3d_manipulator_model/" + model_name + "/checkpoints/",
                name_prefix=model_name,
            )
            model.learn(
                total_timesteps=num_timesteps,
                tb_log_name=model_name,
                callback=checkpoint_callback
            )

            print("Обучение завершено!")
            model.save("src/my_robot_description/models/3d_manipulator_model/" + algorithm + "/" + model_name)
    except rospy.ROSInterruptException:
        pass


def train_old_model(algorithm, model_name, num_timesteps):
    custom_objects = {'exploration_initial_eps': 0.05, 'batch_size': 512}
    model = DQN.load(
        "src/my_robot_description/models/3d_manipulator_model/" + model_name + "/checkpoints/manipulator_DQN_HER_3600000_steps",
        custom_objects=custom_objects)

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e5),
        save_path="src/my_robot_description/models/3d_manipulator_model/" + model_name + "/checkpoints/",
        name_prefix=model_name,
    )

    model.set_env(env)
    model.exploration_initial_eps = 0.05
    model.learn(total_timesteps=num_timesteps,
                tb_log_name=model_name,
                callback=checkpoint_callback)
    model.save("src/my_robot_description/models/3d_manipulator_model/" + algorithm + "/" + model_name)


if __name__ == "__main__":
    # print("Pytorch CUDA Version is", torch.version.cuda)
    # print("Whether CUDA is supported by our system:", torch.cuda.is_available())
    # Cuda_id = torch.cuda.current_device()
    # print("CUDA Device ID: ", torch.cuda.current_device())
    # print("Name of the current CUDA Device: ", torch.cuda.get_device_name(Cuda_id))
    time.sleep(3)
    rospy.init_node("manipulator_control", anonymous=True)

    model_name = "manipulator_DQN_HER"
    algorithm = "DQN"
    num_timesteps = 10_000_000

    env = CustomEnv()
    env.goal = [1, 1, 2]
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2000)
    env.set_goal()
    # env.set_link()
    time.sleep(2)
    train(model_name, algorithm, num_timesteps)
    # train_old_model(algorithm,model_name,num_timesteps)
    # test(model_name,algorithm)
exit(0)
