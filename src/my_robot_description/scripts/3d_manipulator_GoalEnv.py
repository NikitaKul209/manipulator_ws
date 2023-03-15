
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
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.env_checker import check_env
# tensorboard --logdir=/home/nikita/manipulator_ws/src/my_robot_description/logs/logs_3d_manipulator/PPO/manipulator_PPO_1


class CustomEnv(gym.GoalEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
        super(CustomEnv, self).__init__()

        rospy.Subscriber('my_robot/gazebo/link_states', LinkStates, self.callback_links)

        self.x_goal = -1
        self.y_goal = 1            # В какую точку должен попасть конец второго звена
        self.z_goal = 2
        self.goal=[self.x_goal,self.y_goal,self.z_goal]


        self.distance_threshold = 0.25

        self.angle1 = 0
        self.angle2 = 0
        self.angle3 = 0
        self.angles = [self.angle1,self.angle2,self.angle3]


        self.position = [1,2,3]
        self.possible_angles = []  # Возможные значения углов для поворота
        for i in range(-2, 4, 2):
            self.possible_angles.append(i)

        fi = 5


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
        obs = self._get_obs()
        # self.observation_space = gym.spaces.Dict(
        #     dict(
        #         desired_goal=gym.spaces.Box(
        #             low=np.array([-1,-1, -1], dtype="float32"),
        #             high=np.array([1,1,1], dtype="float32"),
        #             shape=obs["achieved_goal"].shape,
        #             dtype="float32",
        #         ),
        #
        #         achieved_goal=gym.spaces.Box(
        #             low=-np.inf,
        #             high=np.inf,
        #             shape=obs["achieved_goal"].shape,
        #             dtype="float32",
        #         ),
        #
        #         observation=gym.spaces.Box(
        #             -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
        #         ),
        #     )
        # )

        self.observation_space = gym.spaces.Dict(dict(
        observation = gym.spaces.Box(low=np.float32(np.array([-180,-120.0, -120.0, -1,-1,-1])),
                                     high=np.float32(np.array([180,120.0, 120.0, 1,1,1])), dtype=np.float32, shape=(6,)),
        achieved_goal = gym.spaces.Box(low=np.float32(np.array([-1.0, -1.0,-1.0])), high=np.float32(np.array([1.0, 1.0,1.0])),
                                       dtype=np.float32, shape=(3,)),
        desired_goal = gym.spaces.Box(low=np.float32(np.array([-1.0, -1.0,-1.0])), high=np.float32(np.array([1.0, 1.0,1.0])),
                                      dtype=np.float32, shape=(3,))
        ))


    def _get_obs(self):
        obs = np.array([
            self.angles,
            self.position

        ], dtype="float32").ravel()

        achieved_goal = np.array(self.position, dtype="float32").ravel()

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def goal_distance(self,goal_a, goal_b):
        """
        Calculated distance between two goal poses (usually an achieved pose
        and a required pose).
        """

        return np.linalg.norm(goal_a - goal_b, axis=-1)



    # def _is_success(self, achieved_goal, desired_goal):
    #     d = self.goal_distance(achieved_goal, desired_goal)
    #
    #     return (d < self.distance_threshold).astype(np.float32)






    def compute_reward(self, achieved_goal, desired_goal,info):
        # Compute distance between goal and the achieved goal.

        d = (self.goal_distance(achieved_goal, desired_goal))
        goal =  -(d > self.distance_threshold).astype(np.float32)

        info ={}
        return goal




    def callback_links(self,msg):
        ind = msg.name.index('my_robot::link_04')
        pos = msg.pose[ind]
        x = pos.position.x
        y = pos.position.y
        z = pos.position.z
        position = [x, y, z]
        self.position = position

    def step(self, action):

        self.iteration +=1
        self.episode_length +=1
        self.angles[0] += self.commands[action][0]
        self.angles[1] += self.commands[action][1]
        self.angles[2] += self.commands[action][2]

        if self.angles[0]>180:
            self.angles[0]=180
        if self.angles[0] < -180:
            self.angles[0] = -180

        if self.angles[1] > 120:
            self.angles[1] = 120
        if self.angles[1] < -120:
            self.angles[1] = -120

        if self.angles[2] > 120:
            self.angles[2] = 120
        if self.angles[2] < -120:
            self.angles[2] = -120

        self.move_manipulator()
        time.sleep(0.01)
        obs = self._get_obs()
        # info = {
        #     "is_success": self._is_success(obs["achieved_goal"], self.goal),
        # }
        info = {}
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"],info ))
        print(reward)

        terminated = False
        # print("Итерация: {}\nТекущая длина эпизода: {}\nЦель: {}\nВыбрано действие: {}\nТекущий угол 1: {}\nТекущий угол 2: {}\nТекущий угол 3: {}\nПозиция по Х: {}\nПозиция по Y: {}\nПозиция по Z: {}\nНаграда: {}\n".format(self.iteration,self.episode_length,self.goal,self.commands[action],self.angles[0],self.angles[1],self.angles[2],self.position[0],self.position[1],self.position[2],self.reward))

        rospy.sleep(0.01)
        return obs, reward,terminated, info



    def move_manipulator(self):
        # rate = rospy.Rate(50)  # 50hz
        angle_topics = ['/my_robot/joint1_position_controller/command','/my_robot/joint2_position_controller/command','/my_robot/joint3_position_controller/command']
        for i in range(0,3):
            pub_angle = rospy.Publisher(angle_topics[i], Float64, queue_size=10)
            pub_angle.publish(self.angles[i]/180*math.pi)
            # rate.sleep()

    def set_goal(self,x,y,z):

        rospy.wait_for_service('/my_robot/gazebo/set_model_state')
        set_state_service = rospy.ServiceProxy('/my_robot/gazebo/set_model_state', SetModelState)
        objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
        objstate.model_state.model_name = "goal"
        objstate.model_state.pose.position.x = x
        objstate.model_state.pose.position.y = y
        objstate.model_state.pose.position.z = z
        objstate.model_state.pose.orientation.w = 0.0
        objstate.model_state.pose.orientation.x = 0
        objstate.model_state.pose.orientation.y = 0
        objstate.model_state.pose.orientation.z = 0.0
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0.0
        result = set_state_service(objstate)


    def reset(self):
        self.done = False
        self.error = 0
        self.episode_length = 0
        a1 = random.randrange(-180, 180, 5)
        a2 = random.randrange(-120, 120, 5)
        a3 = random.randrange(-120, 120, 5)

        self.angles = [a1,a2,a3]
        self.move_manipulator()
        time.sleep(0.9)
        range1 = random.uniform(0.5,1)
        range2 = random.uniform(-1,-0.5)
        # random.choice([range1,range2])
        # random.choice([random.choice(range1),random.choice(range2)])
        # self.goal[0],self.goal[1],self.goal[2] = random.uniform(-10,10)/10,random.uniform(-10,10)/10,random.uniform(5,10)/10

        self.goal[0], self.goal[1], self.goal[2] =   random.choice([range1,range2]), random.choice([range1,range2]),random.uniform(0.5,1)

        self.set_goal(self.goal[0],self.goal[1],self.goal[2])

        observation = self._get_obs()
        return observation  # reward, done, info can't be included



def test(model_name,algorithm):
    env = CustomEnv()
    if algorithm == 'DQN':
        model = DQN.load("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)
    obs = env.reset()
    i=0
    try:
        while not rospy.is_shutdown():
            print("Идёт тестирование")
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
    except rospy.ROSInterruptException:
        pass

def train(model_name, algorithm,num_timesteps):

        model_name = "manipulator_DQN_HER"
        try:
            MAX_EPISODE_LEN = 120
            env = CustomEnv()
            env = gym.wrappers.TimeLimit(env, max_episode_steps=  MAX_EPISODE_LEN )

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
                tensorboard_log="../logs/logs_3d_manipulator_GoalEnv/" + model_name,
                replay_buffer_class=HerReplayBuffer,
                # Parameters for HER
                replay_buffer_kwargs=dict(
                    goal_selection_strategy="episode",
                    n_sampled_goal=4,
                    max_episode_length=MAX_EPISODE_LEN,
                    online_sampling=True,
                    handle_timeout_termination=True
                ),
                verbose=0,
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=int(1e5),
                save_path="../models/3d_manipulator_model_GoalEnv/" + model_name + "/checkpoints/",
                name_prefix=model_name,

            )

            model.learn(
                total_timesteps=num_timesteps,
                tb_log_name=model_name,
                callback=checkpoint_callback

            )
            model.save("../models/3d_manipulator_model_GoalEnv/" + model_name)

            print("Обучение завершено!")

        except rospy.ROSInterruptException:
            pass

def train_old_model(algorithm,model_name,num_timesteps):

    if algorithm == "DQN":
        MAX_EPISODE_LEN = 120
        env = CustomEnv()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_LEN)

        model = DQN.load("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name,env=env)



    checkpoint_callback = CheckpointCallback(
                save_freq=int(1e5),
                save_path="../models/3d_manipulator_model_GoalEnv/" + model_name + "/checkpoints/",
                name_prefix=model_name,
                save_replay_buffer=True,
                save_vecnormalize=True,)

    model.learn(total_timesteps=num_timesteps,tb_log_name=model_name,callback=checkpoint_callback)
    model.save("../models/3d_manipulator_model_GoalEnv/" + model_name)


if __name__ == "__main__":
    rospy.init_node("manipulator_control", anonymous=True)

    model_name = "manipulator_DQN"
    algorithm = "DQN"
    num_timesteps = 1000000


    time.sleep(2)
    train(model_name,algorithm,num_timesteps)
    # train_old_model(algorithm,model_name,num_timesteps)
    # test(model_name,algorithm)
exit(0)

