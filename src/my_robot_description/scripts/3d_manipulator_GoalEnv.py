
#!/usr/bin/env python
import time
import math

import gym
import numpy
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
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import *
from stable_baselines3 import PPO,DQN
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

class CustomEnv(gym.GoalEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 reward_type = None,
                 render_mode = "human",
                 static_goal=None,
                 randomize_goal = True,
                 angle_speed=5):
        super(CustomEnv, self).__init__()

        rospy.Subscriber('my_robot/gazebo/link_states', LinkStates, self.callback_links)

        self.reward_type = reward_type
        self.render_mode = render_mode
        self.static_goal = static_goal
        self.randomized_goal = randomize_goal
        self.angle_speed = angle_speed

        self.goal=np.zeros(3)
        self.angles = np.zeros(3)
        self.position = np.zeros(6)
        self.distance_threshold = 0.25
        self.reward = 0
        self.action = 0
        self.done = False
        self.iteration = 0


        self.commands =[]
        for subset in itertools.product([angle_speed, -angle_speed, 0],repeat = 3):
            self.commands.append(subset)

        self.action_space = gym.spaces.Discrete(len(self.commands))
        self.observation_space = gym.spaces.Dict(dict(
        observation = gym.spaces.Box(
            low=np.float32(np.array([-180,-120.0, -120.0, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])),
            high=np.float32(np.array([180,120.0, 120.0, np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])),
            dtype=np.float32,
            shape=(9,)),

        achieved_goal = gym.spaces.Box(
            low=np.float32(np.array([-np.inf,-np.inf,-np.inf])),
            high=np.float32(np.array([np.inf,np.inf,np.inf])),
            dtype=np.float32,
            shape=(3,)),

        desired_goal = gym.spaces.Box(
            low=np.float32(np.array([-1.0, -1.0,-1.0])),
            high=np.float32(np.array([1.0, 1.0,1.0])),
            dtype=np.float32,
            shape=(3,))
        ))
        obs = self._get_obs()
    def _get_obs(self):
        obs = np.concatenate([
            self.angles,
            self.position
        ], dtype="float32")
        achieved_goal = np.array(self.position[0:3], dtype="float32").ravel()

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

    def compute_reward(self, achieved_goal, desired_goal,info):
        # Compute distance between goal and the achieved goal.
        d = (self.goal_distance(achieved_goal, desired_goal))
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def callback_links(self,msg):
        ind1 = msg.name.index('my_robot::link_04')
        ind2 = msg.name.index('my_robot::link_05')
        pos1 = msg.pose[ind1]
        pos2 = msg.pose[ind2]
        x1 = pos1.position.x
        y1 = pos1.position.y
        z1 = pos1.position.z
        x2 = pos2.position.x
        y2 = pos2.position.y
        z2 = pos2.position.z
        position = [x1,y1,z1,x2,y2,z2]
        self.position = position

    def step(self, action):

        self.iteration +=1
        self.action  = action
        self.angles[0] += self.commands[action][0]
        self.angles[1] += self.commands[action][1]
        self.angles[2] += self.commands[action][2]

        self.angles[0] = np.clip(self.angles[0],-180,180)
        self.angles[1:3] = np.clip(self.angles[1:3],-120,120)
        self.move_manipulator()
        obs = self._get_obs()
        info = {}
        self.reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"],info ))

        terminated = False
        if self.render_mode == True:
            self.render()
        rospy.sleep(0.01)
        return obs, self.reward,terminated, info


    def render(self,mode ="human"):
        print("Итерация: {}\nЦель: {}\nВыбрано действие: {}\nТекущий угол 1: {}\nТекущий угол 2: {}\nТекущий угол 3: "
              "{}\nПозиция по Х: {}\nПозиция по Y: {}\nПозиция по Z: {}\nНаграда: {}\n".
              format(self.iteration,
                     self.goal, self.commands[self.action], self.angles[0], self.angles[1], self.angles[2],
                     self.position[0], self.position[1], self.position[2], self.reward))

    def move_manipulator(self):
        # rate = rospy.Rate(50)  # 50hz
        angle_topics = ['/my_robot/joint1_position_controller/command','/my_robot/joint2_position_controller/command','/my_robot/joint3_position_controller/command']
        for i in range(0,3):
            pub_angle = rospy.Publisher(angle_topics[i], Float64, queue_size=10)
            pub_angle.publish(self.angles[i]/180*math.pi)
        # time.sleep(0.01)

    def set_goal(self,goal):

        rospy.wait_for_service('/my_robot/gazebo/set_model_state')
        set_state_service = rospy.ServiceProxy('/my_robot/gazebo/set_model_state', SetModelState)
        objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
        objstate.model_state.model_name = "goal"
        objstate.model_state.pose.position.x = goal[0]
        objstate.model_state.pose.position.y = goal[1]
        objstate.model_state.pose.position.z = goal[2]
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

    def get_new_goal(self):
        x = random.choice([random.uniform(0.5, 1), random.uniform(-1, -0.5)])
        y = random.choice([random.uniform(0.5, 1), random.uniform(-1, -0.5)])
        z = random.uniform(0.5, 1)
        goal = [x,y,z]
        return goal
    def reset(self):
        self.done = False
        a1 = random.randrange(-180, 180, 5)
        a2 = random.randrange(-120, 120, 5)
        a3 = random.randrange(-120, 120, 5)

        self.angles = [a1,a2,a3]
        self.move_manipulator()
        time.sleep(0.9)

        if self.randomized_goal:
            self.goal= self.get_new_goal()
        else:
            self.goal= np.array(self.static_goal)
        self.set_goal(self.goal)
        observation = self._get_obs()

        return observation  # reward, done, info can't be included

def make_env(max_episode_length,render_mode,randomized_goal,static_goal,reward_type):
    env = CustomEnv(render_mode=render_mode,randomize_goal=randomized_goal,static_goal=static_goal,reward_type=reward_type)
    if max_episode_length:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_length)
    return env


def test(model_name,algorithm):
    env = CustomEnv()
    model = DQN.load("src/my_robot_description/models/3d_manipulator_model/"+algorithm+"/"+model_name)
    obs = env.reset()
    try:
        while not rospy.is_shutdown():
            print("Идёт тестирование")
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
    except rospy.ROSInterruptException:
        pass

def train(model_name,num_timesteps,max_episode_length,render_mode,randomized_goal,static_goal,reward_type):

        env = make_env(max_episode_length,render_mode,randomized_goal,static_goal,reward_type)
        try:
            model = DQN(
                "MultiInputPolicy",
                env,
                learning_rate=0.001,  # 0.0001
                buffer_size=int(1e5),  # 1e6
                learning_starts=256,  # 2048
                batch_size=256,  # 2048
                tau=0.05,  # 1.0
                gamma=0.95,
                train_freq=(max_episode_length, 'step'),
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
                    n_sampled_goal=8,
                    max_episode_length=max_episode_length,
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

    model_name = "manipulator_DQN_HER"
    num_timesteps = 10000000
    max_episode_length = 300
    render_mode = True
    randomized_goal = False
    static_goal = (1,1,1)
    reward_type = "sparse"

    train(model_name,num_timesteps,max_episode_length,render_mode,randomized_goal,static_goal,reward_type)

    # train_old_model(algorithm,model_name,num_timesteps)
    # test(model_name,algorithm)
exit(0)

