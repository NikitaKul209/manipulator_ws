#!/usr/bin/env pythons
import time
import math

import gym
import numpy as np
import rospy
import roslib
import sys
import rospkg

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
from stable_baselines3.common.env_checker import check_env


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
        super(CustomEnv, self).__init__()

        rospy.init_node("talker", anonymous=True)

        self.x_goal = -1
        self.y_goal = 1            # В какую точку должен попасть конец второго звена
        self.z_goal = 1
        self.goal=[self.x_goal,self.y_goal,self.z_goal]

        self.delta_x = 0.3
        self.delta_y = 0.3
        self.delta_z = 0.3

        self.angles = []

        self.position = []
        self.possible_angles = []  # Возможные значения углов для поворота
        for i in range(-5, 10, 5):
            self.possible_angles.append(i)

        self.reward = 0

        self.error = 0

        self.action_space = gym.spaces.MultiDiscrete([len(self.possible_angles),
                                                      len(self.possible_angles),
                                                      len(self.possible_angles)])
        self.observation_space = gym.spaces.Box(low=np.float32(np.array([-160.0, -160.0,-20,-20,-20,-20])),high= np.float32(np.array([160.0,160.0,20,20,20,20])),dtype = np.float32,shape=(6,))









    def step(self, action):
        self.angles[0] = self.possible_angles[action[0]]
        self.angles[1] = self.possible_angles[action[1]]
        self.angles[2] = self.possible_angles[action[2]]
        self.move_manipulator()
        reward = self.reward_calc()

    def reward_calc(self):
        d = math.sqrt(math.fabs(((self.goal[0] - self.position[0])**2)+(self.goal[1] - self.position[1])**2)+(self.goal[2] - self.position[2])**2)
        self.reward = math.exp(-d)
        if math.fabs(self.position[0] - self.x_goal) < self.delta_x and math.fabs(self.position[1]  - self.y_goal) < self.delta_y and math.fabs(self.position[2]  - self.z_goal) < self.delta_z \
                and (self.real_angle_second and self.real_angle_first) in range(-160, 165):
            self.done = 1
            self.reward = 100
        if self.error == True:
            self.reward = -1
            self.error = False

        return self.reward


    def move_manipulator(self):
        rate = rospy.Rate(10)  # 10hz
        angle_topics = ['/my_robot/joint1_position_controller/command','/my_robot/joint2_position_controller/command','/my_robot/joint3_position_controller/command']
        for i in range(0,3):
            pub_angle = rospy.Publisher(angle_topics[i], Float64, queue_size=10)
            pub_angle.publish(self.angles[i])
            rate.sleep()


    def subscriber(self):
        position = rospy.Subscriber('my_robot/gazebo/link_states',LinkStates,callback_links)
    def callback_links(self,msg):
        ind = msg.name.index('my_robot::link_04')
        pos = msg.pose[ind]
        x = pos.position.x
        y = pos.position.y
        z = pos.position.z
        self.position = [x,y,z]






def reset_pose():
    rospy.wait_for_service('/gazebo/reset/world')
    reset_world = rospy.ServiceProxy('/gazebo/reset_world',Empty)

#     rospy.wait_for_service('my_robot/gazebo/set_link_state')
#     set_state_service = rospy.ServiceProxy('my_robot/gazebo/set_link_state', SetLinkState)
#     objstate = SetLinkStateRequest()  # Create an object of type SetModelStateRequest
#     objstate.link_state.link_name = "link_02"
#     objstate.link_state.pose.position.x = 0.0
#     objstate.link_state.pose.position.y = 0.0
#     objstate.link_state.pose.position.z = 0.0
#     objstate.link_state.pose.orientation.w = 0
#     objstate.link_state.pose.orientation.x = 0
#     objstate.link_state.pose.orientation.y = 0
#     objstate.link_state.pose.orientation.z = 0
#     objstate.link_state.twist.linear.x = 0.0
#     objstate.link_state.twist.linear.y = 0.0
#     objstate.link_state.twist.linear.z = 0.0
#     objstate.link_state.twist.angular.x = 0.0
#     objstate.link_state.twist.angular.y = 0.0
#     objstate.link_state.twist.angular.z = 0.0
#     result = set_state_service(objstate)



def move_manipulator():
    for i in range (1,37):
        # j+=0.0872

        angles = [0,-1,0]
        # if i ==1:
        #     angles = [-3, -1, 0]
        # if i == 2:
        #     angles = [1, 1, 1]

        angle_topics = ['/my_robot/joint1_position_controller/command','/my_robot/joint2_position_controller/command','/my_robot/joint3_position_controller/command']
        for j in range(0,3):
            pub_angle = rospy.Publisher(angle_topics[j], Float64, queue_size=3)
            pub_angle.publish(angles[j])
            rospy.sleep(0.1)
        time.sleep(2)


if __name__ == "__main__":
    rospy.init_node("talker", anonymous=True)
    while not rospy.is_shutdown():
        try:
            move_manipulator()
            # time.sleep(3)
            # subscriber()
            # reset_pose()
        except rospy.ROSInterruptException:
            pass


# class Omni_Wheels_Platform():


#         def __init__(self):
#             # Показания дальномеров
#             self.dists = [0.] * 6
#             self.orient = 0
#             self.position = 0
#             self.sonar_topics = ['one_sonar','two_sonar','three_sonar','for_sonar','five_sonar','six_sonar']


#             # rospy.logerr('Orient: {}'.format(self.orient))

#             # create subscribers
#             self.sub_funcs = [partial(self.callback_range,  sonar_index=i)
#                               for i in range(len(self.sonar_topics))]
#             self.subs_range = [rospy.Subscriber('/open_base/' + s, Range, f)
#                                for s, f in zip(self.sonar_topics, self.sub_funcs)]
#             self.sub_links = rospy.Subscriber("/gazebo/link_states", LinkStates,self.callback_links)

#             self.pub_velocity = rospy.Publisher('/open_base/command', Movement, queue_size=10)

#         def callback_range(self, msg, sonar_index):
#             self.dists[sonar_index] = msg.range

#         def move_platform(self,x,y,z):
#             vel = Movement()
#             vel.wheel.v_left = x
#             vel.wheel.v_back = y
#             vel.wheel.v_right = z
#             #rospy.loginfo('x: {} y:{} z:{}'.format(x,y,z))
#             vel.movement = Movement.WHEEL
#             self.pub_velocity.publish(vel)

#         def callback_links(self,msg):
#             ind = msg.name.index('open_base::origin_link')
#             pos = msg.pose[ind]
#             x = pos.position.x
#             y = pos.position.y
#             z = pos.orientation.z
#             w = pos.orientation.w
#             self.position=[x,y]

#             self.orient = int(math.atan2(z,w)*360/math.pi)
#             if self.orient < 0:
#                 self.orient = (self.orient + 360)
#                 self.orient = int(self.orient)

#             # rospy.loginfo("Pos = {} " .format(msg.pose))
#             # rospy.loginfo("Orient = {:.3f} ".format(self.orient))
#             # time.sleep(0.2)

#         def reset_pose(self,angle):
#             self.Move_0_0_0()
#             angle = angle/180*math.pi


#             rospy.wait_for_service('/gazebo/set_model_state')
#             set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
#             objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
#             objstate.model_state.model_name = "open_base"
#             objstate.model_state.pose.position.x = 0.0
#             objstate.model_state.pose.position.y = 0.0
#             objstate.model_state.pose.position.z = 0.0
#             objstate.model_state.pose.orientation.w = math.cos(angle/2)
#             objstate.model_state.pose.orientation.x = 0
#             objstate.model_state.pose.orientation.y = 0
#             objstate.model_state.pose.orientation.z = math.sin(angle/2)
#             objstate.model_state.twist.linear.x = 0.0
#             objstate.model_state.twist.linear.y = 0.0
#             objstate.model_state.twist.linear.z = 0.0
#             objstate.model_state.twist.angular.x = 0.0
#             objstate.model_state.twist.angular.y = 0.0
#             objstate.model_state.twist.angular.z = 0.0
#             result = set_state_service(objstate)


#         def Move_0_0_0(self):
#             self.move_platform(0.0, 0.0, 0.0)
#         def Move_0_0_L(self):
#             self.move_platform(0.0, 0.0, -0.4)
#         def Move_0_0_R(self):
#             self.move_platform(0.0, 0.0, 0.4)
#         def Move_0_L_0(self):
#             self.move_platform(0.0, -0.4, 0.0)
#         def Move_0_R_0(self):
#             self.move_platform(0.0, 0.4, 0.0)
#         def Move_0_L_L(self):
#             self.move_platform(0.0, -0.4, -0.4)
#         def Move_0_R_R(self):
#             self.move_platform(0.0, 0.4, 0.4)
#         def Move_0_L_R(self):
#             self.move_platform(0.0, -0.4, 0.4)
#         def Move_0_R_L(self):
#             self.move_platform(0.0, 0.4, -0.4)
#         def Move_L_0_0(self):
#             self.move_platform(-0.4,0.0,0.0)
#         def Move_R_0_0(self):
#             self.move_platform(0.4,0.0,0.0)
#         def Move_L_0_L(self):
#             self.move_platform(-0.4,0.0,-0.4)
#         def Move_R_0_R(self):
#             self.move_platform(0.4, 0.0, 0.4)
#         def Move_L_0_R(self):
#             self.move_platform(-0.4,0.0,0.4)
#         def Move_R_0_L(self):
#             self.move_platform(0.4,0.0,-0.4)
#         def Move_L_L_0(self):
#             self.move_platform(-0.4,-0.4,0.0)
#         def Move_R_R_0(self):
#             self.move_platform(0.4,0.4,0.0)
#         def Move_L_R_0(self):
#             self.move_platform(-0.4,0.4,0.0)
#         def Move_R_L_0(self):
#             self.move_platform(0.4, -0.4, 0.0)
#         def Move_L_L_L(self):
#             self.move_platform(-0.4, -0.4, -0.4)
#         def Move_R_R_R(self):
#             self.move_platform(0.4, 0.4, 0.4)
#         def Move_L_L_R(self):
#             self.move_platform(-0.4, -0.4, 0.4)
#         def Move_R_R_L(self):
#             self.move_platform(0.4, 0.4, -0.4)
#         def Move_L_R_R(self):
#             self.move_platform(-0.4, 0.4, 0.4)
#         def Move_R_L_L(self):
#             self.move_platform(0.4, -0.4, -0.4)
#         def Move_L_R_L(self):
#             self.move_platform(-0.4, 0.4, -0.4)
#         def Move_R_L_R(self):
#             self.move_platform(0.4, -0.4, 0.4)
