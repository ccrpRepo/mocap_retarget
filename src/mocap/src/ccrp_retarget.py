#!/usr/bin/env python
import csv
import os
import numpy as np
import copy
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
import tf
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Int32
from rotation import *
import pygame
import transforms3d.euler as euler



def safe_convert(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value 

class Motion:
    def __init__(self, motion_path=None, retarget_fps = 120):
        self.motion_path = motion_path
        self.retarget_fps = retarget_fps
        self.frame = 0
        self.playing = False # whether is playing the motion sequence
        self.data_fps = 120

        # whether is dragging
        self.rotate_dragging = False
        self.translate_dragging = False
        # old mouse cursor position
        self.old_x = 0
        self.old_y = 0
        # global rotation
        self.global_rx = 0
        self.global_ry = 0
        # rotation matrix for camera moving
        self.rotation_R = np.eye(3)
        # rotation speed
        self.speed_rx = np.pi / 90
        self.speed_ry = np.pi / 90
        # translation speed
        self.speed_trans = 0.25
        self.speed_zoom = 0.5
        # whether the main loop should break
        self.done = False
        # default translate set manually to make sure the skeleton is in the middle
        # of the window
        # if you can't see anything in the screen, this is the first parameter you
        # need to adjust
        self.default_translate = np.array([0, -20, -100], dtype=np.float32)
        self.translate = np.copy(self.default_translate)
        pygame.init()
        self.screen_size = (480, 240)
        self.screen = pygame.display.set_mode(
        self.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
        )
        self.done = False
        ## init ros pubulisher
        rospy.init_node('ccrp_motion_rollout_node', anonymous=True)
    
        # 创建一个 Publisher，用于发布 JointState 消息
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.frameNumpub = rospy.Publisher('/frame_num', Int32, queue_size=10)
        # 定义发布频率
        self.rate = rospy.Rate(self.retarget_fps)
        
        
        self.joint_names = ['left_up_leg_joint_x', 'left_up_leg_joint_y', 'left_up_leg_joint_z', \
                            'left_leg_joint_x', 'left_leg_joint_y', 'left_leg_joint_z', \
                            'left_foot_joint_x', 'left_foot_joint_y', 'left_foot_joint_z', \
                            'left_toe_joint_x', 'left_toe_joint_y', 'left_toe_joint_z', \
                                
                            'right_up_leg_joint_x', 'right_up_leg_joint_y', 'right_up_leg_joint_z', \
                            'right_leg_joint_x', 'right_leg_joint_y', 'right_leg_joint_z', \
                            'right_foot_joint_x', 'right_foot_joint_y', 'right_foot_joint_z', \
                            'right_toe_joint_x', 'right_toe_joint_y', 'right_toe_joint_z', \
                                
                            'spine1_joint_x', 'spine1_joint_y', 'spine1_joint_z', \
                            'spine2_joint_x', 'spine2_joint_y', 'spine2_joint_z', \
                            'chest_joint_x', 'chest_joint_y', 'chest_joint_z', \
                            'neck_joint_x', 'neck_joint_y', 'neck_joint_z', \
                            'head_joint_x', 'head_joint_y', 'head_joint_z', \
                                
                            'left_shouder_joint_x', 'left_shouder_joint_y', 'left_shouder_joint_z', \
                            'left_arm_joint_x', 'left_arm_joint_y', 'left_arm_joint_z', \
                            'left_fore_arm_joint_x', 'left_fore_arm_joint_y', 'left_fore_arm_joint_z', \
                            'left_hand_joint_x', 'left_hand_joint_y', 'left_hand_joint_z', \
                                
                            'right_shouder_joint_x', 'right_shouder_joint_y', 'right_shouder_joint_z', \
                            'right_arm_joint_x', 'right_arm_joint_y', 'right_arm_joint_z', \
                            'right_fore_arm_joint_x', 'right_fore_arm_joint_y', 'right_fore_arm_joint_z', \
                            'right_hand_joint_x', 'right_hand_joint_y', 'right_hand_joint_z'
                            ]

        
        self.joint_positions = []
        for _ in range(len(self.joint_names)):
            self.joint_positions.append(0.0)
        
        # 时间变量，用于模拟关节的运动
        self.start_time = rospy.Time.now().to_sec()
        self.br = tf.TransformBroadcaster()
        # 发布 tf 转换
        # br.sendTransform(goal_msg.pose.position,goal_msg.pose.orientation,rospy.Time.now(),"pelvis","map")
        self.t = TransformStamped()
        self.t.header.frame_id = 'world'
        self.t.header.stamp = rospy.Time(0)
        self.t.child_frame_id = 'hips'
        self.t.transform.translation.x = 0
        self.t.transform.translation.y = 0
        self.t.transform.translation.z = 0.5
        self.t.transform.rotation.w=1
        self.t.transform.rotation.x=0
        self.t.transform.rotation.y=0
        self.t.transform.rotation.z=0
        
        self.link_names = ['Root','Hips',
                           'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe',
                           'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe',
                           'Spine1', 'Spine2', 'Chest', 'Neck', 'Head',
                           'LeftShouder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                           'RightShouder', 'RightArm', 'RightForeArm', 'RightHand'
                           ]
        
        self.motions = []
    
    def process_event(self):
        """
        Handle user interface events: keydown, close, dragging.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: # reset camera
                    self.translate = self.default_translate
                    self.global_rx = 0
                    self.global_ry = 0
                elif event.key == pygame.K_SPACE:
                    self.playing = not self.playing
            elif event.type == pygame.MOUSEBUTTONDOWN: # dragging
                if event.button == 1:
                    self.rotate_dragging = True
                else:
                    self.translate_dragging = True
                    self.old_x, self.old_y = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.rotate_dragging = False
                else:
                    self.translate_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.translate_dragging:
                    # haven't figure out best way to implement this
                    pass
                elif self.rotate_dragging:
                    new_x, new_y = event.pos
                    self.global_ry -= (new_x - self.old_x) / \
                        self.screen_size[0] * np.pi
                    self.global_rx -= (new_y - self.old_y) / \
                        self.screen_size[1] * np.pi
                    self.old_x, self.old_y = new_x, new_y
        pressed = pygame.key.get_pressed()
        # rotation
        if pressed[pygame.K_DOWN]:
            self.global_rx -= self.speed_rx
        if pressed[pygame.K_UP]:
            self. global_rx += self.speed_rx
        if pressed[pygame.K_LEFT]:
            self.global_ry += self.speed_ry
        if pressed[pygame.K_RIGHT]:
            self.global_ry -= self.speed_ry
        # moving
        if pressed[pygame.K_a]:
            self.translate[0] -= self.speed_trans
        if pressed[pygame.K_d]:
            self.translate[0] += self.speed_trans
        if pressed[pygame.K_w]:
            self.translate[1] += self.speed_trans
        if pressed[pygame.K_s]:
            self.translate[1] -= self.speed_trans
        if pressed[pygame.K_q]:
            self.translate[2] += self.speed_zoom
        if pressed[pygame.K_e]:
            self.translate[2] -= self.speed_zoom
        # forward and rewind
        if pressed[pygame.K_COMMA]:
            self.frame -= 1
        if self.frame < 0:
            self.frame = len(self.motions) - 1
        if pressed[pygame.K_PERIOD]:
            self.frame += 1
        if self.frame >= len(self.motions):
            self.frame = 0
        # global rotation
        grx = euler.euler2mat(self.global_rx, 0, 0)
        gry = euler.euler2mat(0, self.global_ry, 0)
        self.rotation_R = grx.dot(gry)
        
    def draw(self):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  
        joint_state_msg.name = self.joint_names             
        joint_state_msg.position = self.joint_positions      
        joint_state_msg.velocity = []                  
        joint_state_msg.effort = []                    
        
        self.joint_pub.publish(joint_state_msg)
        self.frameNumpub.publish(self.frame)
        self.br.sendTransformMessage(self.t)
        
    def csv_parser(self):
        with open(self.motion_path, 'r', encoding='utf-8') as file:
            
            csv_reader = csv.reader(file)
            signle_link_pose ={'name': ' ',
                               'euler' : np.zeros(3), # xyzw
                                'pos' : np.zeros(3)
                                }
            signle_frame ={'frame': ' ',
                           'time': 0.0,
                            'link_poses': []
                          }
            for idx, row in enumerate(csv_reader):
                # print(row)  # 每行是一个列表，按列顺序存储数据
                if(idx == 0):
                    self.data_fps = row[3]
                elif(idx >= 7):
                    row = [safe_convert(item) for item in row]
                    link_poses = []
                    for link_id, link_name in enumerate(self.link_names):
                        signle_link_pose['name'] = link_name
                        signle_link_pose['euler'] = np.array(row[link_id * 6 + 2 : link_id * 6 + 5])
                        signle_link_pose['pos'] = np.array(row[link_id * 6 + 5 : link_id * 6 + 8]) / 1000.0
                        link_poses.append(copy.deepcopy(signle_link_pose))
                        
                    signle_frame['frame'] = row[0]
                    signle_frame['time'] = row[1]
                    signle_frame['link_poses'] = copy.deepcopy(link_poses)
                    self.motions.append(copy.deepcopy(signle_frame))
        pass
    
    def motion_rollout(self):
        
        while not self.done:
            self.process_event()
            cur_motion = self.motions[self.frame]
            cur_frame = cur_motion['frame']
            cur_time = cur_motion['time']
            cur_link_poses = cur_motion['link_poses']
            
            root_joint = cur_link_poses[1]
            base_euler = root_joint['euler']
            R1 =  rotx(base_euler[2]) @ rotz(base_euler[1]) @ roty(base_euler[0])
            base_pos = root_joint['pos'] # xyz
            rotMat = Rot.from_matrix(R1)
            mat = rotMat.as_matrix()
            new_base_quat = Rot.from_matrix(mat).as_quat()
            self.t.header.stamp = rospy.Time.now()
            self.t.transform.translation.x = base_pos[2]
            self.t.transform.translation.y = base_pos[0]
            self.t.transform.translation.z = base_pos[1]
            self.t.transform.rotation.x = new_base_quat[0]
            self.t.transform.rotation.y = new_base_quat[1]
            self.t.transform.rotation.z = new_base_quat[2]
            self.t.transform.rotation.w = new_base_quat[3]
            
            for idx, link_pose in enumerate(cur_link_poses):
                if(idx >= 2):
                    eulerxyz = link_pose['euler']
                    pos = link_pose['pos']
                    R1 = rotx(eulerxyz[2]) @rotz(eulerxyz[1]) @ roty(eulerxyz[0])  
                    joint_xyz = mat2euler(R1)
                    self.joint_positions[(idx-2)*3:(idx-2)*3+3] = joint_xyz
                    
                
            if self.playing:
                self.frame += 1
            if self.frame >= len(self.motions):
                self.frame = 0
            self.draw() ## 发布关节数据
            pygame.display.set_caption(
            'AMC Parser - frame %d / %d' % (cur_frame, len(self.motions))
            )
            self.rate.sleep()
            
        pygame.quit()
                
                
                
if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    csvfile = rospy.get_param('csv_file', 'walk.csv')
    motion_path = os.path.join(current_path, '../mocap_ccrp', csvfile)
    print("read csv file form", motion_path)
    motion_fps = rospy.get_param('motion_fps', 30)
    motion = Motion(motion_path = motion_path,
                    retarget_fps = motion_fps)
    
    motion.csv_parser()
    motion.motion_rollout()
    
    pass