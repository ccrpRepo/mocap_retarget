#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import tf
from geometry_msgs.msg import TransformStamped
import csv
import os
import numpy as np
from rotation import *
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Header

def validate_positions(positions):
    if not all(isinstance(pos, float) for pos in positions):
        raise ValueError("All elements in 'position' must be of float type.")
    return positions

class Replay:
    def __init__(self, urdf=None, motions=None, fps=120, interpolation=True):
        ## init ros pubulisher
        rospy.init_node('rerun_node', anonymous=True)
        self.interpolation = interpolation
        self.motions_path = motions_path
        self.fps = fps
        self.rate = rospy.Rate(self.fps)
        self.motions_data = []
        self.frame_data={'frame': 0,
                         'time' : 0.0,
                         'rootjoint' : np.zeros(6),
                         'joint' : np.array(29)
                        }
        # self.cur_frame = 0
        
        # 创建一个 Publisher，用于发布 JointState 消息
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', \
                            'left_knee_joint', \
                            'left_ankle_pitch_joint', 'left_ankle_roll_joint', \
                                
                            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', \
                            'right_knee_joint', \
                            'right_ankle_pitch_joint', 'right_ankle_roll_joint', \
                                
                            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',\
                                
                            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', \
                            'left_elbow_joint', \
                            'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', \
                                
                            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', \
                            'right_elbow_joint', \
                            'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
                            ]
        
        self.joint_positions = []
        for _ in range(len(self.joint_names)):
            self.joint_positions.append(0.0)
            
        self.br = tf.TransformBroadcaster()
        # 发布 tf 转换
        # br.sendTransform(goal_msg.pose.position,goal_msg.pose.orientation,rospy.Time.now(),"pelvis","map")
        self.t = TransformStamped()
        self.t.header.frame_id = 'world'
        self.t.header.stamp = rospy.Time(0)
        self.t.child_frame_id = 'root_sphere'
        self.t.transform.translation.x = 0
        self.t.transform.translation.y = 0
        self.t.transform.translation.z = 0.5
        self.t.transform.rotation.w=1
        self.t.transform.rotation.x=0
        self.t.transform.rotation.y=0
        self.t.transform.rotation.z=0
        
    def parse_csv(self):
        with open(self.motions_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 读取表头（可选）
            for row in csv_reader:
                row = [float(item) if isinstance(item, str) and float(item) else item for item in row]
                self.frame_data['frame'] = row[0]
                self.frame_data['time'] = row[1]
                self.frame_data['rootjoint'] = row[2:8]
                self.frame_data['joint'] = row[8:]
                self.motions_data.append(self.frame_data.copy())
        return 0
    
    def interpolate_frames(self, last_frame, cur_frame, idx):
        inter_num = cur_frame - last_frame - 1
        inter_motions = []
        signle_motion ={'frame': 0,
                         'time' : 0.0,
                         'rootjoint' : np.zeros(6),
                         'joint' : np.array(29)
                        }
        time_dist = self.motions_data[idx]['time'] - self.motions_data[idx - 1]['time']
        rootjoint_dist = np.array(self.motions_data[idx]['rootjoint']) - np.array(self.motions_data[idx - 1]['rootjoint'])
        joint_dist = np.array(self.motions_data[idx]['joint']) - np.array(self.motions_data[idx - 1]['joint'])
        unite_time = time_dist / (inter_num + 1)
        unite_rootjoint = rootjoint_dist / (inter_num + 1)
        unite_joint = joint_dist / (inter_num + 1)
        for i in range(int(inter_num)):
            signle_motion['frame'] = self.motions_data[idx - 1]['frame'] + i + 1
            signle_motion['time'] = self.motions_data[idx - 1]['time'] + unite_time * (i + 1)
            signle_motion['rootjoint'] = self.motions_data[idx - 1]['rootjoint'] + unite_rootjoint * (i + 1)
            signle_motion['joint'] = self.motions_data[idx - 1]['joint'] + unite_joint * (i + 1)
            inter_motions.append(signle_motion.copy())
        
                
        return inter_motions
    
    def run(self):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.name = self.joint_names 
        joint_state_msg.velocity = []              
        joint_state_msg.effort = []    
        last_frame = 0
        
        if(self.interpolation):
            after_inter_motion_data = self.motions_data.copy()
            inter_sum = 0
            for i, motion in enumerate(self.motions_data):
                cur_frame = motion['frame']
                if(i > 0):
                    last_frame = self.motions_data[i-1]['frame']
                    inter_motions = []
                    if((cur_frame - last_frame) > 1):
                        inter_motions = self.interpolate_frames(last_frame, cur_frame, i)
                        num = cur_frame - last_frame -1
                        after_inter_motion_data[int(i + inter_sum):int(i + inter_sum)] = inter_motions
                        inter_sum += num
        
        for idx, motion in enumerate(after_inter_motion_data):
            cur_frame = motion['frame']
            cur_time = motion['time']
            root_joint = np.array(motion['rootjoint'])
            joint = np.array(motion['joint'])
            
            # root
            base_pos = root_joint[:3]
            angle = root_joint[3:6]
            # 转换为degree
            angle = angle * 180.0 / 3.1415926
            Rotation_Matrix = eulerxyz2mat(angle)
            rotation = Rot.from_matrix(Rotation_Matrix)
            base_quaternion = rotation.as_quat()
            self.t.header.stamp = rospy.Time.now()
            self.t.transform.translation.x = base_pos[0]
            self.t.transform.translation.y = base_pos[1]
            self.t.transform.translation.z = base_pos[2]
            self.t.transform.rotation.x = base_quaternion[0]
            self.t.transform.rotation.y = base_quaternion[1]
            self.t.transform.rotation.z = base_quaternion[2]
            self.t.transform.rotation.w = base_quaternion[3]
            
            # joint
            self.joint_positions = motion['joint']
            
            # publish
            joint_state_msg.header.stamp = rospy.Time.now()  
            joint_state_msg.position = validate_positions(self.joint_positions)
            self.joint_pub.publish(joint_state_msg)
            self.br.sendTransformMessage(self.t)
            
            print(f"frame: {cur_frame}  time stamp: {cur_time}")
            self.rate.sleep()
            
        
        
if __name__ == '__main__':
    replay_fps = rospy.get_param('replay_fps', 120)
    inter = rospy.get_param('interpolation', True)
    current_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_path, '../../g1_description/urdf', 'g1.urdf')
    print(urdf_path)
    motions_path = os.path.join(current_path, 'data', 'output.csv')
    rerun = Replay(urdf=urdf_path, 
                   motions=motions_path, 
                   fps=replay_fps,
                   interpolation=inter)
    rerun.parse_csv() 
    rospy.sleep(3) # wait for rviz launch 
    
    rerun.run()