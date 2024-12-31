#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import JointState
from robot_ik import *
import pinocchio as pin
import tf
import csv
import os
from std_msgs.msg import Int32

def rotx(theta):
    theta = np.radians(theta) 
    Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])
    return Rx
    
def roty(theta):
    theta = np.radians(theta) 
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
               [0, 1, 0],
               [-np.sin(theta), 0, np.cos(theta)]])
    return Ry
    
def rotz(theta):
    theta = np.radians(theta)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])
    return Rz


class FramePoseSub:
    def __init__(self):
        # initial positon
        rospy.init_node('g1_retarget', anonymous=True)
        self.subscriber = rospy.Subscriber('/frame_num', Int32, self.frameNumcallback)
        self.lhand_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, +0.25, 0.3]),
        )
        self.rhand_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, -0.25, 0.3]),
        )
        self.lfoot_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0, 0.15, -0.80]),
        )
        self.rfoot_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0, -0.15, -0.80]),
        )
        self.head_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0, 0, 0.25]),
        )
        self.root_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0, 0, 0]),
        )
        self.g1_joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', \
                    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', \
                    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', \
                    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', \
                    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', \
                    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', \
                    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', \
                    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', \
                    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
                    ]
        
        self.frame_num = 0
    
    def frameNumcallback(self, data):
        self.frame_num = data.data
    
    def joint_publisher(self):
        self.joint_pub = rospy.Publisher('g1_joint_states', JointState, queue_size=10)
        
if __name__ == "__main__":
    framesub = FramePoseSub()
    framesub.joint_publisher()
    rate = rospy.Rate(1000)  # 1000 Hz
    urdf_path = "../g1_description/urdf/g1.urdf" 
    human_tf = tf.TransformListener()
    data = []
    timestamp = 0.0
    
    robot_ik = RobotIK(Visualization = True)
    last_frame = 0
    sol_q_last = np.zeros(35)
    sol_q_last[7] = -0.1
    sol_q_last[10] = 0.3
    sol_q_last[11] = -0.2
    sol_q_last[13] = -0.1
    sol_q_last[16] = 0.3
    sol_q_last[17] = -0.2
    start_time = rospy.Time.now().to_sec()
    
    while not rospy.is_shutdown():
        
        (trans, rot) = human_tf.lookupTransform('/world', '/lhand', rospy.Time(0))
        framesub.lhand_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.lhand_target.rotation = rotMat.as_matrix() @ rotx(90) @ roty(-90)
        
        (trans, rot) = human_tf.lookupTransform('/world', '/rhand', rospy.Time(0))
        framesub.rhand_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.rhand_target.rotation = rotMat.as_matrix() @ rotx(-90) @ roty(90)
        
        (trans, rot) = human_tf.lookupTransform('/world', '/lfoot', rospy.Time(0))
        framesub.lfoot_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.lfoot_target.rotation = rotMat.as_matrix() @ rotz(-90)
        
        (trans, rot) = human_tf.lookupTransform('/world', '/rfoot', rospy.Time(0))
        framesub.rfoot_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.rfoot_target.rotation = rotMat.as_matrix() @ rotz(-90)
        
        (trans, rot) = human_tf.lookupTransform('/world', '/root', rospy.Time(0))
        framesub.root_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.root_target.rotation = rotMat.as_matrix() @ roty(-12)
        
        (trans, rot) = human_tf.lookupTransform('/world', '/lowerneck', rospy.Time(0))
        framesub.head_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.head_target.rotation = rotMat.as_matrix() @ rotz(-90) @ roty(-90)
        
        timestamp = rospy.Time.now().to_sec() - start_time
        
        sol_q = robot_ik.solve_ik(framesub.lhand_target.homogeneous, 
                                framesub.rhand_target.homogeneous, 
                                framesub.lfoot_target.homogeneous, 
                                framesub.rfoot_target.homogeneous,
                                framesub.root_target.homogeneous,
                                framesub.head_target.homogeneous,
                                current_lr_arm_motor_q=sol_q_last
                                )
        sol_q_last = sol_q
        
        if(framesub.frame_num != last_frame):
            time_sol_q = np.insert(sol_q, 0, timestamp)
            frame_time_sol_q = np.insert(time_sol_q, 0, framesub.frame_num)
            data.append(frame_time_sol_q)
            
        last_frame = framesub.frame_num
        rate.sleep()
        
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(script_directory, 'data')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_path = os.path.join(output_directory, 'output.csv')
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['frame','timestamp','posX','posY','posZ','roll','pitch','yaw'] + framesub.g1_joint_names)
        for arr in data:
            writer.writerow(arr)
        
    
    print("CSV文件已生成")