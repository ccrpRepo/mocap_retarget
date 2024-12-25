#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from kinematics import *
from sensor_msgs.msg import JointState
from robot_ik import *
import pinocchio as pin
import tf

def rotx(theta):
    # 旋转角度（以弧度表示）
    theta = np.radians(theta)  # 30度转化为弧度

    # 绕X轴的旋转矩阵
    Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])
    return Rx
    
def roty(theta):
    # 旋转角度（以弧度表示）
    theta = np.radians(theta)  # 30度转化为弧度

    # 绕X轴的旋转矩阵
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
               [0, 1, 0],
               [-np.sin(theta), 0, np.cos(theta)]])
    return Ry
    
def rotz(theta):
    # 旋转角度（以弧度表示）
    theta = np.radians(theta)  # 30度转化为弧度

    # 绕X轴的旋转矩阵
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])
    return Rz

class FramePoseSub:
    def __init__(self):
        # initial positon
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
        
        
    def joint_publisher(self):
        self.joint_pub = rospy.Publisher('g1_joint_states', JointState, queue_size=10)
        

if __name__ == "__main__":
    rospy.init_node('g1_retarget', anonymous=True)
    framesub = FramePoseSub()
    # framesub.pose_subscriber()
    framesub.joint_publisher()
    rate = rospy.Rate(1000)  # 1000 Hz
    urdf_path = "/home/crp/mocap_ws/src/g1_description/urdf/g1.urdf" 
    listener = tf.TransformListener()
    
    
    arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = True)
    sol_q_last = np.zeros(35)
    sol_q_last[7] = -0.1
    sol_q_last[10] = 0.3
    sol_q_last[11] = -0.2
    sol_q_last[13] = -0.1
    sol_q_last[16] = 0.3
    sol_q_last[17] = -0.2
    while not rospy.is_shutdown():
        (trans, rot) = listener.lookupTransform('/world', '/lhand', rospy.Time(0))
        framesub.lhand_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.lhand_target.rotation = rotMat.as_matrix() @ rotx(90) @ roty(-90)
        
        (trans, rot) = listener.lookupTransform('/world', '/rhand', rospy.Time(0))
        framesub.rhand_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.rhand_target.rotation = rotMat.as_matrix() @ rotx(-90) @ roty(90)
        
        (trans, rot) = listener.lookupTransform('/world', '/lfoot', rospy.Time(0))
        framesub.lfoot_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.lfoot_target.rotation = rotMat.as_matrix() @ rotz(-90)
        
        (trans, rot) = listener.lookupTransform('/world', '/rfoot', rospy.Time(0))
        framesub.rfoot_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.rfoot_target.rotation = rotMat.as_matrix() @ rotz(-90)
        
        (trans, rot) = listener.lookupTransform('/world', '/root', rospy.Time(0))
        framesub.root_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.root_target.rotation = rotMat.as_matrix() @ roty(-12)
        
        (trans, rot) = listener.lookupTransform('/world', '/lowerneck', rospy.Time(0))
        framesub.head_target.translation = np.array([trans[0], trans[1], trans[2]])
        quat = np.array([rot[0], rot[1], rot[2], rot[3]])
        rotMat = Rot.from_quat(quat)
        framesub.head_target.rotation = rotMat.as_matrix() @ rotz(-90) @ roty(-90)
        
        sol_q = arm_ik.solve_ik(framesub.lhand_target.homogeneous, 
                            framesub.rhand_target.homogeneous, 
                            framesub.lfoot_target.homogeneous, 
                            framesub.rfoot_target.homogeneous,
                            framesub.root_target.homogeneous, 
                            framesub.head_target.homogeneous,
                            current_lr_arm_motor_q=sol_q_last
                            )
        sol_q_last = sol_q
        
        
        rate.sleep()