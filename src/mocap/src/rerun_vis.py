#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import tf
from geometry_msgs.msg import TransformStamped

class Rerun:
    def __init__(self, motions=None, fps=None):
        ## init ros pubulisher
        rospy.init_node('rerun_node', anonymous=True)
        self.motions = motions
        self.fps = fps
        self.rate = rospy.Rate(self.fps)
        
        # 创建一个 Publisher，用于发布 JointState 消息
        self.joint_pub = rospy.Publisher('rerun_joint_states', JointState, queue_size=10)
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
        self.t.child_frame_id = 'root'
        self.t.transform.translation.x = 0
        self.t.transform.translation.y = 0
        self.t.transform.translation.z = 0.5
        self.t.transform.rotation.w=1
        self.t.transform.rotation.x=0
        self.t.transform.rotation.y=0
        self.t.transform.rotation.z=0
        
if __name__ == '__main__':
    urdf_path = "../../g1_description/urdf/g1.urdf"
    
    rerun = Rerun(urdf_path)