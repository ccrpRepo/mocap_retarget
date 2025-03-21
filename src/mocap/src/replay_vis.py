#!/usr/bin/env python3
import casadi  
import rospy
from sensor_msgs.msg import JointState
import tf
from geometry_msgs.msg import TransformStamped
import csv
import numpy as np
from rotation import *
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Header
import pinocchio as pin   
from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper   
from pinocchio.visualize import MeshcatVisualizer 
import os
import sys
import pickle

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

def validate_positions(positions):
    if not all(isinstance(pos, float) for pos in positions):
        raise ValueError("All elements in 'position' must be of float type.")
    return positions

class Replay:
    def __init__(self,
                 urdf=None,
                 motions=None,
                 fps=120,
                 interpolation=True,
                 wrist_motion=False,
                 extend_link=True,
                 start_frame=1,
                 end_frame=None,
                 root_height_offset=0.0
                 ):
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
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.root_height_offset = root_height_offset
        
        self.joint_model = pin.JointModelComposite()
        # 添加 3 个平移自由度
        self.joint_model.addJoint(pin.JointModelTranslation())

        # 添加 3 个旋转自由度 (roll, pitch, yaw)
        self.joint_model.addJoint(pin.JointModelRX())  # Roll
        self.joint_model.addJoint(pin.JointModelRY())  # Pitch
        self.joint_model.addJoint(pin.JointModelRZ())  # Yaw

        current_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_path, '../../g1_description/urdf', 'g1.urdf')
        urdf_dirs = os.path.join(current_path, '../../g1_description/urdf')
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path,
                                                    root_joint = self.joint_model,
                                                    package_dirs = urdf_dirs)

        self.mixed_jointsToLockIDs =[]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        
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
        
        self.link_names = ['pelvis', \
                           'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',\
                           'right_hip_pitch_link',  'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', \
                           'waist_yaw_link', 'waist_roll_link', 'torso_link', \
                           'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', \
                           'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',\
                           'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link',\
                           'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link'
                            ]
        
        self.extend_link_names = ['left_hand_link', 'right_hand_link', 'head_link_o']
        
        self.link_names += self.extend_link_names
        
        if(not wrist_motion):
            self.link_names = [link for link in self.link_names if "wrist" not in link] 
            
        
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
        
        self.robot_tf = tf.TransformListener()
        self.dof_pos = []
        self.dof_vels = []
        self.global_translation_extend = []
        self.global_rotation_extend = []
        self.global_velocity_extend = []
        self.global_angular_velocity_extend = []
        
        if True:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[101, 102], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))
    
    def parse_csv(self):
        with open(self.motions_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  
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
    
    def output(self):
        self.output_dict = {'fps': self.fps,
                            'dof_pos' : self.dof_pos,
                            'dof_vels' : self.dof_vels,
                            'global_translation_extend': self.global_translation_extend,
                            'global_rotation_extend': self.global_rotation_extend,
                            'global_velocity_extend': self.global_velocity_extend,
                            'global_angular_velocity_extend': self.global_angular_velocity_extend
                            }
        
        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_directory = os.path.join(script_directory, 'data')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_path = os.path.join(output_directory, 'output_dict.pkl')
        
        with open(file_path, "wb") as f:
            pickle.dump(self.output_dict, f)
            
        print("PKL FILE OUTPUT COMPLETE!")
        
    
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
        
        final_frame = after_inter_motion_data[-1]['frame']
        motion_length = len(after_inter_motion_data)
        if(self.end_frame == -1):
            self.end_frame = final_frame
        elif(self.end_frame > final_frame):
            self.end_frame = final_frame
            
        
        for idx, motion in enumerate(after_inter_motion_data):
            last_idx = max(idx - 1, 0)          
            next_idx = min(idx + 1, motion_length - 1) 
    
            last_motion = after_inter_motion_data[last_idx]
            next_motion = after_inter_motion_data[next_idx]
            
            cur_frame = motion['frame']
            if(cur_frame < self.start_frame):
                continue
            elif(cur_frame > self.end_frame):
                break
            
            cur_time = motion['time']
            last_time = last_motion['time']
            next_time = next_motion['time']
            
            root_joint = np.array(motion['rootjoint'])
            last_root_joint = np.array(last_motion['rootjoint'])
            next_root_joint = np.array(next_motion['rootjoint'])
            
            root_joint[2] += self.root_height_offset
            last_root_joint[2] += self.root_height_offset
            next_root_joint[2] += self.root_height_offset
            
            joint = np.array(motion['joint'])
            last_joint = np.array(last_motion['joint'])
            next_joint = np.array(next_motion['joint'])
            
            root_joint_vel = (next_root_joint - last_root_joint) / (next_time - last_time)
            joint_vel = (next_joint - last_joint) / (next_time - last_time)
            
            q_vis = np.concatenate([root_joint,joint])
            qd_vis = np.concatenate([root_joint_vel,joint_vel])
            
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
            
            pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_vis, qd_vis)
            pin.computeForwardKinematicsDerivatives(self.reduced_robot.model, self.reduced_robot.data, q_vis, qd_vis, np.zeros(self.reduced_robot.model.nv))
            
            # output data
            global_translation_extend_frame = []
            global_rotation_extend_frame = []
            global_velocity_extend_frame = []
            global_angular_velocity_extend_frame = []
            
            if(wrist_motion):
                self.dof_pos.append(joint)
                self.dof_vels.append(joint_vel)
            else:
                self.dof_pos.append(np.concatenate([joint[:19], joint[23:27]]))
                self.dof_vels.append(np.concatenate([joint_vel[:19], joint_vel[23:27]]))
                
            for idx, link_name in enumerate(self.link_names):
                link_id = self.reduced_robot.model.getFrameId(link_name)
                pin.updateFramePlacement(self.reduced_robot.model, self.reduced_robot.data, link_id)
                se3_pose = self.reduced_robot.data.oMf[link_id]
                global_translation_extend_frame.append(se3_pose.translation.tolist())
                global_rotation_extend_frame.append(pin.Quaternion(se3_pose.rotation).coeffs()) # x y z w
                
                velocity = pin.getFrameVelocity(self.reduced_robot.model, self.reduced_robot.data, link_id)
                global_velocity_extend_frame.append(velocity.linear)
                global_angular_velocity_extend_frame.append(velocity.angular)
                
            self.global_translation_extend.append(global_translation_extend_frame)
            self.global_rotation_extend.append(global_rotation_extend_frame)
            self.global_velocity_extend.append(global_velocity_extend_frame)
            self.global_angular_velocity_extend.append(global_angular_velocity_extend_frame)
                    
            self.vis.display(q_vis)
                
            self.rate.sleep()
        
        
if __name__ == '__main__':
    replay_fps = rospy.get_param('replay_fps', 120)
    inter = rospy.get_param('interpolation', True)
    wrist_motion = rospy.get_param('wrist_motion', False)
    extend_link = rospy.get_param('extend_link', True)
    start_frame = rospy.get_param('start_frame', 1)
    end_frame = rospy.get_param('end_frame', -1)
    root_height_offset = rospy.get_param('root_height_offset', 0.0)
    outputdata = rospy.get_param('outputdata', True)
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_path, '../../g1_description/urdf', 'g1.urdf')
    print(urdf_path)
    motions_path = os.path.join(current_path, 'data', 'output.csv')
    rerun = Replay(urdf=urdf_path,
                   motions=motions_path,
                   fps=replay_fps,
                   interpolation=inter,
                   wrist_motion = wrist_motion,
                   extend_link=extend_link,
                   start_frame=start_frame,
                   end_frame=end_frame,
                   root_height_offset=root_height_offset
                   )
    rerun.parse_csv()
    rospy.sleep(1) # wait for rviz launch
    
    rerun.run()
    if(outputdata):
        rerun.output()