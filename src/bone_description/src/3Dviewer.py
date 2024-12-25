#!/usr/bin/env python3

import pygame
import numpy as np
import time
import transforms3d.euler as euler
from amc_parser import *
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
import math
import tf
from scipy.spatial.transform import Rotation as Rot
from transforms3d.euler import euler2mat as euler2mat_api
from kinematics import *

rr_R_root = np.zeros((3,3))
rr_R_root[1,0] = 1
rr_R_root[2,1] = 1
rr_R_root[0,2] = 1
root_R_rr = np.linalg.inv(rr_R_root)

world_R_root = np.zeros((3,3))
world_R_root[0,0] = 1
world_R_root[1,1] = 1
world_R_root[2,2] = 1

R1_offset = np.zeros((3,3))
R1_offset[2,0] = 1
R1_offset[0,1] = 1
R1_offset[1,2] = 1

R2_offset = np.zeros((3,3))
R2_offset[1,0] = 1
R2_offset[2,1] = 1
R2_offset[0,2] = 1
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

def eulerxyz2mat(axis):
    Rx = rotx(axis[0])
    Ry = roty(axis[1])
    Rz = rotz(axis[2])
    
    res = (Rx.dot(Ry)).dot(Rz)
    
    return res

def eulerzyx2mat(axis):
    Rx = rotx(axis[0])
    Ry = roty(axis[1])
    Rz = rotz(axis[2])
    
    res = (Rz.dot(Ry)).dot(Rx)
    
    return res


class Viewer:
  def __init__(self, motions=None, robot=None, pub_frame_name=None):
    """
    Display motion sequence in 3D.

    Parameter
    ---------
    joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
    values are instance of Joint class.

    motions: List returned from `amc_parser.parse_amc. Each element is a dict
    with joint names as keys and relative rotation degree as values.

    """
    self.robot = robot
    self.motions = motions
    self.pub_frame_name = pub_frame_name
    self.frame = 0 # current frame of the motion sequence
    self.playing = False # whether is playing the motion sequence
    self.fps = 15 # frame rate

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
    self.screen_size = (1024, 768)
    self.screen = pygame.display.set_mode(
      self.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
    )
    pygame.display.set_caption(
      'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
    )
    self.clock = pygame.time.Clock()
    
    ## init ros pubulisher
    rospy.init_node('joint_state_publisher_node', anonymous=True)
    
    # 创建一个 Publisher，用于发布 JointState 消息
    self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    # 定义发布频率
    self.rate = rospy.Rate(self.fps)  # 120Hz
    self.rootPose_pub = rospy.Publisher('/root_pose', PoseStamped, queue_size=10)
    self.lfootPose_pub = rospy.Publisher('/lfoot_pose', PoseStamped, queue_size=10)
    self.rfootPose_pub = rospy.Publisher('/rfoot_pose', PoseStamped, queue_size=10)
    self.lowerneckPose_pub = rospy.Publisher('/lowerneck_pose', PoseStamped, queue_size=10)
    self.lhandPose_pub = rospy.Publisher('/lhand_pose', PoseStamped, queue_size=10)
    self.rhandPose_pub = rospy.Publisher('/rhand_pose', PoseStamped, queue_size=10)
    self.frame_pub = {
      'root' : self.rootPose_pub,
      'lfoot' : self.lfootPose_pub,
      'rfoot' : self.rfootPose_pub,
      'lowerneck' : self.lowerneckPose_pub,
      'lhand' : self.lhandPose_pub,
      'rhand' : self.rhandPose_pub
      
    }
    
    
    
    self.joint_names = ['left_hip_joint_x', 'left_hip_joint_y', 'left_hip_joint_z', \
                    'left_knee_joint', \
                    'lfoot_joint_x', 'lfoot_joint_z', \
                    'ltoe_joint', \
                    'right_hip_joint_x', 'right_hip_joint_y', 'right_hip_joint_z',
                    'right_knee_joint', \
                    'rfoot_joint_x', 'rfoot_joint_z', \
                    'rtoe_joint', \
                    'lowerback_joint_x', 'lowerback_joint_y', 'lowerback_joint_z', \
                    'upperback_joint_x', 'upperback_joint_y', 'upperback_joint_z', \
                    'thorax_joint_x', 'thorax_joint_y', 'thorax_joint_z', \
                    'lowerneck_joint_x', 'lowerneck_joint_y', 'lowerneck_joint_z', \
                    'upperneck_joint_x', 'upperneck_joint_y', 'upperneck_joint_z', \
                    'head_joint_x', 'head_joint_y', 'head_joint_z', \
                    'lclavicle_joint_y', 'lclavicle_joint_z', \
                    'lshoulder_joint_x', 'lshoulder_joint_y', 'lshoulder_joint_z', \
                    'left_elbow_joint', \
                    'lwrist_joint', \
                    'lhand_joint_x', 'lhand_joint_z', \
                    'lfingers_joint', \
                    'lthumb_joint_x', 'lthumb_joint_z', \
                    'rclavicle_joint_y', 'rclavicle_joint_z', \
                    'rshoulder_joint_x', 'rshoulder_joint_y', 'rshoulder_joint_z', \
                    'right_elbow_joint', \
                    'rwrist_joint', \
                    'rhand_joint_x', 'rhand_joint_z', \
                    'rfingers_joint', \
                    'rthumb_joint_x', 'rthumb_joint_z']
    
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
    self.t.child_frame_id = 'root'
    self.t.transform.translation.x = 0
    self.t.transform.translation.y = 0
    self.t.transform.translation.z = 0.5
    self.t.transform.rotation.w=1
    self.t.transform.rotation.x=0
    self.t.transform.rotation.y=0
    self.t.transform.rotation.z=0


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
    joint_state_msg.header.stamp = rospy.Time.now()  # 时间戳
    joint_state_msg.name = self.joint_names              # 关节名称
    joint_state_msg.position = self.joint_positions      # 关节位置
    joint_state_msg.velocity = []                   # （可选）关节速度
    joint_state_msg.effort = []                     # （可选）关节受力
    
    self.joint_pub.publish(joint_state_msg)
    self.br.sendTransformMessage(self.t)
    
    
    # for index, frame_name in enumerate(self.pub_frame_name):
    #     frame_pose = robot.get_frame_position(frame_name)
    #     if frame_pose:
    #         # print(f"Frame {frame_name} position:\n{frame_pose}")
    #         pose = PoseStamped()
    #         pose.header.stamp = rospy.Time.now()
    #         pose.header.frame_id = frame_name + '_pose'
    #         pose.pose.position.x = frame_pose.translation[0]
    #         pose.pose.position.y = frame_pose.translation[1]
    #         pose.pose.position.z = frame_pose.translation[2]
    #         rotM = Rot.from_matrix(frame_pose.rotation)
    #         quater = rotM.as_quat()
    #         pose.pose.orientation.x = quater[0]
    #         pose.pose.orientation.y = quater[1]
    #         pose.pose.orientation.z = quater[2]
    #         pose.pose.orientation.w = quater[3]
    #         self.frame_pub[frame_name].publish(pose)
            
            

  def set_motion(self, motions):
    """
    Set motion sequence for viewer.

    Paramter
    --------
    motions: List returned from `amc_parser.parse_amc. Each element is a dict
    with joint names as keys and relative rotation degree as values.

    """
    self.motions = motions

  def run(self):
    """
    Main loop.

    """
    
    while not self.done:
      self.process_event()
      ## 关节赋值
      motion = self.motions[self.frame]
      
      # root
      root_joint = np.array(motion['root'])
      base_pos = rr_R_root @ (root_joint[:3] * 0.056444)
      angle = root_joint[3:6]
      Rotation_Matrix = rr_R_root @ eulerzyx2mat(angle) @ rotz(-90.0) @ roty(-90.0)
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
      self.joint_positions[0:3] = np.array(motion['lfemur'])
      self.joint_positions[3] = motion['ltibia'][0] 
      self.joint_positions[4:6] = np.array(motion['lfoot']) 
      self.joint_positions[6] = np.array(motion['ltoes'])[0] 
      self.joint_positions[7:10] = np.array(motion['rfemur']) 
      self.joint_positions[10] = np.array(motion['rtibia'])[0]
      self.joint_positions[11:13] = np.array(motion['rfoot']) 
      self.joint_positions[13] = np.array(motion['rtoes'])[0] 
      
      self.joint_positions[14:17] = np.array(motion['lowerback']) 
      self.joint_positions[17:20] = np.array(motion['upperback']) 
      self.joint_positions[20:23] = np.array(motion['thorax']) 
      self.joint_positions[23:26] = np.array(motion['lowerneck']) 
      self.joint_positions[26:29] = np.array(motion['upperneck']) 
      self.joint_positions[29:32] = np.array(motion['head'])
      
      self.joint_positions[32:34] = np.array(motion['lclavicle'])
      self.joint_positions[34:37] = np.array(motion['lhumerus'])
      self.joint_positions[37] = np.array(motion['lradius'])[0]
      self.joint_positions[38] = np.array(motion['lwrist'])[0]
      self.joint_positions[39:41] = np.array(motion['lhand'])
      self.joint_positions[41] = np.array(motion['lfingers'])[0]
      self.joint_positions[42:44] = np.array(motion['lthumb'])
      
      self.joint_positions[44:46] = np.array(motion['rclavicle'])
      self.joint_positions[46:49] = np.array(motion['rhumerus'])
      self.joint_positions[49] = np.array(motion['rradius'])[0]
      self.joint_positions[50] = np.array(motion['rwrist'])[0]
      self.joint_positions[51:53] = np.array(motion['rhand'])
      self.joint_positions[53] = np.array(motion['rfingers'])[0]
      self.joint_positions[54:56] = np.array(motion['rthumb'])
      
      # 转换为radius
      self.joint_positions = [x * 3.1415926 / 180.0 for x in self.joint_positions]
      
      q_cur = np.concatenate((base_pos, base_quaternion, self.joint_positions))
      self.robot.forward_kinematics(q_cur)
      
      if self.playing:
        self.frame += 1
        if self.frame >= len(self.motions):
          self.frame = 0
      self.draw() ## 发布关节数据
      pygame.display.set_caption(
        'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
      )
      # self.clock.tick(self.fps) ## ros 延迟
      self.rate.sleep()
    pygame.quit()


if __name__ == '__main__':
  amc_path = '/home/crp/all_asfamc/subjects/86/86_14.amc'
  urdf_path = "/home/crp/mocap_ws/src/bone_description/urdf/bone.urdf"
  package_dirs = '/home/crp/mocap_ws/src/g1_description/urdf/'
  pub_frame_name=['root', 'lfoot', 'rfoot', 'lowerneck', 'lhand', 'rhand']
  robot = RobotPinocchio(urdf_path, verbose=True, package_dirs = package_dirs)
  motions = parse_amc(amc_path)
  v = Viewer(motions, robot, pub_frame_name=pub_frame_name)
  v.run()
