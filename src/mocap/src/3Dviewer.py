#!/usr/bin/env python3

import pygame
import numpy as np
import transforms3d.euler as euler
from amc_parser import *
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
import tf
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Int32
from rotation import *


class Viewer:
  def __init__(self, motions=None, fps=None):
    """
    Display motion sequence in 3D.

    Parameter
    ---------
    joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
    values are instance of Joint class.

    motions: List returned from `amc_parser.parse_amc. Each element is a dict
    with joint names as keys and relative rotation degree as values.

    """
    self.motions = motions
    self.frame = 0 # current frame of the motion sequence
    self.playing = False # whether is playing the motion sequence
    self.fps = fps # frame rate

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
    pygame.display.set_caption(
      'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
    )
    
    ## init ros pubulisher
    rospy.init_node('joint_state_publisher_node', anonymous=True)
    
    # 创建一个 Publisher，用于发布 JointState 消息
    self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    self.frameNumpub = rospy.Publisher('/frame_num', Int32, queue_size=10)
    # 定义发布频率
    self.rate = rospy.Rate(self.fps)
    
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
    joint_state_msg.header.stamp = rospy.Time.now()  
    joint_state_msg.name = self.joint_names             
    joint_state_msg.position = self.joint_positions      
    joint_state_msg.velocity = []                  
    joint_state_msg.effort = []                    
    
    self.joint_pub.publish(joint_state_msg)
    self.frameNumpub.publish(self.frame)
    self.br.sendTransformMessage(self.t)
            

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
      
      if self.playing:
        self.frame += 1
        if self.frame >= len(self.motions):
          self.frame = 0
      self.draw() ## 发布关节数据
      pygame.display.set_caption(
        'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
      )
      self.rate.sleep()
    pygame.quit()


if __name__ == '__main__':
  fps = rospy.get_param('motion_fps', 12)
  amcfile = rospy.get_param('amc_file', '86_01.amc')
  amc_path = '../all_asfamc/subjects/86/'+ str(amcfile)
  urdf_path = "../../bone_description/urdf/bone.urdf"
  
  motions = parse_amc(amc_path)
  v = Viewer(motions, fps=fps)
  v.run()
