#!/usr/bin/env python
import casadi                                                                       
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from weighted_moving_filter import WeightedMovingFilter



class RobotIK:
    def __init__(self, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Visualization = Visualization
        
        self.joint_model = pin.JointModelComposite()
        # 添加 3 个平移自由度
        self.joint_model.addJoint(pin.JointModelTranslation())

        # 添加 3 个旋转自由度 (roll, pitch, yaw)
        self.joint_model.addJoint(pin.JointModelRX())  # Roll
        self.joint_model.addJoint(pin.JointModelRY())  # Pitch
        self.joint_model.addJoint(pin.JointModelRZ())  # Yaw

        self.robot = pin.RobotWrapper.BuildFromURDF('/home/crp/mocap_ws/src/g1_description/urdf/g1.urdf',
                                                    root_joint = self.joint_model,
                                                    package_dirs = '/home/crp/mocap_ws/src/g1_description/urdf/') # for test

        self.mixed_jointsToLockIDs =[]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # self.reduced_robot.model.addFrame(
        #     pin.Frame('L_ee',
        #               self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
        #               pin.SE3(np.eye(3),
        #                       np.array([0.05,0,0]).T),
        #               pin.FrameType.OP_FRAME)
        # )
        
        # self.reduced_robot.model.addFrame(
        #     pin.Frame('R_ee',
        #               self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
        #               pin.SE3(np.eye(3),
        #                       np.array([0.05,0,0]).T),
        #               pin.FrameType.OP_FRAME)
        # )

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     print(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_lhand = casadi.SX.sym("tf_lhand", 4, 4)
        self.cTf_rhand = casadi.SX.sym("tf_rhand", 4, 4)
        self.cTf_root = casadi.SX.sym("tf_root", 4, 4)
        self.cTf_lfoot = casadi.SX.sym("tf_lfoot", 4, 4)
        self.cTf_rfoot = casadi.SX.sym("tf_rfoot", 4, 4)
        self.cTf_head = casadi.SX.sym("tf_head", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.lhand_id = self.reduced_robot.model.getFrameId("left_wrist_yaw_link")
        self.rhand_id = self.reduced_robot.model.getFrameId("right_wrist_yaw_link")
        self.root_id = self.reduced_robot.model.getFrameId("root_sphere") # pelvis
        self.head_id = self.reduced_robot.model.getFrameId("head_sphere")
        self.lfoot_id = self.reduced_robot.model.getFrameId("left_ankle_roll_link")
        self.rfoot_id = self.reduced_robot.model.getFrameId("right_ankle_roll_link")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_lhand, self.cTf_rhand, self.cTf_root, self.cTf_lfoot, self.cTf_rfoot],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.lhand_id].translation - self.cTf_lhand[:3,3],
                    self.cdata.oMf[self.rhand_id].translation - self.cTf_rhand[:3,3],
                    self.cdata.oMf[self.root_id].translation - self.cTf_root[:3,3],
                    self.cdata.oMf[self.lfoot_id].translation - self.cTf_lfoot[:3,3],
                    self.cdata.oMf[self.rfoot_id].translation - self.cTf_rfoot[:3,3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_lhand, self.cTf_rhand, self.cTf_root, self.cTf_lfoot, self.cTf_rfoot, self.cTf_head],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.lhand_id].rotation @ self.cTf_lhand[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.rhand_id].rotation @ self.cTf_rhand[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.root_id].rotation @ self.cTf_root[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.lfoot_id].rotation @ self.cTf_lfoot[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.rfoot_id].rotation @ self.cTf_rfoot[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.head_id].rotation @ self.cTf_head[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_lhand = self.opti.parameter(4, 4)
        self.param_tf_rhand = self.opti.parameter(4, 4)
        self.param_tf_root = self.opti.parameter(4, 4)
        self.param_tf_lfoot = self.opti.parameter(4, 4)
        self.param_tf_rfoot = self.opti.parameter(4, 4)
        self.param_tf_head = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_lhand, self.param_tf_rhand, self.param_tf_root, self.param_tf_lfoot, self.param_tf_rfoot))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_lhand, self.param_tf_rhand, self.param_tf_root, self.param_tf_lfoot, self.param_tf_rfoot,self.param_tf_head))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-3
            },
            'print_time':False,# print or not
            'calc_lam_p':False 
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.reduced_robot.model.nq)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[101, 102], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['lhand_target', 'rhand_target', 'lfoot_target', 'rfoot_target', 'root_target', 'head_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 10
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )
    # If the robot arm is not the same size as your arm :)
    def scale_lengths(self, human_lhand_pose, human_rhand_pose, human_lfoot_pose, human_rfoot_pose, robot_root_pose, robot_head_pose,
                      human_arm_length=0.784, robot_arm_length=0.50, human_leg_length=0.95, robot_leg_length=0.60):
        arm_scale_factor = robot_arm_length / human_arm_length
        robot_lhand_pose = human_lhand_pose.copy()
        robot_rhand_pose = human_rhand_pose.copy()
        robot_lhand_pose[:3, 3] = robot_head_pose[:3, 3] + arm_scale_factor * (robot_lhand_pose[:3, 3] - robot_head_pose[:3, 3])
        robot_rhand_pose[:3, 3] = robot_head_pose[:3, 3] + arm_scale_factor * (robot_rhand_pose[:3, 3] - robot_head_pose[:3, 3])
        
        leg_scale_factor = robot_leg_length / human_leg_length
        robot_lfoot_pose = human_lfoot_pose.copy()
        robot_rfoot_pose = human_rfoot_pose.copy()
        robot_lfoot_pose[:3, 3] = robot_root_pose[:3, 3] + leg_scale_factor * (robot_lfoot_pose[:3, 3] - robot_root_pose[:3, 3])
        robot_rfoot_pose[:3, 3] = robot_root_pose[:3, 3] + leg_scale_factor * (robot_rfoot_pose[:3, 3] - robot_root_pose[:3, 3])
        
        return robot_lhand_pose, robot_rhand_pose, robot_lfoot_pose, robot_rfoot_pose

    def solve_ik(self, left_wrist, right_wrist, left_foot, right_foot, root, head, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        left_wrist, right_wrist, left_foot, right_foot = self.scale_lengths(left_wrist, right_wrist, left_foot, right_foot, root, head)
        if self.Visualization:
            self.vis.viewer['lhand_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['rhand_target'].set_transform(right_wrist)  # for visualization
            self.vis.viewer['lfoot_target'].set_transform(left_foot)   # for visualization
            self.vis.viewer['rfoot_target'].set_transform(right_foot)  # for visualization
            self.vis.viewer['root_target'].set_transform(root)  # for visualization
            self.vis.viewer['head_target'].set_transform(head)  # for visualization

        self.opti.set_value(self.param_tf_lhand, left_wrist)
        self.opti.set_value(self.param_tf_rhand, right_wrist)
        self.opti.set_value(self.param_tf_lfoot, left_foot)
        self.opti.set_value(self.param_tf_rfoot, right_foot)
        self.opti.set_value(self.param_tf_root, root)
        self.opti.set_value(self.param_tf_head, head)
        
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
            # print(sol_q)
            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            # sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            # sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q


if __name__ == "__main__":
    arm_ik = RobotIK(Visualization = True)

    # initial positon
    lhand_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.3]),
    )

    rhand_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.3]),
    )
    
    lfoot_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0, 0.15, -0.80]),
    )
    
    rfoot_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0, -0.15, -0.80]),
    )
    
    head_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0, 0, 0.25]),
    )
    
    root_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0, 0, 0]),
    )

    rotation_speed = 0.01
    noise_amplitude_translation = 0.001
    noise_amplitude_rotation = 0.01
    
    sol_q_last = np.zeros(35)
    sol_q_last[3] = 1.0
    sol_q_last[7] = -0.1
    sol_q_last[10] = 0.3
    sol_q_last[11] = -0.2
    sol_q_last[13] = -0.1
    sol_q_last[16] = 0.3
    sol_q_last[17] = -0.2
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        step = 0
        while True:
            # Apply rotation noise with bias towards y and z axes
            rotation_noise_L = pin.Quaternion(
                np.cos(np.random.normal(0, noise_amplitude_rotation) / 2),0,np.random.normal(0, noise_amplitude_rotation / 2),0).normalized()  # y bias

            rotation_noise_R = pin.Quaternion(
                np.cos(np.random.normal(0, noise_amplitude_rotation) / 2),0,0,np.random.normal(0, noise_amplitude_rotation / 2)).normalized()  # z bias
            
            if step <= 30:
                angle = rotation_speed * step
                # lhand_target.rotation = (rotation_noise_L * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)).toRotationMatrix()  # y axis
                # rhand_target.rotation = (rotation_noise_R * pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))).toRotationMatrix()  # z axis
                lhand_target.translation += (np.array([0.001,  0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                rhand_target.translation += (np.array([0.001, -0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                # lfoot_target.rotation = (rotation_noise_L * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)).toRotationMatrix()  # y axis
                # rfoot_target.rotation = (rotation_noise_R * pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))).toRotationMatrix()  # z axis
                lfoot_target.translation += (np.array([0.001,  0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                rfoot_target.translation += (np.array([0.001, -0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
            else:
                angle = rotation_speed * (60 - step)
                # lhand_target.rotation = (rotation_noise_L * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)).toRotationMatrix()  # y axis
                # rhand_target.rotation = (rotation_noise_R * pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))).toRotationMatrix()  # z axis
                lhand_target.translation -= (np.array([0.001,  0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                rhand_target.translation -= (np.array([0.001, -0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                # lfoot_target.rotation = (rotation_noise_L * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)).toRotationMatrix()  # y axis
                # rfoot_target.rotation = (rotation_noise_R * pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))).toRotationMatrix()  # z axis
                lfoot_target.translation -= (np.array([0.001,  0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))
                rfoot_target.translation -= (np.array([0.001, -0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))

            sol_q = arm_ik.solve_ik(lhand_target.homogeneous, 
                            rhand_target.homogeneous, 
                            lfoot_target.homogeneous, 
                            rfoot_target.homogeneous, 
                            root_target.homogeneous, 
                            head_target.homogeneous,
                            current_lr_arm_motor_q=sol_q_last
                            )
            sol_q_last = sol_q

            step += 1
            if step > 240:
                step = 0
            # time.sleep(0.001)