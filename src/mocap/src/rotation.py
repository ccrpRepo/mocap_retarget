#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as Rot

rr_R_root = np.zeros((3,3))
rr_R_root[1,0] = 1
rr_R_root[2,1] = 1
rr_R_root[0,2] = 1
root_R_rr = np.linalg.inv(rr_R_root)

rr_R_hips = np.zeros((3,3))
rr_R_hips[1,0] = 1
rr_R_hips[2,1] = 1
rr_R_hips[0,2] = 1
hips_R_rr = np.linalg.inv(rr_R_hips)

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

def mat2euler(R):

    # 提取旋转矩阵的元素
    R11, R12, R13 = R[0, 0], R[0, 1], R[0, 2]
    R21, R22, R23 = R[1, 0], R[1, 1], R[1, 2]
    R31, R32, R33 = R[2, 0], R[2, 1], R[2, 2]

    # 计算欧拉角（绕XYZ轴的旋转）
    theta_x = -np.arctan2(R23, R33)
    theta_y = -np.arctan2(-R13, np.sqrt(R11**2 + R12**2))
    theta_z = -np.arctan2(R12, R11)
    
    # print(f"Euler Angles (XYZ):")
    # print(f"Theta_x (rotation about X-axis): {theta_x} radius")
    # print(f"Theta_y (rotation about Y-axis): {theta_y} radius")
    # print(f"Theta_z (rotation about Z-axis): {theta_z} radius")

    # 将角度从弧度转换为度
    theta_x_deg = np.degrees(theta_x)
    theta_y_deg = np.degrees(theta_y)
    theta_z_deg = np.degrees(theta_z)

    # print(f"Euler Angles (XYZ):")
    # print(f"Theta_x (rotation about X-axis): {theta_x_deg} degrees")
    # print(f"Theta_y (rotation about Y-axis): {theta_y_deg} degrees")
    # print(f"Theta_z (rotation about Z-axis): {theta_z_deg} degrees")
    
    return np.array([theta_x,theta_y,theta_z])

def mat2euler_zyx(R):
    # 提取旋转矩阵的元素
    R11, R12, R13 = R[0, 0], R[0, 1], R[0, 2]
    R21, R22, R23 = R[1, 0], R[1, 1], R[1, 2]
    R31, R32, R33 = R[2, 0], R[2, 1], R[2, 2]

    # 计算欧拉角（绕ZYX轴的旋转）
    theta_x = np.arctan2(R32, R33)  # 绕X轴的旋转
    theta_y = np.arctan2(-R31, np.sqrt(R11**2 + R21**2))  # 绕Y轴的旋转
    theta_z = np.arctan2(R21, R11)  # 绕Z轴的旋转
    
    # 将角度从弧度转换为度
    theta_x_deg = np.degrees(theta_x)
    theta_y_deg = np.degrees(theta_y)
    theta_z_deg = np.degrees(theta_z)

    # print(f"Euler Angles (XYZ):")
    # print(f"Theta_x (rotation about X-axis): {theta_x_deg} degrees")
    # print(f"Theta_y (rotation about Y-axis): {theta_y_deg} degrees")
    # print(f"Theta_z (rotation about Z-axis): {theta_z_deg} degrees")
    
    return np.array([theta_x,theta_y,theta_z])

if __name__ == '__main__':
    R1 = rotx(-12) @ roty(-88) @rotz(0)
    R2 = rotx(168) @ roty(-88) @rotz(180)
    e1 = mat2euler(R1)
    e2 = mat2euler(R2)
    euler_xyz1 = mat2euler(R1)
    euler_xyz2 = mat2euler(R2)
    print("euler", euler_xyz2 / np.pi * 180.0)
    R3 = rotx(euler_xyz1[0]/ np.pi * 180.0) @ roty(euler_xyz1[1]/ np.pi * 180.0) @rotz(euler_xyz1[2]/ np.pi * 180.0)
    print(R1 - R2)
    print(e1)
    print(e2)
    newmat_1 = eulerxyz2mat(e1 * 180.0 / np.pi)
    newmat_2 = eulerxyz2mat(e2 * 180.0 / np.pi)
    print(newmat_1)
    print(newmat_2)
    
    mat1 = Rot.from_quat(np.array([-0.041597,	-0.917272,	-0.099532,	0.383374]))
    e1 = mat2euler(mat1.as_matrix())
    mat2 = Rot.from_quat(np.array([0.040883,	0.920141,	0.099948,	-0.376404]))
    e2 = mat2euler(mat2.as_matrix())
    # print(e1)
    # print(e2)
    
    