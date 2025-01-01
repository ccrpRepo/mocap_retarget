#!/usr/bin/env python3
import numpy as np

rr_R_root = np.zeros((3,3))
rr_R_root[1,0] = 1
rr_R_root[2,1] = 1
rr_R_root[0,2] = 1
root_R_rr = np.linalg.inv(rr_R_root)

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