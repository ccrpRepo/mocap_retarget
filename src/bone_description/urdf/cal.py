import numpy as np
import math

def cal_length(a, b):
    return a * 0.056444 * b

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

def cal_norm_length(a):
    return 0.056444 * a


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

def null_space(A, tol=1e-12):
    """
    计算矩阵 A 的零空间 (null space)
    :param A: 输入矩阵
    :param tol: 判断奇异值是否为零的容差
    :return: 零空间基的列向量组成的矩阵
    """
    # 使用奇异值分解 (SVD)
    u, s, vh = np.linalg.svd(A)
    
    # 找到接近零的奇异值对应的索引
    null_mask = (s <= tol)
    null_space = vh.T[:, null_mask]  # 取 V^T 的对应列向量
    
    return null_space

def orthonormal_basis_from_z(z):
    """
    根据给定的 z 轴单位向量，求解对应的 x 轴和 y 轴，使它们与 z 互相正交。
    :param z: 已知的 z 轴向量 (3,)
    :return: x, y, z 构成的正交基向量
    """
    z = z / np.linalg.norm(z)  # 单位化 z 轴
    # 选择一个任意向量（非平行于 z）
    arbitrary = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    
    # 计算与 z 垂直的 x 轴
    x = np.cross(arbitrary, z)
    x = x / np.linalg.norm(x)  # 单位化
    
    # 计算 y 轴（x 和 z 的叉积保证正交）
    y = np.cross(z, x)
    y = y / np.linalg.norm(y) 
    
    return x, y, z

    
    

if __name__ == '__main__':
    
    
    rr_R_root = np.zeros((3,3))
    rr_R_root[1,0] = 1
    rr_R_root[2,1] = 1
    rr_R_root[0,2] = 1
    root_R_rr = np.linalg.inv(rr_R_root)
    # print(rr_R_root)
    
    # rr_d_lclavicle = cal_length(np.array([-0.068874, 0.964749, 0.254]), 3.26503)
    # print("rr_d_lclavicle", rr_d_lclavicle)
    
    # root_R_lhumerus = eulerzyx2mat(np.array([180, -30, -90]))
    # print("root_R_lhumerus: \n", root_R_lhumerus)
    
    # rr_R_lhumerus = rr_R_root @ root_R_lhumerus
    # print("rr_R_lhumerus: \n",rr_R_lhumerus)
    
    # euler_lhumerus = mat2euler_zyx(rr_R_lhumerus)
    # print("euler_lhumerus: ", euler_lhumerus)
    
    # lhumerus_R_root = np.linalg.inv(root_R_lhumerus)
    # lhumerus_R_rr = np.linalg.inv(rr_R_lhumerus)
    
    # rr_d_lhumerus = cal_length(np.array([-2.05079e-027, 1, -4.48941e-011]), 5.68236)
    # lhumerus_d_lhumerus = lhumerus_R_rr @ rr_d_lhumerus
    # print("lhumerus_d_lhumerus", lhumerus_d_lhumerus)
    
    # lhumerus_R_lradius = np.eye(3)
    # lradius_R_lhumerus = np.eye(3)
    
    
    ##---------------------------------------------------------------------------------
    rr_d_lhip = cal_length(np.array([0.220866, 0.561145, -0.797705]), 2.46425)
    print("rr_d_lhip: ",rr_d_lhip)
    rr_R_lhip = np.eye(3)
    root_R_lhip = root_R_rr @ rr_R_lhip
    
    ##---------------------------------------------------------------------------------
    rr_d_rhip = cal_length(np.array([ 0.21134, -0.610499, -0.763299]), 2.57533)
    print("rr_d_rhip: ",rr_d_rhip)
    rr_R_rhip = np.eye(3)
    root_R_rhip = root_R_rr @ rr_R_rhip
    
    ##---------------------------------------------------------------------------------
    ##--------------------------------left _leg----------------------------------------
    root_R_lfemur = eulerzyx2mat(np.array([0, 0, 20]))
    lhip_R_root = np.linalg.inv(root_R_lhip)
    lhip_R_lfemur = lhip_R_root @ root_R_lfemur
    euler_lfemur = mat2euler_zyx(lhip_R_lfemur)
    print("euler_lfemur: ",euler_lfemur)
    
    ##---------------------------------------------------------------------------------
    rr_d_lfemur = cal_length(np.array([0, 0.34202, -0.939693]), 7.95556)
    rr_R_lfemur = rr_R_root @ root_R_lfemur
    lfemur_R_rr = np.linalg.inv(rr_R_lfemur)
    lfemur_d_lfemur = lfemur_R_rr @ rr_d_lfemur
    print("lfemur_d_lfemur: ",lfemur_d_lfemur)
    
    root_R_ltibia = eulerzyx2mat(np.array([0, 0, 20]))
    lfemur_R_root = np.linalg.inv(root_R_lfemur)
    lfemur_R_ltibia =  lfemur_R_root @ root_R_ltibia
    euler_ltibia = mat2euler_zyx(lfemur_R_ltibia)
    print("euler_ltibia: ",euler_ltibia)
    
    ##---------------------------------------------------------------------------------
    rr_d_ltibia = cal_length(np.array([0, 0.34202, -0.939693]), 7.9832)
    rr_R_ltibia = rr_R_root @ root_R_ltibia
    ltibia_R_rr = np.linalg.inv(rr_R_ltibia)
    ltibia_d_ltibia = ltibia_R_rr @ rr_d_ltibia
    print("ltibia_d_ltibia: ",ltibia_d_ltibia)
    
    root_R_lfoot = eulerzyx2mat(np.array([-90, 7.62852e-016, 20]))
    ltibia_R_root = np.linalg.inv(root_R_ltibia)
    ltibia_R_lfoot = ltibia_R_root @ root_R_lfoot
    euler_lfoot = mat2euler_zyx(ltibia_R_lfoot)
    print("euler_lfoot: ", euler_lfoot)
    
    ##---------------------------------------------------------------------------------
    rr_d_lfoot = cal_length(np.array([0.977343, 0.0723927, -0.198897]), 2.23085)
    rr_R_lfoot = rr_R_root @ root_R_lfoot
    lfoot_R_rr = np.linalg.inv(rr_R_lfoot)
    lfoot_d_lfoot = lfoot_R_rr @ rr_d_lfoot
    print("lfoot_d_lfoot: ", lfoot_d_lfoot)
    
    root_R_ltoes = eulerzyx2mat(np.array([-90, 7.62852e-016, 20]))
    lfoot_R_root = np.linalg.inv(root_R_lfoot)
    lfoot_R_ltoes = lfoot_R_root @ root_R_ltoes
    euler_ltoes = mat2euler_zyx(lfoot_R_ltoes)
    print("euler_ltoes: ", euler_ltoes)
    
    ##--------------------------------------------------------------------------------
    rr_d_toes = cal_length(np.array([1, 1.53826e-011, -4.21988e-011]), 1.12127)
    rr_R_ltoes = rr_R_root @ root_R_ltoes
    ltoes_R_rr = np.linalg.inv(rr_R_ltoes)
    ltoes_d_ltoes = ltoes_R_rr @ rr_d_toes
    print("ltoes_d_ltoes: ",ltoes_d_ltoes)
    
    ##--------------------------------------------------------------------------------
    ##------------------------------right_leg-----------------------------------------
    root_R_rfemur = eulerzyx2mat(np.array([0, 0, -20]))
    rhip_R_root = np.linalg.inv(root_R_rhip)
    rhip_R_rfemur = rhip_R_root @ root_R_rfemur
    euler_rfemur = mat2euler_zyx(rhip_R_rfemur)
    print("euler_rfemur: ", euler_rfemur)
    
    ##--------------------------------------------------------------------------------
    rr_d_rfemur = cal_length(np.array([ 0, -0.34202, -0.939693]), 8.65257)
    rr_R_rfemur = rr_R_root @ root_R_rfemur
    rfemur_R_rr = np.linalg.inv(rr_R_rfemur)
    rfemur_d_rfemur = rfemur_R_rr @ rr_d_rfemur
    print("rfemur_d_rfemur: ", rfemur_d_rfemur)
    
    root_R_rtibia = eulerzyx2mat(np.array([0, 0, -20]))
    rfemur_R_root = np.linalg.inv(root_R_rfemur)
    rfemur_R_rtibia = rfemur_R_root @ root_R_rtibia
    euler_rtibia = mat2euler_zyx(rfemur_R_rtibia)
    print("euler_rtibia: ", euler_rtibia)
    
    ##--------------------------------------------------------------------------------
    rr_d_rtibia = cal_length(np.array([0, -0.342021, -0.939692 ]), 7.37807)
    rr_R_rtibia = rr_R_root @ root_R_rtibia
    rtibia_R_rr = np.linalg.inv(rr_R_rtibia)
    rtibia_d_rtibia = rtibia_R_rr @ rr_d_rtibia
    print("rtibia_d_rtibia: ", rtibia_d_rtibia)
    
    root_R_rfoot = eulerzyx2mat(np.array([-90, -7.62852e-016, -20]))
    rtibia_R_root = np.linalg.inv(root_R_rtibia)
    rtibia_R_rfoot = rtibia_R_root @ root_R_rfoot
    euler_rfoot = mat2euler_zyx(rtibia_R_rfoot)
    print("euler_rfoot: ", euler_rfoot)
    
    ##--------------------------------------------------------------------------------
    rr_d_rfoot = cal_length(np.array([0.998233, -0.0239304, -0.0543852]), 1.87018)
    rr_R_rfoot = rr_R_root @ root_R_rfoot
    rfoot_R_rr = np.linalg.inv(rr_R_rfoot)
    rfoot_d_rfoot = rfoot_R_rr @ rr_d_rfoot
    print("rfoot_d_rfoot", rfoot_d_rfoot)
    
    root_R_rtoes = eulerzyx2mat(np.array([-90, -7.62852e-016, -20]))
    rfoot_R_root = np.linalg.inv(root_R_rfoot)
    rfoot_R_rtoes = rfoot_R_root @ root_R_rtoes
    euler_rtoes = mat2euler_zyx(rfoot_R_rtoes)
    print("euler_rtoes: ", euler_rtoes)
    
    ##-------------------------------------------------------------------------------
    rr_d_rtoes = cal_length(np.array([1, -1.53584e-011, -4.2242e-011]), 0.951568)
    rr_R_rtoes = rr_R_root @ root_R_rtoes
    rtoes_R_rr = np.linalg.inv(rr_R_rtoes)
    rtoes_d_rtoes = rtoes_R_rr @ rr_d_rtoes
    print("rtoes_d_rtoes", rtoes_d_rtoes)
    
    ##-------------------------------------------------------------------------------
    # root_R_clavicle = np.eye(3)
    
    # rr_d_rclavicle = cal_length(np.array([-0.102435, -0.954374, 0.280496]), 3.25419)
    # rr_R_rclavicle = rr_R_root @ root_R_clavicle
    # rclavicle_R_rr = np.linalg.inv(rr_R_rclavicle)
    # rclavicle_d_rclavicle = rclavicle_R_rr @ rr_d_rclavicle
    # print("rclavicle_d_rclavicle: ", rclavicle_d_rclavicle)
    
    # root_R_rhumerus = eulerzyx2mat(np.array([180, 30, 90]))
    # rclavicle_R_root = np.linalg.inv(root_R_clavicle)
    # rclavicle_R_rhumerus = rclavicle_R_root @ root_R_rhumerus
    # euler_rhumerus = mat2euler_zyx(rclavicle_R_rhumerus)
    # print("euler_rhumerus: ", euler_rhumerus)
    
    ##------------------------------------------------------------------------------
    root_R_lowerback = np.eye(3)
    rr_R_lowerback = rr_R_root @ root_R_lowerback
    euler_lowerback = mat2euler_zyx(rr_R_lowerback)
    print("euler_lowerback: ", euler_lowerback)
    
    ##------------------------------------------------------------------------------
    rr_d_lowerback = cal_length(np.array([0.0438587, 0.00322089, 0.999033]), 2.07568)
    lowerback_R_rr = np.linalg.inv(rr_R_lowerback)
    lowerback_d_lowerback = lowerback_R_rr @ rr_d_lowerback
    print("lowerback_d_lowerback: ", lowerback_d_lowerback)
    
    root_R_upperback = eulerzyx2mat(np.array([0, 0, 0]))
    lowerback_R_root = np.linalg.inv(root_R_lowerback)
    lowerback_R_upperback = lowerback_R_root @ root_R_upperback
    euler_upperback = mat2euler_zyx(lowerback_R_upperback)
    print("euler_upperback: ", euler_upperback)
    
    ##------------------------------------------------------------------------------
    rr_d_upperback = cal_length(np.array([-0.0229403, -0.00321932, 0.999732]), 2.06253)
    rr_R_upperback = rr_R_root @ root_R_upperback
    upperback_R_rr = np.linalg.inv(rr_R_upperback)
    upperback_d_upperback = upperback_R_rr @ rr_d_upperback
    print("upperback_d_upperback: ", upperback_d_upperback)
    
    root_R_thorax = eulerzyx2mat(np.array([0,0,0]))
    upperback_R_root = np.linalg.inv(root_R_upperback)
    upperback_R_thorax = upperback_R_root @ root_R_thorax
    euler_thorax = mat2euler_zyx(upperback_R_thorax)
    print("euler_thorax: ", euler_thorax)
    
    ##------------------------------------------------------------------------------
    rr_d_throax = cal_length(np.array([-0.0453918, -0.00537537, 0.998955]), 2.06975)
    rr_R_throax = rr_R_root @ root_R_thorax
    throax_R_rr = np.linalg.inv(rr_R_throax)
    throax_d_throax = throax_R_rr @ rr_d_throax
    print("throax_d_throax: ", throax_d_throax)
    
    root_R_lowerneck = eulerzyx2mat(np.array([0, 0, 0]))
    throax_R_root = np.linalg.inv(root_R_thorax)
    throax_R_lowerneck = throax_R_root @ root_R_lowerneck
    euler_lowerneck = mat2euler_zyx(throax_R_lowerneck)
    print("euler_lowerneck: ", euler_lowerneck)
    
    ##-----------------------------------------------------------------------------
    rr_d_lowerneck = cal_length(np.array([0.108363, -0.00013604, 0.994111]), 1.50836)
    rr_R_lowerneck = rr_R_root @ root_R_lowerneck
    lowerneck_R_rr = np.linalg.inv(rr_R_lowerneck)
    lowerneck_d_lowerneck = lowerneck_R_rr @ rr_d_lowerneck
    print("lowerneck_d_lowerneck: ", lowerneck_d_lowerneck)
    
    root_R_upperneck = eulerzyx2mat(np.array([0, 0, 0]))
    lowerneck_R_root = np.linalg.inv(root_R_lowerneck)
    lowerneck_R_upperneck = lowerneck_R_root @ root_R_upperneck
    euler_upperneck = mat2euler_zyx(lowerneck_R_upperneck)
    print("euler_upperneck: ", euler_upperneck)
    
    ##-----------------------------------------------------------------------------
    rr_d_upperneck = cal_length(np.array([-0.174482, 0.0397262, 0.983859]), 1.53037)
    rr_R_upperneck = rr_R_root @ root_R_upperneck
    upperneck_R_rr = np.linalg.inv(rr_R_upperneck)
    upperneck_d_upperneck = upperneck_R_rr @ rr_d_upperneck
    print("upperneck_d_upperneck: ", upperneck_d_upperneck)
    
    root_R_head = eulerzyx2mat(np.array([0, 0, 0]))
    upperneck_R_root = np.linalg.inv(root_R_upperneck)
    upperneck_R_head = upperneck_R_root @ root_R_head
    euler_head = mat2euler_zyx(upperneck_R_head)
    print("euler_head: ", euler_head)
    
    ##-----------------------------------------------------------------------------
    rr_d_head = cal_length(np.array([-0.0755378, 0.0109991, 0.997082]), 1.56928)
    rr_R_head = rr_R_root @ root_R_head
    head_R_rr = np.linalg.inv(rr_R_head)
    head_d_head = head_R_rr @ rr_d_head
    print("head_d_head: ", head_d_head)
    
    ##----------------------------------------------------------------------------
    root_R_lclavicle = np.eye(3)
    
    rr_d_lclavicle = cal_length(np.array([-0.068874, 0.964749, 0.254]), 3.26503)
    rr_R_lclavicle = rr_R_root @ root_R_lclavicle
    lclavicle_R_rr = np.linalg.inv(rr_R_lclavicle)
    lclavicle_d_lclavicle = lclavicle_R_rr @ rr_d_lclavicle
    print("lclavicle_d_lclavicle: ", lclavicle_d_lclavicle)
    
    root_R_lhumerus = eulerzyx2mat(np.array([180, -30, -90]))
    lclavicle_R_root = np.linalg.inv(root_R_lclavicle)
    lclavicle_R_lhumerus = lclavicle_R_root @ root_R_lhumerus
    euler_lhumerus = mat2euler_zyx(lclavicle_R_lhumerus)
    print("euler_lhumerus: ", euler_lhumerus)
    
    ##----------------------------------------------------------------------------
    rr_d_lhumerus = cal_length(np.array([-2.05079e-027, 1, -4.48941e-011]), 5.68236)
    rr_R_lhumerus = rr_R_root @ root_R_lhumerus
    lhuermus_R_rr = np.linalg.inv(rr_R_lhumerus)
    lhumerus_d_lhumerus = lhuermus_R_rr @ rr_d_lhumerus
    print("lhumerus_d_lhumerus: ", lhumerus_d_lhumerus)
    
    root_R_lradius = eulerzyx2mat(np.array([180, -30, -90]))
    lhuermus_R_root = np.linalg.inv(root_R_lhumerus)
    lhuermus_R_lradius = lhuermus_R_root @ root_R_lradius
    euler_lradius = mat2euler_zyx(lhuermus_R_lradius)
    print("euler_lradius: ", euler_lradius)
    
    ##----------------------------------------------------------------------------
    rr_d_lradius = cal_length(np.array([6.2882e-027, 1, -4.48956e-011]), 3.17112)
    rr_R_lradius = rr_R_root @ root_R_lradius
    lradius_R_rr = np.linalg.inv(rr_R_lradius)
    lradius_d_lradius = lradius_R_rr @ rr_d_lradius
    print("lradius_d_lradius: ", lradius_d_lradius)
    
    root_R_lwrist = eulerzyx2mat(np.array([-1.04724e-014, 90, 90]))
    lradius_R_root = np.linalg.inv(root_R_lradius)
    lradius_R_lwrist =  lradius_R_root @ root_R_lwrist
    euler_lwrist = mat2euler_zyx(lradius_R_lwrist)
    print("euler_lwrist: ",euler_lwrist)
    
    ##---------------------------------------------------------------------------------
    rr_d_lwrist = cal_length(np.array([6.2882e-027, 1, -4.48956e-011]), 3.17112)
    rr_R_lwrist = rr_R_root @ root_R_lwrist
    lwrist_R_rr = np.linalg.inv(rr_R_lwrist)
    lwrist_d_lwrist = lwrist_R_rr @ rr_d_lwrist
    print("lradius_d_lradius: ", lradius_d_lradius)
    
    root_R_lwrist = eulerzyx2mat(np.array([-1.04724e-014, 90, 90]))
    lradius_R_root = np.linalg.inv(root_R_lradius)
    lradius_R_lwrist =  lradius_R_root @ root_R_lwrist
    euler_lwrist = mat2euler_zyx(lradius_R_lwrist)
    print("euler_lwrist: ",euler_lwrist)
    
    # ##---------------------------------------------------------------------------------
    # rr_d_lwrist = cal_length(np.array([ 1.73219e-026, 1, -4.49014e-011]), 1.58556)
    # lwrist_R_lradius = np.linalg.inv(lradius_R_lwrist)
    # lwrist_R_rr = lwrist_R_lradius @ lradius_R_rr
    # lwrist_d_lwrist = lwrist_R_rr @ rr_d_lwrist
    # print("lwrist_d_lwrist: ",lwrist_d_lwrist)
    
    # root_R_lhand = eulerzyx2mat(np.array([-2.21057e-014, 90, 90]))
    # lwrist_R_root = lwrist_R_lradius @ lradius_R_root
    # lwrsit_R_lhand = lwrist_R_root @ root_R_lhand
    # euler_lhand = mat2euler_zyx(lwrsit_R_lhand)
    # print("euler_lhand: ",euler_lhand)
    
    # ##---------------------------------------------------------------------------------
    # rr_d_lfingers = cal_length(np.array([ 3.46438e-026, 1, -4.49312e-011]), 0.632691)
    # lhand_R_lwrist = np.linalg.inv(lwrsit_R_lhand)
    # lhand_R_rr = lhand_R_lwrist @ lwrist_R_rr
    # lhand_d_lhand = lhand_R_rr @ rr_d_lfingers
    # print("lhand_d_lhand: ",lhand_d_lhand)
    
    # root_R_lfingers = eulerzyx2mat(np.array([-4.42114e-014, 90, 90]))
    # lhand_R_root = lhand_R_lwrist @ lwrist_R_root
    # lhand_R_fingers = lhand_R_root @ root_R_lfingers
    # euler_lfingers = mat2euler_zyx(lhand_R_fingers)
    # print("euler_lfingers: ",euler_lfingers)
    
    # ##---------------------------------------------------------------------------------
    # rr_d_lfingerstop = cal_length(np.array([6.92876e-026, 1, -4.48894e-011]), 0.510091)
    # fingers_R_lhand = np.linalg.inv(lhand_R_fingers)
    # fingers_R_rr = fingers_R_lhand @ lhand_R_rr
    # lfingers_d_lfingers = fingers_R_rr @ rr_d_lfingerstop
    # print("lfingers_d_lfingers: ",lfingers_d_lfingers)
    
    # ##---------------------------------------------------------------------------------
    # root_R_lthumb = eulerzyx2mat(np.array([-90, 45, 2.85299e-015]))
    # lwrist_R_lthumb = lwrist_R_root @ root_R_lthumb
    # euler_lthumb = mat2euler_zyx(lwrist_R_lthumb)
    # print("euler_lthumb: ",euler_lthumb)
    
    # ##---------------------------------------------------------------------------------
    # rr_d_lthumbtop = cal_length(np.array([0.707107, 0.707107, -6.34705e-011]), 0.732392)
    # lthumb_R_root = np.linalg.inv(root_R_lthumb)
    # lthumb_R_rr = lthumb_R_root @ root_R_rr
    # lthumb_d_lthumb = lthumb_R_rr @ rr_d_lthumbtop
    # print("lthumb_d_lthumb: ", lthumb_d_lthumb)
    
    ##-----------------------------------------------------------------------------------
    root_R_rclavicle = np.eye(3)
    lowerneck_R_lrclavicle = lowerneck_R_root @ root_R_rclavicle
    eluer_rclavicle = mat2euler_zyx(lowerneck_R_lrclavicle)
    print("eluer_rclavicle: ", eluer_rclavicle)
    
    ##-----------------------------------------------------------------------------------
    rr_d_rclavicle = cal_length(np.array([-0.102435, -0.954374, 0.280496]), 3.25419)
    rr_R_rclavicle = rr_R_root @ root_R_rclavicle
    rclavicle_R_rr = np.linalg.inv(rr_R_rclavicle)
    rclavicle_d_rclavicle = rclavicle_R_rr @ rr_d_rclavicle
    print("rclavicle_d_rclavicle: ", rclavicle_d_rclavicle)
    
    root_R_rhumerus = eulerzyx2mat(np.array([180, 30, 90]))
    rclavicle_R_root = np.linalg.inv(root_R_rclavicle)
    rclavicle_R_rhumerus = rclavicle_R_root @ root_R_rhumerus
    euler_rhumerus = mat2euler_zyx(rclavicle_R_rhumerus)
    print("euler_rhumerus: ", euler_rhumerus)
    
    ##-----------------------------------------------------------------------------------
    rr_d_rhumerus = cal_length(np.array([-1.07204e-017, -1, -4.48994e-011]), 5.8712)
    rr_R_rhumerus = rr_R_root @ root_R_rhumerus
    rhumerus_R_rr = np.linalg.inv(rr_R_rhumerus)
    rhumerus_d_rhumerus = rhumerus_R_rr @ rr_d_rhumerus
    print("rhumerus_d_rhumerus: ", rhumerus_d_rhumerus)
    
    root_R_rradius = eulerzyx2mat(np.array([180, 30, 90]))
    rhumerus_R_root = np.linalg.inv(root_R_rhumerus)
    rhumerus_R_rradius = rhumerus_R_root @ root_R_rradius
    euler_rhumerus = mat2euler_zyx(rhumerus_R_rradius)
    print("euler_rhumerus: ", euler_rhumerus)
    
    ##----------------------------------------------------------------------------------
    rr_d_rradius = cal_length(np.array([5.89292e-027, -1, -4.48972e-011]), 3.37589)
    rr_R_rradius = rr_R_root @ root_R_rradius
    rradius_R_rr = np.linalg.inv(rr_R_rradius)
    rradius_d_rradius = rradius_R_rr @ rr_d_rradius
    print("rradius_d_rradius: ", rradius_d_rradius)
    
    root_R_rwrist = eulerzyx2mat(np.array([-1.04724e-014, -90, -90]))
    rradius_R_root = np.linalg.inv(root_R_rradius)
    rradius_R_rwrist = rradius_R_root @ root_R_rwrist
    euler_rwrist = mat2euler_zyx(rradius_R_rwrist)
    print("euler_rwrist", euler_rwrist)
    
    ##----------------------------------------------------------------------------------
    rr_d_rwrist = cal_length(np.array([ 1.73219e-026, -1, -4.48942e-011]), 1.68794)
    rr_R_rwrist = rr_R_root @ root_R_rwrist
    rwrist_R_rr = np.linalg.inv(rr_R_rwrist)
    rwrist_d_rwrist = rwrist_R_rr @ rr_d_rwrist
    print("rwrist_d_rwrist: ",rwrist_d_rwrist)
    
    root_R_rhand = eulerzyx2mat(np.array([-2.21057e-014, -90, -90]))
    rwrist_R_root = np.linalg.inv(root_R_rwrist)
    rwrist_R_rhand = rwrist_R_root @ root_R_rhand
    euler_rhand = mat2euler_zyx(rwrist_R_rhand)
    print("euler_rhand: ", euler_rhand)
    
    ##----------------------------------------------------------------------------------
    rr_d_rhand = cal_length(np.array([3.46438e-026, -1, -4.48514e-011]), 0.603257)
    rr_R_rhand = rr_R_root @ root_R_rhand
    rhand_R_rr = np.linalg.inv(rr_R_rhand)
    rhand_d_rhand = rhand_R_rr @ rr_d_rhand
    print("rhand_d_rhand: ", rhand_d_rhand)
    
    root_R_rfingers = eulerzyx2mat(np.array([-4.42114e-014, -90, -90]))
    rhand_R_root = np.linalg.inv(root_R_rhand)
    rhand_R_rfingers = rhand_R_root @ root_R_rfingers
    euler_rfingers = mat2euler_zyx(rhand_R_rfingers)
    print("euler_rfingers: ", euler_rfingers)
    
    ##---------------------------------------------------------------------------------
    rr_d_rfingers = cal_length(np.array([6.92876e-026, -1, -4.50028e-011]), 0.486362)
    rr_R_rfingers = rr_R_root @ root_R_rfingers
    fingers_R_rr = np.linalg.inv(rr_R_rfingers)
    fingers_d_fingers = fingers_R_rr @ rr_d_rfingers
    print("fingers_d_fingers: ", fingers_d_fingers)
    
    ##---------------------------------------------------------------------------------
    root_R_rthumb = eulerzyx2mat(np.array([-90, -45, -2.85299e-015]))
    rwrist_R_rthumb = rwrist_R_root @ root_R_rthumb
    euler_rthumb = mat2euler_zyx(rwrist_R_rthumb)
    print("euler_rthumb: ",euler_rthumb)
    
    ##---------------------------------------------------------------------------------
    
    rr_d_rthumb = cal_length(np.array([0.707107, -0.707107, -6.34926e-011]), 0.698321)
    rr_R_rthumb = rr_R_root @ root_R_rthumb
    rthumb_R_rr = np.linalg.inv(rr_R_rthumb)
    rthumb_d_rthumb = rthumb_R_rr @ rr_d_rthumb
    print("rthumb_d_rthumb: ", rthumb_d_rthumb)
    
    ##*********************************************************************************
    ##*********************************************************************************
    ##*********************************************************************************
    ##*********************************************************************************
    
    lhip_size = cal_length(np.array([ 0.220866, 0.561145, -0.797705]), 2.46425)
    lhip_midpoint = lhip_size / 2
    print("lhip_midpoint: ", lhip_midpoint) 
    
    x_axis, y_axis, z_axis = orthonormal_basis_from_z(np.array([0.220866, 0.561145, -0.797705]))
    rr_R_lhip_link = np.zeros([3,3])
    rr_R_lhip_link[:, 0] = x_axis
    rr_R_lhip_link[:, 1] = y_axis
    rr_R_lhip_link[:, 2] = z_axis
    lhip_R_rr = np.linalg.inv(rr_R_lhip)
    lhip_R_link = lhip_R_rr @ rr_R_lhip_link
    euler_lhip_link = mat2euler_zyx(lhip_R_link)
    print("euler_lhip_link: ", euler_lhip_link)
    
    ##--------------------------------------------------------------------------------
    root_z = np.array([-0.610499, -0.763299, 0.21134])
    normal_len =  2.57533
    rr_z = rr_R_root @ root_z
    rhip_size = cal_length(rr_z, normal_len)
    rhip_midpoint = rhip_size / 2
    print("norm_length: ", cal_norm_length(normal_len))
    print("rhip_midpoint: ", rhip_midpoint) 
    
    
    x_axis, y_axis, z_axis = orthonormal_basis_from_z(rr_z)
    rr_R_rhip_link = np.zeros([3,3])
    rr_R_rhip_link[:, 0] = x_axis
    rr_R_rhip_link[:, 1] = y_axis
    rr_R_rhip_link[:, 2] = z_axis
    rhip_R_rr = np.linalg.inv(rr_R_rhip)
    rhip_R_link = rhip_R_rr @ rr_R_rhip_link
    euler_rhip_link = mat2euler_zyx(rhip_R_link)
    print("euler_rhip_link: ", euler_rhip_link)
    
    ##--------------------------------------------------------------------------------
    root_R_lhand = eulerzyx2mat(np.array([-2.21057e-014, 90, 90]))
    rr_R_lhand = rr_R_root @ root_R_lhand
    root_R_lfingers = eulerzyx2mat(np.array([-4.42114e-014, 90, 90]))
    rr_R_lfingers = rr_R_root @ root_R_lfingers
    root_R_lthumb = eulerzyx2mat(np.array([-90, 45, 2.85299e-015]))
    rr_R_lthumb = rr_R_root @ root_R_lthumb
    
    print("\n")
    joint_R_root = np.linalg.inv(root_R_lhip)
    root_z = np.array([0.561145, -0.797705, 0.220866 ])
    normal_len =  0.632691  
    
    # rr_z = root_z
    link_size = cal_length(root_z, normal_len)
    root_link_midpoint = link_size / 2
    joint_link_midpoint = joint_R_root @ root_link_midpoint
    print("norm_length: ", cal_norm_length(normal_len))
    print("link_midpoint: ", joint_link_midpoint) 
    
    x_axis, y_axis, z_axis = orthonormal_basis_from_z(root_z)
    root_R_link = np.zeros([3,3])
    root_R_link[:, 0] = x_axis
    root_R_link[:, 1] = y_axis
    root_R_link[:, 2] = z_axis
    joint_R_link = joint_R_root @ root_R_link
    euler_link = mat2euler_zyx(joint_R_link)
    print("euler_link: ", euler_link)
    print("\n")
    # limit = np.array([-45.0, 45.0]) * 3.14159 / 180.0
    # print("limit: ", limit)
    
    quat = [0.7071, 0.7071, 0.0, 0.0]
    