from scipy.spatial.transform import Rotation as Rot
import numpy as np

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



quat = np.array([-0.806177, -0.090286,	0.347565,	0.470241])
rotMat = Rot.from_quat(quat)
mat = rotMat.as_matrix()
euler_zyx = mat2euler_zyx(mat)
print(euler_zyx)

xita = np.arctan(0.0289/0.117)
len = np.sqrt(0.0289**2 + 0.117**2)
print(xita)
print(len)
