import pinocchio as pin
import numpy as np

class RobotPinocchio:
    def __init__(self, urdf_path, model_path=None, verbose=False, floating_base=True, package_dirs = None):
        """
        初始化机器人模型。

        :param urdf_path: URDF 文件的路径
        :param model_path: 模型的基本路径（用于加载几何文件），可选
        :param verbose: 是否打印加载信息
        """
        self.urdf_path = urdf_path
        self.model_path = model_path
        self.package_dirs = package_dirs
        if verbose:
            print(f"Loading URDF from: {urdf_path}")
            if model_path:
                print(f"Model path: {model_path}")

       # 加载模型，支持浮动基座
        self.model = pin.Model()
        self.root_joint = pin.JointModelFreeFlyer()
        pin.buildModelFromUrdf(urdf_path,self.root_joint, self.model)
        self.data = self.model.createData()
       

        # 打印机器人基本信息
        if verbose:
            print(f"Model name: {self.model.name}")
            print(f"Number of joints: {self.model.njoints}")
            print(f"Number of frames: {self.model.nframes}")

    def get_joint_names(self):
        """
        获取机器人中所有关节的名称。
        :return: 关节名称列表
        """
        return self.model.names

    def get_joint_index(self, joint_name):
        """
        获取关节的索引。
        :param joint_name: 关节名称
        :return: 索引（从 1 开始），如果关节不存在则返回 -1
        """
        if joint_name in self.model.names:
            return self.model.getJointId(joint_name)
        else:
            return -1

    def forward_kinematics(self, q):
        """
        执行前向运动学计算。
        :param q: 关节位置数组（长度应与 self.model.nq 一致）
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def get_frame_position(self, frame_name):
        """
        获取指定帧的位姿。
        :param frame_name: 帧名称
        :return: 帧的位姿(SE3 对象），如果帧不存在则返回 None
        """
        if self.model.existFrame(frame_name):
            frame_id = self.model.getFrameId(frame_name)
            return self.data.oMf[frame_id]
        else:
            print(f"Frame '{frame_name}' not found.")
            return None

    def get_default_configuration(self):
        """
        获取机器人的默认关节配置（零配置）。
        :return: 默认配置数组
        """
        defalut_q = np.zeros(self.model.nq)
        defalut_q[3] = 1.
        return defalut_q

    def compute_jacobian(self, q, frame_name):
        """
        计算指定帧的雅可比矩阵。
        :param q: 关节位置数组
        :param frame_name: 帧名称
        :return: 雅可比矩阵(6xN),如果帧不存在则返回 None
        """
        if self.model.existFrame(frame_name):
            frame_id = self.model.getFrameId(frame_name)
            pin.computeJointJacobians(self.model, self.data, q)
            return pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL)
        else:
            print(f"Frame '{frame_name}' not found.")
            return None

    def print_robot_info(self):
        """
        打印机器人的关节和帧信息。
        """
        print("Joint names:")
        for i, name in enumerate(self.model.names):
            print(f"  {i}: {name}")

        print("\nFrame names:")
        for i, frame in enumerate(self.model.frames):
            print(f"  {i}: {frame.name}")
        
        

# 使用示例
if __name__ == "__main__":
    urdf_path = "/home/crp/mocap_ws/src/bone_description/urdf/bone.urdf"  # 替换为实际路径

    # 初始化机器人类
    robot = RobotPinocchio(urdf_path, verbose=True)

    # 获取默认关节配置
    q = robot.get_default_configuration()

    # 执行前向运动学
    robot.forward_kinematics(q)

    # 获取末端执行器的位姿
    pub_frame_name=['root', 'lfoot', 'rfoot', 'lowerneck', 'lhand', 'rhand']
        
    for frame_name in pub_frame_name:
        frame_position = robot.get_frame_position(frame_name)
        if frame_position:
            print(f"Frame {frame_name} position:\n{frame_position}")

    # 打印机器人信息
    robot.print_robot_info()
