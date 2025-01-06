# 项目简介

这是一个用于人体动作捕捉数据重映射的开源项目，骨架模型和运动数据来源于[CMU开源数据集](http://mocap.cs.cmu.edu/faqs.php)  
项目将骨架模型转换为urdf格式，方便使用ROS-tf以及pinocchio等工具进行开发，支持CMU开源数据集的86号人体运动数据 

视频教程及演示：[开源！人体运动捕捉数据可视化与重映射，支持宇树G1机器人](https://www.bilibili.com/video/BV1tC66YTEYh/?spm_id_from=333.1387.homepage.video_card.click&vd_source=713b35f59bdf42930757aea07a44e7cb)
  
本项目参考宇树开源的[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)以及[AMCParser](https://github.com/CalciferZh/AMCParser)，环境的安装可以参考这两个项目，建议使用conda。目前仅支持G1机器人的重映射，但相同的方法迁移到其他机器人上也是不难的。  

# 环境安装

1. 创建conda虚拟环境
    ```bash
    conda create -n myenv python=3.8
    ```
2. 安装依赖
    ```bash
    pip install numpy transforms3d matplotlib pygame meshcat
    ```
3. ROS + casadi + pinocchio 确保pinocchio版本是3.1.0
    ```bash
    conda install -c conda-forge pyyaml rospkg casadi
    ```
    ```bash
    conda install pinocchio=3.1.0 -c conda-forge
    ```

# 使用说明

1. ROS编译  + source
    ```bash
    cd 到当前目录
    catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3
    source ./devel/setup.bash 
    ```


2. 可视化骨架模型运动数据 + 重映射 + 保存机器人运动数据, 
    ```bash
    roslaunch mocap g1_retarget.launch
    ```
    launch文件的参数说明：  
    - motion_fps ： 运动数据每秒运行的帧数，CMU数据集默认是120帧每秒，建议运行在30帧以下，  
    motion_fps过高会导致重映射丢帧，会出现提示`[INFO] [1735800433.903647]: fps is too high! skip xx frames at xxx`
    - amc_file ：读取的运动数据文件 86_01 ～ 86_15；
    - render ： 实时显示重映射效果（meshcat网页显示）；
    - OutputData：是否保存当前重映射的机器人运动数据，保存在 `data/output.csv` 
      
    等待meshcat中机器人初始化完毕，激活生成的小窗口，点击空格键播放运动数据。  
    再次按空格暂停，关闭该窗口，并按ctrl+c按键中止程序，机器人运动数据会自动保存，显示 `CSV FILE OUTPUT COMPLETE!`。



3. 播放机器人重映射效果 + 插值处理
    ```bash
    roslaunch mocap g1_replay.launch
    ```
    launch文件的参数说明：  
    - replay_fps ： 运动数据每秒运行的帧数，CMU数据集默认是120帧每秒
    - Interpolation ：是否对重映射丢帧的进行插值处理（线性插值）  

# 初期项目，一些功能和细节还有待完善，如有问题请提交issue！
    


