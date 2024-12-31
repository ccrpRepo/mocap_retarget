#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState

class Rerun:
    def __init__(self, motions=None):
        self.motions = motions
        
        
if __name__ == '__main__':
    urdf_path = "../../g1_description/urdf/g1.urdf"
    
    rerun = Rerun(urdf_path)