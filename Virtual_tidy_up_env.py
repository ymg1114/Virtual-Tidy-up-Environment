# -*- coding: utf-8 -*-
import os
import time
import pdb
import pybullet as p
import pybullet_data
# import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
from time import sleep
import gym
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import cv2
# from PIL import Image
from tensorboardX import SummaryWriter
from itertools import permutations
from itertools import combinations
from itertools import groupby
import os.path
from math import pi, atan, tan, sin, cos, sqrt, atan2
from math import degrees as deg
from math import radians as rad
from skimage.draw import polygon
import imutils
from PIL import Image
my_path = os.path.abspath(os.path.dirname(__file__))

max_step = 20

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5,))])   


class sequence_env():
    def __init__(self, seed):
        
        seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.physicsClient = p.connect(p.GUI)
        # self.physicsClient = p.connect(p.DIRECT)
        self._timeStep = 1. / 240.
        self._urdfRoot = pybullet_data.getDataPath()

        self.max_step = max_step

        self.viewMatrix_main = p.computeViewMatrix(
            cameraEyePosition=[0.6, -0.248, 0.69],
            cameraTargetPosition=[0.6, 0.3, 0],
            cameraUpVector=[0, 1, 0])

            
        self.NEAR_PLANE = 0.001
        self.FAR_PLANE = 1000
        self.HFOV = 90.0
        self.VFOV = 59.0


        self.projectionMatrix_main  = p.computeProjectionMatrix(
                                        left=-math.tan(math.pi*self.HFOV/360.0)*self.NEAR_PLANE,
                                        right=math.tan(math.pi*self.HFOV/360.0)*self.NEAR_PLANE,
                                        bottom=-math.tan(math.pi*self.VFOV/360.0)*self.NEAR_PLANE,
                                        top=math.tan(math.pi*self.VFOV/360.0)*self.NEAR_PLANE,
                                        nearVal=self.NEAR_PLANE,
                                        farVal=self.FAR_PLANE)

        self.max_dist = 1e+10


        self.plane = p.loadURDF(my_path + '/object/plane/plane.urdf', [0, 0, -1])
    
    
        self.table = p.loadURDF(my_path + '/object/table/table.urdf', basePosition=[0.6, 0.3, -0.625], baseOrientation=p.getQuaternionFromEuler([0., 0., math.pi]))



    def get_key(self, which_dict, val): 
        for key, value in which_dict.items(): 
            if val == value: 
                return key 
    
        return "key doesn't exist"



    def set_init_parameter(self):
        
        #############################################################################
        # laptop
        self.xpos_laptop_0 = self.xpos_laptop_goal
        self.ypos_laptop_0 = self.ypos_laptop_goal
        self.zori_laptop_0 = self.zori_laptop_goal
        
        self.laptop_init_idx = random.randint(0,1)
        if self.laptop_init_idx:
            self.joint_ang_laptop_0 = math.pi*(1/3) + math.pi*(1/3)*random.random()
        
        else:
            self.joint_ang_laptop_0 = 0.
        #############################################################################


        #############################################################################
        # bin & bin_cover
        self.xpos_bin_cover_0_table = 0.1 + 1.0 * random.random()
        self.ypos_bin_cover_0_table = 0.05 + 0.5 * random.random() # 0.05 + 0.25 * random.random() # 0.1 + 0.15 * random.random()
        self.zori_bin_cover_0_table = math.pi*random.random()
        
        self.bin_bin_cover_init_idx = random.randint(0,1)
        
        if self.bin_bin_cover_init_idx:
            self.xpos_bin_cover_0 = self.xpos_bin_cover_0_table
            self.ypos_bin_cover_0 = self.ypos_bin_cover_0_table
            self.zori_bin_cover_0 = self.zori_bin_cover_0_table
        
        else:
            self.xpos_bin_cover_0 = self.xpos_bin_goal
            self.ypos_bin_cover_0 = self.ypos_bin_goal
            self.zori_bin_cover_0 = self.zori_bin_goal
        #############################################################################
        
        
        #############################################################################
        # drawer
        self.drawer_init_idx = random.randint(0,1)
        self.xpos_drawer_0 = self.xpos_drawer_goal
        self.ypos_drawer_0 = self.ypos_drawer_goal
        
        if self.drawer_init_idx:
            self.joint_ang_drawer0_0 = 0.165
        
        else:
            self.joint_ang_drawer0_0 = 0.
        #############################################################################

        
        #############################################################################
        # tumbler & tumbler_cover
        self.xpos_tumbler_0 = self.xpos_tumbler_goal 
        self.ypos_tumbler_0 = self.ypos_tumbler_goal 
        self.xori_tumbler_0 = math.pi/2
        self.yori_tumbler_0 = 0.  
        self.zori_tumbler_0 = 0.  
        
        self.xpos_tumbler_cover_0 = 0.1 + 1.0 * random.random()
        self.ypos_tumbler_cover_0 = 0.05 + 0.35 * random.random() 
        self.xori_tumbler_cover_0 = math.pi
        self.yori_tumbler_cover_0 = 0.
        self.zori_tumbler_cover_0 = math.pi*random.random()
        #############################################################################
        
        
        #############################################################################
        # lotion
        self.xpos_lotion_0 = 0.1 + 1.0 * random.random()
        self.ypos_lotion_0 = 0.05 + 0.35 * random.random() 
        self.xori_lotion_0 = 0.
        self.yori_lotion_0 = 0.
        self.zori_lotion_0 = math.pi*random.random()
        #############################################################################


    def set_goal_parameter(self):

        # laptop
        self.xpos_laptop_goal = 0.1 + 1.0 * random.random() 
        self.ypos_laptop_goal = 0.3 + 0.3 * random.random() 
        self.zori_laptop_goal = math.pi*(2/3) + math.pi*(2/3)*random.random()
        self.joint_ang_laptop_goal = 0.

        # bin & bin_cover
        self.xpos_bin_goal = 0.1 + 1.0 * random.random() 
        self.ypos_bin_goal = 0.05 + 0.5 * random.random()
        self.zori_bin_goal = math.pi*random.random()
        self.xpos_bin_cover_goal = self.xpos_bin_goal
        self.ypos_bin_cover_goal = self.ypos_bin_goal
        self.zori_bin_cover_goal = self.zori_bin_goal

        # drawer
        self.xpos_drawer_goal = 0.1 + 1.0 * random.random()   
        self.ypos_drawer_goal = 0.35 + 0.15 * random.random() 
        self.zori_drawer_goal = -math.pi*(1/3) + math.pi*(2/3)*random.random()
        self.joint_ang_drawer0_goal = 0.
        self.drawer_joint_idx = random.randint(0,1)
        if self.drawer_joint_idx:
            self.drawer_joint = 4
        else:
            self.drawer_joint = 8


        # tumbler & tumbler_cover
        self.xpos_tumbler_goal = 0.1 + 1.0 * random.random() 
        self.ypos_tumbler_goal = 0.05 + 0.35 * random.random() 
        self.zori_tumbler_goal = 0. 
        self.xpos_tumbler_cover_goal = self.xpos_tumbler_goal 
        self.ypos_tumbler_cover_goal = self.ypos_tumbler_goal
        self.zori_tumbler_cover_goal = math.pi*random.random() 


        # lotion
        self.xpos_lotion_goal_bin = self.xpos_bin_goal        
        self.ypos_lotion_goal_bin = self.ypos_bin_goal        
        self.zpos_lotion_goal_bin = 0.1                      
        self.xori_lotion_goal_bin = 0.
        self.yori_lotion_goal_bin = 0.
        self.zori_lotion_goal_bin = self.zori_bin_goal

        self.xpos_lotion_goal_drawer = self.xpos_drawer_goal + 0.2 * math.sin(self.zori_drawer_goal)  
        self.ypos_lotion_goal_drawer = self.ypos_drawer_goal - 0.2 * math.cos(self.zori_drawer_goal) 
        self.zpos_lotion_goal_drawer = 0.165                                                                 
        self.xpos_lotion_goal_drawer_back = self.xpos_drawer_goal  
        self.ypos_lotion_goal_drawer_back = self.ypos_drawer_goal   
        self.zpos_lotion_goal_drawer_back = 0.08                    
        self.xori_lotion_goal_drawer = 0.
        self.yori_lotion_goal_drawer = 0.
        self.zori_lotion_goal_drawer = self.zori_drawer_goal

        self.xpos_lotion_goal_table = 0.1 + 1.0 * random.random()              
        self.ypos_lotion_goal_table = 0.05 + 0.35 * random.random()        
        self.zpos_lotion_goal_table = 0.05                               
        self.xori_lotion_goal_table = 0.                                  
        self.yori_lotion_goal_table = 0.                                   
        self.zori_lotion_goal_table = math.pi*random.random()                               



    def whether_contact_plane(self):
        if self.laptop:
            IDs_contact_with_laptop, _ = self.get_which_contactID(self.laptop)
            pos_laptop, _ = p.getBasePositionAndOrientation(self.laptop)
        else:
            IDs_contact_with_laptop = [None]
            pos_laptop = [100,100,100]
        
        
        if self.bin:
            IDs_contact_with_bin, _ = self.get_which_contactID(self.bin)
            IDs_contact_with_bin_cover, _ = self.get_which_contactID(self.bin_cover)
            pos_bin, _ = p.getBasePositionAndOrientation(self.bin)
            pos_bin_cover, _ = p.getBasePositionAndOrientation(self.bin_cover)
        else:
            IDs_contact_with_bin = [None]
            IDs_contact_with_bin_cover = [None]
            pos_bin = [100,100,100]
            pos_bin_cover = [100,100,100]
            
            
        if self.drawer:
            IDs_contact_with_drawer, _ = self.get_which_contactID(self.drawer)
            pos_drawer, _ = p.getBasePositionAndOrientation(self.drawer)
        else:
            IDs_contact_with_drawer = [None]
            pos_drawer = [100,100,100]
        
        
        if self.tumbler:
            IDs_contact_with_tumbler, _ = self.get_which_contactID(self.tumbler)
            IDs_contact_with_tumbler_cover, _ = self.get_which_contactID(self.tumbler_cover)
            pos_tumbler, _ = p.getBasePositionAndOrientation(self.tumbler)
            pos_tumbler_cover, _ = p.getBasePositionAndOrientation(self.tumbler_cover)
        else:
            IDs_contact_with_tumbler = [None]
            IDs_contact_with_tumbler_cover = [None]
            pos_tumbler = [100,100,100]
            pos_tumbler_cover = [100,100,100]
        
        
        if self.lotion:
            IDs_contact_with_lotion, _ = self.get_which_contactID(self.lotion)
            pos_lotion, _ = p.getBasePositionAndOrientation(self.lotion)
        else:
            IDs_contact_with_lotion = [None]
            pos_lotion = [100,100,100]
        
        
        contact_check_IDs = set(IDs_contact_with_laptop + IDs_contact_with_bin + IDs_contact_with_bin_cover + IDs_contact_with_drawer + IDs_contact_with_tumbler + IDs_contact_with_tumbler_cover + IDs_contact_with_lotion)

        pos_z = set([pos_laptop[2]] + [pos_drawer[2]] + [pos_bin[2]] + [pos_bin_cover[2]] + [pos_tumbler[2]] + [pos_tumbler_cover[2]] + [pos_lotion[2]])

        print('self.plane :', self.plane)
        print('pos_z :', pos_z)
        

        if 0 in contact_check_IDs:
            print('Bad pos-Z !, Epi. is terminated @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            done_z = True
        else:
            print('Good pos-Z !, Epi. is keep going @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            done_z = False


        for i in pos_z:
            if i < -0.1:
                done_z = True 
                print('Bad pos-Z !~!@#$ !~!@#$ !~!@#$')

        return done_z

                
    def random_init_order(self):

        random.shuffle(self.list_of_objs)

        for i in self.list_of_objs:
            if i == 2:
                self.generate_init_bin_cover()
            
            elif i == 3:
                self.generate_init_tumbler()
                self.generate_init_tumbler_cover()
            
            elif i == 4:
                self.generate_init_lotion()


                
    # 0 -> laptop
    # 1 -> drawer
    # 2 -> bin & bin_cover
    # 3 -> tumbler & tumbler_cover
    # 4 -> lotion
    def generate_init(self, n_epi):
        if self.laptop:
            self.generate_init_laptop()
    
        self.random_init_order()
        
        if self.drawer:
            self.generate_init_drawer()

        if self.is_drawer == 'goal' and self.is_bin == 'goal' and (not self.lotion_org_loc == "table" ) and (not self.is_lotion == None):  
            self.bin_bin_cover_disabled = 0.
            self.drawer_disabled = 0.
            print('Both Drawer & Bin goal is Closed !')
            print('Lotion goal is not Table !')
            print('and Lotion is alive !')
            
        elif self.is_drawer == None and self.is_bin == 'goal' and (self.lotion_org_loc == 'bin' ) and (not self.is_lotion == None):
            self.bin_bin_cover_disabled = 0.
            print('Bin goal is Closed !')
            print('NO Drawer !')
            print('Lotion goal is Bin !')
            print('and Lotion is alive !')


        elif self.is_drawer == 'goal' and self.is_bin == None and (self.lotion_org_loc == 'drawer' ) and (not self.is_lotion == None):
            self.drawer_disabled = 0.
            print('Drawer goal is Closed !')
            print('NO Bin !')
            print('Lotion goal is Drawer !')
            print('and Lotion is alive !')



    def generate_init_laptop(self):
        p.resetJointState(self.laptop, 0, targetValue=self.joint_ang_laptop_0)
        
   
        self.pos_laptop_0, self.ori_laptop_0 = p.getBasePositionAndOrientation(self.laptop)   
        pos = self.pos_laptop_0
        ori = p.getEulerFromQuaternion(self.ori_laptop_0)

        self.xpos_laptop_0 = pos[0]  
        self.ypos_laptop_0 = pos[1] 
        self.zpos_laptop_0 = pos[2]

        self.xori_laptop_0 = ori[0]
        self.yori_laptop_0 = ori[1]
        self.zori_laptop_0 = ori[2]
        
        
        
    def generate_init_bin_cover(self):
        p.resetBasePositionAndOrientation(self.bin_cover, [self.xpos_bin_cover_0, self.ypos_bin_cover_0, 0.2], p.getQuaternionFromEuler([0, 0, self.zori_bin_cover_0]))
        

        # p.changeVisualShape(self.bin_cover, linkIndex = -1, rgbaColor=[0., 0., 1., 1.])       
        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT
            
        self.pos_bin_cover_0, self.ori_bin_cover_0 = p.getBasePositionAndOrientation(self.bin_cover)   
        pos = self.pos_bin_cover_0
        ori = p.getEulerFromQuaternion(self.ori_bin_cover_0)

        self.xpos_bin_cover_0 = pos[0]  
        self.ypos_bin_cover_0 = pos[1] 
        self.zpos_bin_cover_0 = pos[2]

        self.xori_bin_cover_0 = ori[0]
        self.yori_bin_cover_0 = ori[1]
        self.zori_bin_cover_0 = ori[2]
        

    def generate_init_drawer(self):
        p.resetJointState(self.drawer, self.drawer_joint, targetValue=self.joint_ang_drawer0_0)
            
        self.pos_drawer_0, self.ori_drawer_0 = p.getBasePositionAndOrientation(self.drawer)   
        pos = self.pos_drawer_0
        ori = p.getEulerFromQuaternion(self.ori_drawer_0)

        self.xpos_drawer_0 = pos[0]  
        self.ypos_drawer_0 = pos[1] 
        self.zpos_drawer_0 = pos[2]

        self.xori_drawer_0 = ori[0]
        self.yori_drawer_0 = ori[1]
        self.zori_drawer_0 = ori[2]
        
        
    def generate_init_tumbler(self):
        p.resetBasePositionAndOrientation(self.tumbler, [self.xpos_tumbler_0, self.ypos_tumbler_0, 0.05], p.getQuaternionFromEuler([self.xori_tumbler_0, self.yori_tumbler_0, self.zori_tumbler_0]))
        
        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT
 
        self.pos_tumbler_0, self.ori_tumbler_0 = p.getBasePositionAndOrientation(self.tumbler)   
        pos = self.pos_tumbler_0
        ori = p.getEulerFromQuaternion(self.ori_tumbler_0)

        self.xpos_tumbler_0 = pos[0]  
        self.ypos_tumbler_0 = pos[1] 
        self.zpos_tumbler_0 = pos[2]

        self.xori_tumbler_0 = ori[0]
        self.yori_tumbler_0 = ori[1]
        self.zori_tumbler_0 = ori[2]
        


    def generate_init_tumbler_cover(self):
        p.resetBasePositionAndOrientation(self.tumbler_cover, [self.xpos_tumbler_cover_0, self.ypos_tumbler_cover_0, 0.05], p.getQuaternionFromEuler([self.xori_tumbler_cover_0, self.yori_tumbler_cover_0, self.zori_tumbler_cover_0]))

        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT
            
        self.pos_tumbler_cover_0, self.ori_tumbler_cover_0 = p.getBasePositionAndOrientation(self.tumbler)   
        pos = self.pos_tumbler_cover_0
        ori = p.getEulerFromQuaternion(self.ori_tumbler_cover_0)

        self.xpos_tumbler_cover_0 = pos[0]  
        self.ypos_tumbler_cover_0 = pos[1] 
        self.zpos_tumbler_cover_0 = pos[2]

        self.xori_tumbler_cover_0 = ori[0]
        self.yori_tumbler_cover_0 = ori[1]
        self.zori_tumbler_cover_0 = ori[2]


    def generate_init_lotion(self):
        p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_0, self.ypos_lotion_0, 0.02], p.getQuaternionFromEuler([self.xori_lotion_0, self.yori_lotion_0, self.zori_lotion_0]))
        self.is_lotion = 'init'
    
        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT
            
        self.pos_lotion_0, self.ori_lotion_0 = p.getBasePositionAndOrientation(self.lotion)   
        pos = self.pos_lotion_0
        ori = p.getEulerFromQuaternion(self.ori_lotion_0)

        self.xpos_lotion_0 = pos[0]  
        self.ypos_lotion_0 = pos[1] 
        self.zpos_lotion_0 = pos[2]

        self.xori_lotion_0 = ori[0]
        self.yori_lotion_0 = ori[1]
        self.zori_lotion_0 = ori[2]




    def list_of_objs_create(self, how_many):
        lists = []
        for i in range(how_many):
            lists.append(i)
        return lists

    
    # 0 -> laptop
    # 1 -> drawer
    # 2 -> bin & bin_cover
    # 3 -> tumbler & tumbler_cover
    # 4 -> lotion
    def generate_goal(self):
        
        for i in self.list_of_objs:
            
            if i == 0:
                self.generate_goal_laptop()

            elif i == 1:
                self.generate_goal_drawer()
 
                if (self.allocate_lotion_goal() == "3" and 0.33 <= self.org_flag < 0.66) or (self.allocate_lotion_goal() == "5" and self.org_flag < 0.5):
                    self.lotion_org_loc = "drawer"   
                    self.is_lotion = 'drawer'                      
                    self.generate_goal_lotion(self.lotion_org_loc)
                    
                    print(11111)
                    print("self.lotion :", self.lotion)
                    print("self.org_flag :", self.org_flag)
                    print("self.allocate_lotion_goal() :", self.allocate_lotion_goal())
                
                self.generate_goal_drawer_step()
                
            elif i == 2:
                self.generate_goal_bin()

                if (self.allocate_lotion_goal() == "2" and self.org_flag < 0.5) or (self.allocate_lotion_goal() == "3" and self.org_flag < 0.33):
                    self.lotion_org_loc = "bin"   
                    self.is_lotion = 'bin'                    
                    self.generate_goal_lotion(self.lotion_org_loc)
                    
                    print(22222)
                    print("self.lotion :", self.lotion)
                    print("self.org_flag :", self.org_flag)
                    print("self.allocate_lotion_goal() :", self.allocate_lotion_goal())
                
                self.generate_goal_bin_cover()
            
            elif i == 3:
                self.generate_goal_tumbler()
                self.generate_goal_tumbler_cover()
            
            
            elif i == 4:

                if (self.allocate_lotion_goal() == "2" and 0.5 <= self.org_flag) or (self.allocate_lotion_goal() == "3" and 0.66 <= self.org_flag) or (self.allocate_lotion_goal() == "5" and 0.5 <= self.org_flag) or (self.allocate_lotion_goal() == "6"):
                    self.lotion_org_loc = "table"  
                    self.is_lotion = 'table' 
                    self.generate_goal_lotion(self.lotion_org_loc)
                    
                    print(44444)
                    print("self.lotion :", self.lotion)
                    print("self.org_flag :", self.org_flag)
                    print("self.allocate_lotion_goal() :", self.allocate_lotion_goal())

    
    def generate_goal_laptop(self):
        
        self.laptop = p.loadURDF(my_path + '/object/laptop_2.urdf', basePosition=[self.xpos_laptop_goal, self.ypos_laptop_goal, 0.01], baseOrientation=p.getQuaternionFromEuler([0., 0, self.zori_laptop_goal]), useFixedBase=True, globalScaling=self.laptop_gscale)
        
        p.resetJointState(self.laptop, 0, targetValue=self.joint_ang_laptop_goal)
         
        self.pos_laptop_goal, self.ori_laptop_goal = p.getBasePositionAndOrientation(self.laptop)   
        pos = self.pos_laptop_goal
        ori = p.getEulerFromQuaternion(self.ori_laptop_goal)

        self.xpos_laptop_goal = pos[0]  
        self.ypos_laptop_goal = pos[1] 
        self.zpos_laptop_goal = pos[2]

        self.xori_laptop_goal = ori[0]
        self.yori_laptop_goal = ori[1]
        self.zori_laptop_goal = ori[2]
        
        
    def generate_goal_laptop_step(self):
        
        p.resetJointState(self.laptop, 0, targetValue=self.joint_ang_laptop_goal)     
        
        for _ in range(250):
            p.stepSimulation()

        
    def generate_goal_bin(self):
        
        self.bin = p.loadURDF(my_path + '/object/bin.urdf', basePosition=[self.xpos_bin_goal, self.ypos_bin_goal, 0.001], baseOrientation=p.getQuaternionFromEuler([0, 0, self.zori_bin_goal]), useFixedBase=True, globalScaling=self.bin_gscale)
            
        self.pos_bin_goal, self.ori_bin_goal = p.getBasePositionAndOrientation(self.bin)   
        pos = self.pos_bin_goal
        ori = p.getEulerFromQuaternion(self.ori_bin_goal)

        self.xpos_bin_goal = pos[0]  
        self.ypos_bin_goal = pos[1] 
        self.zpos_bin_goal = pos[2]

        self.xori_bin_goal = ori[0]
        self.yori_bin_goal = ori[1]
        self.zori_bin_goal = ori[2]
        
  
        
    def generate_goal_bin_cover(self):
        
        self.bin_cover = p.loadURDF(my_path + '/object/bin_cover.urdf', basePosition=[self.xpos_bin_cover_goal, self.ypos_bin_cover_goal, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, self.zori_bin_cover_goal]), globalScaling=self.bin_gscale)
            
        self.pos_bin_cover_goal, self.ori_bin_cover_goal = p.getBasePositionAndOrientation(self.bin_cover)   
        pos = self.pos_bin_cover_goal
        ori = p.getEulerFromQuaternion(self.ori_bin_cover_goal)

        self.xpos_bin_cover_goal = pos[0]  
        self.ypos_bin_cover_goal = pos[1] 
        self.zpos_bin_cover_goal = pos[2]

        self.xori_bin_cover_goal = ori[0]
        self.yori_bin_cover_goal = ori[1]
        self.zori_bin_cover_goal = ori[2]
        
        
    def make_step_action_z_up(self, pos):
        
        corrected = []
        corrected.append(pos[0])
        corrected.append(pos[1])
        corrected.append(pos[2]+0.17)

        return corrected
        
  
        
    def generate_goal_bin_bin_cover_step(self):
        # p.resetBasePositionAndOrientation(self.bin_cover, self.pos_bin_cover_goal, self.ori_bin_cover_goal)
        p.resetBasePositionAndOrientation(self.bin_cover, self.make_step_action_z_up(self.pos_bin_cover_goal), self.ori_bin_cover_goal)
        # p.changeVisualShape(self.bin, linkIndex = -1, rgbaColor=[0., 0., 1., 1.])       
        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT

    def generate_goal_bin_bin_cover_step_open(self):
        p.resetBasePositionAndOrientation(self.bin_cover, [self.xpos_bin_cover_0_table, self.ypos_bin_cover_0_table, 0.2], p.getQuaternionFromEuler([0,0,self.zori_bin_cover_0_table]))
        # p.changeVisualShape(self.bin, linkIndex = -1, rgbaColor=[0., 0., 1., 1.])       
        for _ in range(250):
            p.stepSimulation()

    def generate_goal_drawer(self):
        
        self.drawer = p.loadURDF(my_path + '/object/drawer_2.urdf', basePosition=[self.xpos_drawer_goal, self.ypos_drawer_goal, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, self.zori_drawer_goal]), useFixedBase=True, globalScaling=self.drawer_gscale)

        p.resetJointState(self.drawer, self.drawer_joint, targetValue=0.165)

            
        self.pos_drawer_goal, self.ori_drawer_goal = p.getBasePositionAndOrientation(self.drawer)   
        pos = self.pos_drawer_goal
        ori = p.getEulerFromQuaternion(self.ori_drawer_goal)

        self.xpos_drawer_goal = pos[0]  
        self.ypos_drawer_goal = pos[1] 
        self.zpos_drawer_goal = pos[2]

        self.xori_drawer_goal = ori[0]
        self.yori_drawer_goal = ori[1]
        self.zori_drawer_goal = ori[2]
        

    def generate_goal_drawer_step(self):
        if self.is_lotion == 'drawer':
            p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_goal_drawer_back, self.ypos_lotion_goal_drawer_back, self.zpos_lotion_goal_drawer_back], p.getQuaternionFromEuler([self.xori_lotion_goal_drawer, self.yori_lotion_goal_drawer, self.zori_lotion_goal_drawer]))
            p.resetJointState(self.drawer, self.drawer_joint, targetValue=self.joint_ang_drawer0_goal)
    
        
        else:
            p.resetJointState(self.drawer, self.drawer_joint, targetValue=self.joint_ang_drawer0_goal)
            

        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT




    def generate_goal_drawer_step_open(self):

        p.resetJointState(self.drawer, self.drawer_joint, targetValue=0.165)
            
            
        for _ in range(250):
            p.stepSimulation()


        
    def generate_goal_tumbler(self):
        self.tumbler = p.loadURDF(my_path + '/object/tumbler.urdf', basePosition=[self.xpos_tumbler_goal, self.ypos_tumbler_goal, 0.01], baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, self.zori_tumbler_goal]), globalScaling=self.tumbler_gscale)
        # p.changeVisualShape(self.tumbler, linkIndex = -1, rgbaColor=[1., 0., 0., 1.])       
        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT
  
            
        self.pos_tumbler_goal, self.ori_tumbler_goal = p.getBasePositionAndOrientation(self.tumbler)   
        pos = self.pos_tumbler_goal
        ori = p.getEulerFromQuaternion(self.ori_tumbler_goal)

        self.xpos_tumbler_goal = pos[0]  
        self.ypos_tumbler_goal = pos[1] 
        self.zpos_tumbler_goal = pos[2]

        self.xori_tumbler_goal = ori[0]
        self.yori_tumbler_goal = ori[1]
        self.zori_tumbler_goal = ori[2]
        

        
    def generate_goal_tumbler_cover(self):
        self.tumbler_cover = p.loadURDF(my_path + '/object/tumbler_cover.urdf', basePosition=[self.xpos_tumbler_goal, self.ypos_tumbler_goal, 0.13], baseOrientation=p.getQuaternionFromEuler([math.pi, 0, self.zori_tumbler_cover_goal]), globalScaling=self.tumbler_gscale)

            
        self.pos_tumbler_cover_goal, self.ori_tumbler_cover_goal = p.getBasePositionAndOrientation(self.tumbler_cover)   
        pos = self.pos_tumbler_cover_goal
        ori = p.getEulerFromQuaternion(self.ori_tumbler_cover_goal)

        self.xpos_tumbler_cover_goal = pos[0]  
        self.ypos_tumbler_cover_goal = pos[1] 
        self.zpos_tumbler_cover_goal = pos[2]

        self.xori_tumbler_cover_goal = ori[0]
        self.yori_tumbler_cover_goal = ori[1]
        self.zori_tumbler_cover_goal = ori[2]


        
    def generate_goal_tumbler_step(self):
        p.resetBasePositionAndOrientation(self.tumbler, self.pos_tumbler_goal, self.ori_tumbler_goal)
        p.resetBasePositionAndOrientation(self.tumbler_cover, self.pos_tumbler_cover_goal, self.ori_tumbler_cover_goal)
        # p.changeVisualShape(self.bin, linkIndex = -1, rgbaColor=[0., 0., 1., 1.])       
        for _ in range(250):
            p.stepSimulation()
    
        # FOR AVOIDING WRONG POS/ORI MEASUREMENT
 
        


    def generate_goal_lotion(self, lotion_org_loc):         
        
        
        if lotion_org_loc == 'table':          
            self.lotion = p.loadURDF(my_path + '/object/lotion.urdf', basePosition=[self.xpos_lotion_goal_table, self.ypos_lotion_goal_table, self.zpos_lotion_goal_table], baseOrientation=p.getQuaternionFromEuler([self.xori_lotion_goal_table, self.yori_lotion_goal_table, self.zori_lotion_goal_table]), globalScaling=self.lotion_gscale)
            # p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_goal_table, self.ypos_lotion_goal_table, self.zpos_lotion_goal_table], p.getQuaternionFromEuler([self.xori_lotion_goal_table, self.yori_lotion_goal_table, self.zori_lotion_goal_table]))
            # p.changeVisualShape(self.lotion, linkIndex = -1, rgbaColor=[1., 0., 0., 1.])       
            for _ in range(250):
                p.stepSimulation()

            self.pos_lotion_goal, self.ori_lotion_goal = p.getBasePositionAndOrientation(self.lotion)   
            pos = self.pos_lotion_goal
            ori = p.getEulerFromQuaternion(self.ori_lotion_goal)

            self.xpos_lotion_goal_table = pos[0]  
            self.ypos_lotion_goal_table = pos[1] 
            self.zpos_lotion_goal_table = pos[2]

            self.xori_lotion_goal_table = ori[0]
            self.yori_lotion_goal_table = ori[1]
            self.zori_lotion_goal_table = ori[2]
        
            self.xpos_lotion_goal = self.xpos_lotion_goal_table
            self.ypos_lotion_goal = self.ypos_lotion_goal_table
            self.zpos_lotion_goal = self.zpos_lotion_goal_table            
            self.xori_lotion_goal = self.xori_lotion_goal_table          
            self.yori_lotion_goal = self.yori_lotion_goal_table            
            self.zori_lotion_goal = self.zori_lotion_goal_table
        
            # FOR AVOIDING WRONG POS/ORI MEASUREMENT
     
        
        elif lotion_org_loc == 'bin': 
            self.lotion = p.loadURDF(my_path + '/object/lotion.urdf', basePosition=[self.xpos_lotion_goal_bin, self.ypos_lotion_goal_bin, self.zpos_lotion_goal_bin], baseOrientation=p.getQuaternionFromEuler([self.xori_lotion_goal_bin, self.yori_lotion_goal_bin, self.zori_lotion_goal_bin]), globalScaling=self.lotion_gscale)
            # p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_goal_bin, self.ypos_lotion_goal_bin, self.zpos_lotion_goal_bin], p.getQuaternionFromEuler([self.xori_lotion_goal_bin, self.yori_lotion_goal_bin, self.zori_lotion_goal_bin]))
            # p.changeVisualShape(self.lotion, linkIndex = -1, rgbaColor=[1., 0., 0., 1.])       
            for _ in range(250):
                p.stepSimulation()
        
            # FOR AVOIDING WRONG POS/ORI MEASUREMENT

            self.pos_lotion_goal, self.ori_lotion_goal = p.getBasePositionAndOrientation(self.lotion)   
            pos = self.pos_lotion_goal
            ori = p.getEulerFromQuaternion(self.ori_lotion_goal)

            self.xpos_lotion_goal_bin = pos[0]  
            self.ypos_lotion_goal_bin = pos[1] 
            self.zpos_lotion_goal_bin = pos[2]

            self.xori_lotion_goal_bin = ori[0]
            self.yori_lotion_goal_bin = ori[1]
            self.zori_lotion_goal_bin = ori[2]

            self.xpos_lotion_goal = self.xpos_lotion_goal_bin
            self.ypos_lotion_goal = self.ypos_lotion_goal_bin
            self.zpos_lotion_goal = self.zpos_lotion_goal_bin            
            self.xori_lotion_goal = self.xori_lotion_goal_bin          
            self.yori_lotion_goal = self.yori_lotion_goal_bin            
            self.zori_lotion_goal = self.zori_lotion_goal_bin


        elif lotion_org_loc == 'drawer':      
            self.lotion = p.loadURDF(my_path + '/object/lotion.urdf', basePosition=[self.xpos_lotion_goal_drawer, self.ypos_lotion_goal_drawer, self.zpos_lotion_goal_drawer], baseOrientation=p.getQuaternionFromEuler([self.xori_lotion_goal_drawer, self.yori_lotion_goal_drawer, self.zori_lotion_goal_drawer]), globalScaling=self.lotion_gscale)
            # p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_goal_drawer, self.ypos_lotion_goal_drawer, self.zpos_lotion_goal_drawer], p.getQuaternionFromEuler([self.xori_lotion_goal_drawer, self.yori_lotion_goal_drawer, self.zori_lotion_goal_drawer]))
            # p.changeVisualShape(self.lotion, linkIndex = -1, rgbaColor=[1., 0., 0., 1.])       
            for _ in range(250):
                p.stepSimulation()

            self.pos_lotion_goal, self.ori_lotion_goal = p.getBasePositionAndOrientation(self.lotion)   
            pos = self.pos_lotion_goal
            ori = p.getEulerFromQuaternion(self.ori_lotion_goal)

            self.xpos_lotion_goal_drawer = pos[0]  
            self.ypos_lotion_goal_drawer = pos[1] 
            self.zpos_lotion_goal_drawer = pos[2]

            self.xori_lotion_goal_drawer = ori[0]
            self.yori_lotion_goal_drawer = ori[1]
            self.zori_lotion_goal_drawer = ori[2]


            self.xpos_lotion_goal = self.xpos_lotion_goal_drawer
            self.ypos_lotion_goal = self.ypos_lotion_goal_drawer
            self.zpos_lotion_goal = self.zpos_lotion_goal_drawer            
            self.xori_lotion_goal = self.xori_lotion_goal_drawer          
            self.yori_lotion_goal = self.yori_lotion_goal_drawer            
            self.zori_lotion_goal = self.zori_lotion_goal_drawer

    

    def generate_goal_lotion_step(self, lotion_org_loc):         
        if lotion_org_loc == 'table':          
            p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_goal_table, self.ypos_lotion_goal_table, self.zpos_lotion_goal_table], p.getQuaternionFromEuler([self.xori_lotion_goal_table, self.yori_lotion_goal_table, self.zori_lotion_goal_table]))
            # p.changeVisualShape(self.lotion, linkIndex = -1, rgbaColor=[1., 0., 0., 1.])       
            for _ in range(250):
                p.stepSimulation()
        
            # FOR AVOIDING WRONG POS/ORI MEASUREMENT
 
        
        elif lotion_org_loc == 'bin': 
            p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_goal_bin, self.ypos_lotion_goal_bin, self.zpos_lotion_goal_bin+0.15], p.getQuaternionFromEuler([self.xori_lotion_goal_bin, self.yori_lotion_goal_bin, self.zori_lotion_goal_bin]))
            # p.changeVisualShape(self.lotion, linkIndex = -1, rgbaColor=[1., 0., 0., 1.])       
            for _ in range(250):
                p.stepSimulation()
        
            # FOR AVOIDING WRONG POS/ORI MEASUREMENT


        elif lotion_org_loc == 'drawer':      
            p.resetBasePositionAndOrientation(self.lotion, [self.xpos_lotion_goal_drawer, self.ypos_lotion_goal_drawer, self.zpos_lotion_goal_drawer], p.getQuaternionFromEuler([self.xori_lotion_goal_drawer, self.yori_lotion_goal_drawer, self.zori_lotion_goal_drawer]))
            # p.changeVisualShape(self.lotion, linkIndex = -1, rgbaColor=[1., 0., 0., 1.])       
            for _ in range(250):
                p.stepSimulation()
        



    # True -> bad
    # False -> good
    def ensure_not_overlap_init_2(self):
        if self.bin:
            self.init_bin_pos, _ = p.getBasePositionAndOrientation(self.bin)
            self.init_bin_cover_pos, _ = p.getBasePositionAndOrientation(self.bin_cover)
        else:
            self.init_bin_pos = [-10., -10., -10.]
            self.init_bin_cover_pos = [-10., -10., -10.]
        
        if self.tumbler:
            self.init_tumbler_pos, _ = p.getBasePositionAndOrientation(self.tumbler)
            self.init_tumbler_cover_pos, _ = p.getBasePositionAndOrientation(self.tumbler_cover)
        else:
            self.init_tumbler_pos = [-10., -10., -10.]
            self.init_tumbler_cover_pos = [-10., -10., -10.]
            
        if self.lotion:
            self.init_lotion_pos, _ = p.getBasePositionAndOrientation(self.lotion)
        else:
            self.init_lotion_pos = [-10., -10., -10.]

        if self.init_bin_pos[2] > 0.15 or self.init_bin_cover_pos[2] > 0.15 or self.init_tumbler_pos[2] > 0.15 or self.init_tumbler_cover_pos[2] > 0.15 or self.init_lotion_pos[2] > 0.15:
            return True  # bad
        else:
            return False # good
        






    # True -> bad
    # False -> good
    def ensure_not_overlap_init(self):
        if self.bin:
            flag = self.ensure_bin_init(self.bin)        # True -> bad
        else:
            flag = False
        
        if self.drawer:
            flag2 = self.ensure_drawer_init(self.drawer) # True -> bad
        else:
            flag2 = False
        
        if self.laptop:
            flag3 = self.ensure_laptop_init(self.laptop) # True -> bad
        else:
            flag3 = False

        return flag or flag2 or flag3
        # return flag and flag2
        
        
    def ensure_bin_init(self, obj):
        bin_thres = 0.08
        
        if self.tumbler:
            result = p.getClosestPoints(obj, self.tumbler, self.max_dist, -1, -1)
            flag1, _, _, _, _, _, _, _, dist1, _, _, _, _, _, = result[0]

            result = p.getClosestPoints(obj, self.tumbler_cover, self.max_dist, -1, -1)
            flag2, _, _, _, _, _, _, _, dist2, _, _, _, _, _, = result[0]
        else:
            flag1, flag2 = False, False
            dist1, dist2 = 3., 3
           

        if self.lotion:
            result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
            flag3, _, _, _, _, _, _, _, dist3, _, _, _, _, _, = result[0]
        else:
            flag3 = False
            dist3 = 3.

        if flag1 or flag2 or flag3 :
            flag = True # bad
        else:
            flag = False  # good

        if dist1 < bin_thres or dist2 < bin_thres or dist3 < bin_thres:
            flag_d = True  # bad
        else:
            flag_d = False # good

        return flag or flag_d
        # return flag_d
        


    def ensure_drawer_init(self, obj):
        drawer_thres = 0.08
        
        if self.bin:
            result = p.getClosestPoints(obj, self.bin, self.max_dist, -1, -1)
            flag1, _, _, _, _, _, _, _, dist1, _, _, _, _, _, = result[0]
            result = p.getClosestPoints(obj, self.bin_cover, self.max_dist, -1, -1)
            flag2, _, _, _, _, _, _, _, dist2, _, _, _, _, _, = result[0]
        else:
            flag1, flag2 = False, False
            dist1, dist2 = 10., 10.
        
        if self.tumbler:
            result = p.getClosestPoints(obj, self.tumbler, self.max_dist, -1, -1)
            flag3, _, _, _, _, _, _, _, dist3, _, _, _, _, _, = result[0]
            result = p.getClosestPoints(obj, self.tumbler_cover, self.max_dist, -1, -1)
            flag4, _, _, _, _, _, _, _, dist4, _, _, _, _, _, = result[0]
        else:
            flag3, flag4 = False, False
            dist3, dist4 = 10., 10.

        if self.lotion:
            result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
            flag5, _, _, _, _, _, _, _, dist5, _, _, _, _, _, = result[0]

        else:
            flag5 = False
            dist5 = 10.

        if self.laptop:
            result = p.getClosestPoints(obj, self.laptop, self.max_dist, -1, -1)
            flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = False
            dist6 = 10.

                    
        if flag1 or flag2 or flag3 or flag4 or flag5 or flag6:
            flag = True # bad
        else:
            flag = False  # good

        if dist1 < drawer_thres or dist2 < drawer_thres or dist3 < drawer_thres or dist4 < drawer_thres or dist5 < drawer_thres or dist6 < drawer_thres:
            flag_d = True  # bad
        else:
            flag_d = False # good

        return flag or flag_d
        # return flag_d
        
        
    def ensure_laptop_init(self, obj):
        laptop_thres = 0.02
        
        if self.bin:
            result = p.getClosestPoints(obj, self.bin, self.max_dist, -1, -1)
            flag1, _, _, _, _, _, _, _, dist1, _, _, _, _, _, = result[0]
        else:
            flag1 = False
            dist1 = 10.
    

        if self.drawer:
            result = p.getClosestPoints(obj, self.drawer, self.max_dist, -1, -1)
            flag2, _, _, _, _, _, _, _, dist2, _, _, _, _, _, = result[0]
        else:
            flag2 = False
            dist2 = 10.
    
            
        if flag1 or flag2:
            flag = True # bad
        else:
            flag = False  # good

        if dist1 < laptop_thres or dist2 < laptop_thres:
            flag_d = True  # bad
        else:
            flag_d = False # good

        return flag or flag_d
        # return flag_d
        
            

    def ensure_laptop_step(self, obj):
        laptop_thres = 0.02
        
        if self.bin:
            result = p.getClosestPoints(obj, self.bin, self.max_dist, -1, -1)
            flag1, _, _, _, _, _, _, _, dist1, _, _, _, _, _, = result[0]

            result = p.getClosestPoints(obj, self.bin_cover, self.max_dist, -1, -1)
            flag2, _, _, _, _, _, _, _, dist2, _, _, _, _, _, = result[0]
        else:
            flag1, flag2 = False, False
            dist1, dist2 = 3., 3.
        
        if self.drawer:
            result = p.getClosestPoints(obj, self.drawer, self.max_dist, -1, -1)
            flag3, _, _, _, _, _, _, _, dist3, _, _, _, _, _, = result[0]
        else:
            flag3 = False
            dist3 = 3.
        
        if self.tumbler:
            result = p.getClosestPoints(obj, self.tumbler, self.max_dist, -1, -1)
            flag4, _, _, _, _, _, _, _, dist4, _, _, _, _, _, = result[0]
            
            result = p.getClosestPoints(obj, self.tumbler_cover, self.max_dist, -1, -1)
            flag5, _, _, _, _, _, _, _, dist5, _, _, _, _, _, = result[0]
        else:
            flag4, flag5 = False, False
            dist4, dist5 = 3., 3.
        
        if self.lotion:
            result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
            flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = False
            dist6 = 3.
        
        
        if flag1 or flag2 or flag3 or flag4 or flag5 or flag6:
            flag = False # bad
        else:
            flag = True  # good
        
        if dist1 > laptop_thres and dist2 > laptop_thres and dist3 > laptop_thres and dist4 > laptop_thres and dist5 > laptop_thres and dist6 > laptop_thres:
            flag_d = True # good
        else:
            flag_d = False # good

        return flag and flag_d

        
        
        
        
    # True -> good
    # False -> bad
    def ensure_not_overlap_goal(self):
        if self.laptop:
            flag1 = self.ensure_laptop(self.laptop, status="goal")
        else:
            flag1 = True
        
        if self.bin:
            flag2 = self.ensure_bin(self.bin, status="goal")
            flag3 = self.ensure_bin_cov(self.bin_cover, status="goal")
        else:
            flag2 = True
            flag3 = True
            
        if self.drawer:
            flag4 = self.ensure_drawer(self.drawer, status="goal")
        else:
            flag4 = True
            
        if self.tumbler:
            flag5 = self.ensure_tumbler(self.tumbler, status="goal")
            flag6 = self.ensure_tumbler_cover(self.tumbler_cover, status="goal")
        else:
            flag5 = True
            flag6 = True

        return flag1 and flag2 and flag3 and flag4 and flag5 and flag6 
    
    
    def ensure_laptop(self, obj, status="init"):
        laptop_thres = 0.02
        if self.bin:
            result = p.getClosestPoints(obj, self.bin, self.max_dist, -1, -1)
            flag1, _, _, _, _, _, _, _, dist1, _, _, _, _, _, = result[0]
            result = p.getClosestPoints(obj, self.bin_cover, self.max_dist, -1, -1)
            flag2, _, _, _, _, _, _, _, dist2, _, _, _, _, _, = result[0]
        else:
            flag1, flag2 = 0, 0
            dist1, dist2 = 3., 3.

        if self.drawer:
            result = p.getClosestPoints(obj, self.drawer, self.max_dist, -1, -1)
            flag3, _, _, _, _, _, _, _, dist3, _, _, _, _, _, = result[0]
        else:
            flag3 = 0
            dist3 = 3.
            
        if self.tumbler:
            result = p.getClosestPoints(obj, self.tumbler, self.max_dist, -1, -1)
            flag4, _, _, _, _, _, _, _, dist4, _, _, _, _, _, = result[0]
            result = p.getClosestPoints(obj, self.tumbler_cover, self.max_dist, -1, -1)
            flag5, _, _, _, _, _, _, _, dist5, _, _, _, _, _, = result[0]
        else:
            flag4, flag5 = 0, 0
            dist4, dist5 = 3., 3.

        if self.lotion:
            result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
            flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = 0
            dist6 = 3.
            
        
        if flag1 or flag2 or flag3 or flag4 or flag5 or flag6:
            flag = False # bad
        else:
            flag = True  # good
        
        if dist1 > laptop_thres and dist2 > laptop_thres and dist3 > laptop_thres and dist4 > laptop_thres and dist5 > laptop_thres and dist6 > laptop_thres:
            flag_d = True # good
        else:
            flag_d = False # good

        return flag and flag_d



    def ensure_bin(self, obj, status="init"):
        bin_thres = 0.02
        
        if self.drawer:
            result = p.getClosestPoints(obj, self.drawer, self.max_dist, -1, -1)
            flag3, _, _, _, _, _, _, _, dist3, _, _, _, _, _, = result[0]
        else:
            flag3 = 0
            dist3 = 3.
        
        if self.tumbler:
            result = p.getClosestPoints(obj, self.tumbler, self.max_dist, -1, -1)
            flag4, _, _, _, _, _, _, _, dist4, _, _, _, _, _, = result[0]
            result = p.getClosestPoints(obj, self.tumbler_cover, self.max_dist, -1, -1)
            flag5, _, _, _, _, _, _, _, dist5, _, _, _, _, _, = result[0]
        else:
            flag4, flag5 = 0, 0
            dist4, dist5 = 3., 3.
        
        if self.lotion:
            # Bin에다 로션을 정리해야 하는게 goal 인 Epi 
            if (self.allocate_lotion_goal() == "2" and self.org_flag < 0.5) or (self.allocate_lotion_goal() == "3" and self.org_flag < 0.33):
                flag6 = 0
                dist6 = 3.
            # lotion 의 goal -> table
            else:
                result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
                flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = 0
            dist6 = 3.
        

        if flag3 or flag4 or flag5 or flag6:
            flag = False # bad
        else:
            flag = True  # good
        
        if dist3 > bin_thres and dist4 > bin_thres and dist5 > bin_thres and dist6 > bin_thres:
            flag_d = True # good
        else:
            flag_d = False # good

        return flag and flag_d



    def ensure_bin_cov(self, obj, status="init"):
        bin_thres = 0.02
        
        if self.drawer:
            result = p.getClosestPoints(obj, self.drawer, self.max_dist, -1, -1)
            flag3, _, _, _, _, _, _, _, dist3, _, _, _, _, _, = result[0]
        else:
            flag3 = 0
            dist3 = 3.
        
        if self.tumbler:
            result = p.getClosestPoints(obj, self.tumbler, self.max_dist, -1, -1)
            flag4, _, _, _, _, _, _, _, dist4, _, _, _, _, _, = result[0]
            result = p.getClosestPoints(obj, self.tumbler_cover, self.max_dist, -1, -1)
            flag5, _, _, _, _, _, _, _, dist5, _, _, _, _, _, = result[0]
        else:
            flag4, flag5 = 0, 0
            dist4, dist5 = 3., 3.
        
        if self.lotion:
            # Bin에다 로션을 정리해야 하는게 goal 인 Epi 
            if (self.allocate_lotion_goal() == "2" and self.org_flag < 0.5) or (self.allocate_lotion_goal() == "3" and self.org_flag < 0.33):
                flag6 = 0
                dist6 = 3.
            # lotion 의 goal -> table
            else:
                result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
                flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = 0
            dist6 = 3.
        

        if flag3 or flag4 or flag5 or flag6:
            flag = False # bad
        else:
            flag = True  # good
        
        if dist3 > bin_thres and dist4 > bin_thres and dist5 > bin_thres and dist6 > bin_thres:
            flag_d = True # good
        else:
            flag_d = False # good

        return flag and flag_d



    def ensure_drawer(self, obj, status="init"):
        drawer_thres = 0.02
        
        if self.tumbler:
            result = p.getClosestPoints(obj, self.tumbler, self.max_dist, -1, -1)
            flag4, _, _, _, _, _, _, _, dist4, _, _, _, _, _, = result[0]
            result = p.getClosestPoints(obj, self.tumbler_cover, self.max_dist, -1, -1)
            flag5, _, _, _, _, _, _, _, dist5, _, _, _, _, _, = result[0]
        else:
            flag4, flag5 = 0, 0
            dist4, dist5 = 3., 3.
        
        if self.lotion:
            # Drawer에다 로션을 정리해야 하는게 goal 인 Epi 
            if (self.allocate_lotion_goal() == "3" and 0.33 <= self.org_flag < 0.66) or (self.allocate_lotion_goal() == "5" and self.org_flag < 0.5):
                flag6 = 0
                dist6 = 3.
            # lotion 의 goal -> table
            else:
                result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
                flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = 0
            dist6 = 3.
        

        if flag4 or flag5 or flag6:
            flag = False # bad
        else:
            flag = True  # good
        
        if dist4 > drawer_thres and dist5 > drawer_thres and dist6 > drawer_thres:
            flag_d = True # good
        else:
            flag_d = False # good

        return flag and flag_d


    def ensure_tumbler(self, obj, status="init"):
        tumbler_thres = 0.02
        
        if self.lotion:
            # Bin에다 로션을 정리해야 하는게 goal 인 Epi 
            result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
            flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = 0
            dist6 = 3.
        

        if flag6:
            flag = False # bad
        else:
            flag = True  # good
        
        if dist6 > tumbler_thres:
            flag_d = True # good
        else:
            flag_d = False # good

        return flag and flag_d
    

    def ensure_tumbler_cover(self, obj, status="init"):
        tumbler_thres = 0.02
        
        if self.lotion:
            # Bin에다 로션을 정리해야 하는게 goal 인 Epi 
            result = p.getClosestPoints(obj, self.lotion, self.max_dist, -1, -1)
            flag6, _, _, _, _, _, _, _, dist6, _, _, _, _, _, = result[0]
        else:
            flag6 = 0
            dist6 = 3.
        

        if flag6:
            flag = False # bad
        else:
            flag = True  # good
        
        if dist6 > tumbler_thres:
            flag_d = True # good
        else:
            flag_d = False # good

        return flag and flag_d



    def rgb2gray_main_goal_ori(self, n_epi):

        _, _, _, _, segImg_goal = p.getCameraImage(width=1280, height=720, viewMatrix=self.viewMatrix_main, projectionMatrix=self.projectionMatrix_main, shadow = 0, renderer=p.ER_BULLET_HARDWARE_OPENGL, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        # _, _, bgr, _, segImg_goal = p.getCameraImage(width=1280, height=720, viewMatrix=self.viewMatrix_main, projectionMatrix=self.projectionMatrix_main, shadow = 0, renderer=p.ER_BULLET_HARDWARE_OPENGL, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        
        seg_rgb_goal = self.seg_get_img(segImg_goal)
        # roi = seg_rgb_goal
        roi = seg_rgb_goal[20:620, 80:1200]
        
        grayImg__ = cv2.resize(roi, dsize=(140, 75), interpolation=cv2.INTER_CUBIC)

        self.rgb_image_main_goal = transform(grayImg__)
        
        
        
    def rgb2gray_main_state_ori(self, step=0, n_epi=0):

        _, _, _, _, segImg_main = p.getCameraImage(width=1280, height=720, viewMatrix=self.viewMatrix_main, projectionMatrix=self.projectionMatrix_main, shadow = 0, renderer=p.ER_BULLET_HARDWARE_OPENGL, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        # _, _, bgr, _, segImg_main = p.getCameraImage(width=1280, height=720, viewMatrix=self.viewMatrix_main, projectionMatrix=self.projectionMatrix_main, shadow = 0, renderer=p.ER_BULLET_HARDWARE_OPENGL, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        
        seg_rgb_main = self.seg_get_img(segImg_main)
        # roi = seg_rgb_main
        roi = seg_rgb_main[20:620, 80:1200]

        grayImg__ = cv2.resize(roi, dsize=(140, 75), interpolation=cv2.INTER_CUBIC)

        self.rgb_image_main = transform(grayImg__)
        
        
        
    def goal_desir_ed_pose(self, n_epi):
        p.resetSimulation()

        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF(my_path + '/object/plane/plane.urdf', [0, 0, -1])

        self.table = p.loadURDF(my_path + '/object/table/table.urdf', basePosition=[0.6, 0.3, -0.625], baseOrientation=p.getQuaternionFromEuler([0., 0., math.pi]))

        self.laptop_gscale = 0.95 + 0.1*random.random()      # >>> 2 / 1
        self.bin_gscale = 0.95 + 0.1*random.random()         # >>> 3 / 2
        self.drawer_gscale = 0.95 + 0.1*random.random()      # >>> 2 / 1
        self.tumbler_gscale = 0.95 + 0.1*random.random()     # >>> 3 / 2
        self.lotion_gscale = 0.95 + 0.1*random.random()      # >>> 4 / 3
        
        self.generate_goal()

        for _ in range(250):
            p.stepSimulation()
        
        self.index()
        # self.reset_goal_(n_epi)
        self.done_z_goal = self.whether_contact_plane()
        self.goal_check2 = self.ensure_not_overlap_goal()

        # True -> good
        # False -> bad
        if self.goal_check2:
            print('Goal, No Occulusion, Success goal reset pos.')
        else:
            print('Goal, Yes Occulusion, Fail goal reset pos.')


        
        
    def initial_random_pose(self, n_epi):
        self.generate_init(n_epi)

        for _ in range(250):
            p.stepSimulation()

        self.done_z_init = self.whether_contact_plane()

        # True -> bad
        # False -> good
        self.init_check = self.ensure_not_overlap_init()

        self.init_check_2 = self.ensure_not_overlap_init_2()


    def seg_get_img(self, seg_img):
        rgb = np.zeros((720, 1280, 3)) # (y, x, c) // (h, w, d)
        
        for index in self.table_idx:
            rgb[:,:,0][seg_img == index] = 0 # b
            rgb[:,:,1][seg_img == index] = 0 # g 
            rgb[:,:,2][seg_img == index] = 0 # r

    # if not self.is_laptop == None:
        # 노트북은 흰색
        for index in self.laptop_idx:
            rgb[:,:,0][seg_img == index] = 255 # b
            rgb[:,:,1][seg_img == index] = 255 # g 
            rgb[:,:,2][seg_img == index] = 255 # r
        
    # if not self.is_drawer == None:
        # 서랍은 노란색
        for index in self.drawer_idx:
            rgb[:,:,0][seg_img == index] = 0   # b
            rgb[:,:,1][seg_img == index] = 255 # g 
            rgb[:,:,2][seg_img == index] = 255 # r
            
    # if not self.is_bin == None:
        # 빈은 파란색
        for index in self.bin_idx:
            rgb[:,:,0][seg_img == index] = 255  # b
            rgb[:,:,1][seg_img == index] = 0    # g 
            rgb[:,:,2][seg_img == index] = 0    # r
            
        # 빈 뚜껑은 파란색
        for index in self.bin_cover_idx:
            rgb[:,:,0][seg_img == index] = 255  # b
            rgb[:,:,1][seg_img == index] = 0    # g 
            rgb[:,:,2][seg_img == index] = 0    # r
    
    # if not self.is_tumbler == None:
        # 텀블러는 빨강색
        for index in self.tumbler_idx:
            rgb[:,:,0][seg_img == index] = 0   # b
            rgb[:,:,1][seg_img == index] = 0   # g 
            rgb[:,:,2][seg_img == index] = 255 # r
            
        # 텀블러 뚜껑은 빨강색
        for index in self.tumbler_cover_idx:
            rgb[:,:,0][seg_img == index] = 0   # b
            rgb[:,:,1][seg_img == index] = 0   # g 
            rgb[:,:,2][seg_img == index] = 255 # r

    # if not self.is_lotion == None:
        # 핸드크림은 초록색
        for index in self.lotion_idx:
            rgb[:,:,0][seg_img == index] = 0   # b
            rgb[:,:,1][seg_img == index] = 255 # g 
            rgb[:,:,2][seg_img == index] = 0   # r

        return rgb.astype(np.uint8)



    def index(self):
        self.table_idx = []
        self.laptop_idx = []
        self.drawer_idx = []
        self.bin_idx = []
        self.bin_cover_idx = []
        self.tumbler_idx = []
        self.tumbler_cover_idx = []
        self.lotion_idx = []

        # for i in range(1):
        #     self.table_idx.append([self.table+(i<<24)])
        self.table_idx.append([self.table+(0<<24)])
        
        
        if not self.is_laptop == None:
            for i in range(3):
                self.laptop_idx.append([self.laptop+(i<<24)])
            # self.laptop_idx.append([self.laptop+(1<<24)])
        else:
            self.laptop_idx.append([None])
        
        
        if not self.is_drawer == None:
            for i in range(13):
                self.drawer_idx.append([self.drawer+(i<<24)])
            # self.drawer_idx.append([self.drawer+(1<<24)])
        else:
            self.drawer_idx.append([None])


        if not self.is_bin == None:
            for i in range(5):
                self.bin_idx.append([self.bin+(i<<24)])
            # self.bin_idx.append([self.bin+(1<<24)])
            
            for i in range(5):
                self.bin_cover_idx.append([self.bin_cover+(i<<24)])
            # self.bin_cover_idx.append([self.bin_cover+(1<<24)])
        else:
            self.bin_idx.append([None])
            self.bin_cover_idx.append([None])
        
        
        if not self.is_tumbler == None:
            # for i in range(1):
            #     self.tumbler_idx.append([self.tumbler+(i<<24)])
            self.tumbler_idx.append([self.tumbler+(0<<24)])
            
            # for i in range(1):
            #     self.tumbler_cover_idx.append([self.tumbler_cover+(i<<24)])
            self.tumbler_cover_idx.append([self.tumbler_cover+(0<<24)])
        else:
            self.tumbler_idx.append([None])
            self.tumbler_cover_idx.append([None])
            
            
        if not self.is_lotion == None:
            # for i in range(1):
            #     self.lotion_idx.append([self.lotion+(i<<24)])
            self.lotion_idx.append([self.lotion+(0<<24)])
        else:
            self.lotion_idx.append([None])



    def get_which_contactID(self, objectID): 

        contactFlag = p.getContactPoints(objectID)
        contact_z_value = []
        contact_obj_ID = []
        for i in range(len(contactFlag)):
            contact_obj_ID.append(contactFlag[i][2])   
            contact_z_value.append(contactFlag[i][5][2])   

        contact_obj_ID = set(contact_obj_ID)
        contact_obj_ID = list(contact_obj_ID)

        return contact_obj_ID, contact_z_value 



    def allocate_lotion_goal(self):
        # 1
        if 2 in self.list_of_objs and not 4 in self.list_of_objs and not 1 in self.list_of_objs:
            return "1"

        # 2
        elif 2 in self.list_of_objs and 4 in self.list_of_objs and not 1 in self.list_of_objs:
            return "2"

        # 3
        elif 2 in self.list_of_objs and 4 in self.list_of_objs and 1 in self.list_of_objs:
            return "3"

        # 4
        elif not 2 in self.list_of_objs and not 4 in self.list_of_objs and 1 in self.list_of_objs:
            return "4"

        # 5
        elif not 2 in self.list_of_objs and 4 in self.list_of_objs and 1 in self.list_of_objs:
            return "5"

        # 6
        elif not 2 in self.list_of_objs and 4 in self.list_of_objs and not 1 in self.list_of_objs:
            return "6"
        
        # 0
        else:
            return "0"
        

    def reset(self, n_epi):
        while True:
            print('+'*60)
            print('Env. Reset start !')
            print('+'*60)   
            
            self.org_flag = random.random()

            #######################################################################################
            self.set_goal_parameter()
            self.set_init_parameter()
            #######################################################################################
            
            self.is_laptop, self.is_drawer, self.is_bin, self.is_tumbler, self.is_lotion = None, None, None, None, None
            
            # 0 -> laptop
            # 1 -> drawer
            # 2 -> bin & bin_cover
            # 3 -> tumbler & tumbler_cover
            # 4 -> lotion
            self.num_objs = random.randint(2,5)
            
            self.total_num_objs = 5
            self.list_of_objs = self.list_of_objs_create(self.total_num_objs)      # [0,1,2,3,4]
            self.list_of_objs = np.random.choice(self.list_of_objs, self.num_objs, replace=False) # [2.3]
            self.list_of_objs.sort()
            
            print("list_of_objs :", self.list_of_objs)

            #######################################################################################
            if 0 in self.list_of_objs:

                if self.laptop_init_idx:
                    self.laptop_disabled = 0.
                    self.is_laptop = 'init'

                else:
                    self.laptop_disabled = 1.
                    self.is_laptop = 'goal'
            else:
                self.laptop_disabled = 1.
            ##############################################################
            if 1 in self.list_of_objs:

                if self.drawer_init_idx:
                    self.drawer_disabled = 0.
                    self.is_drawer = 'init'

                else:
                    self.drawer_disabled = 1.
                    self.is_drawer = 'goal'
            else:
                self.drawer_disabled = 1.
            ##############################################################
            if 2 in self.list_of_objs:

                if self.bin_bin_cover_init_idx:
                    self.bin_bin_cover_disabled = 0.
                    self.is_bin = 'init' 

                else:
                    self.bin_bin_cover_disabled = 1.
                    self.is_bin = 'goal' 
            else:
                self.bin_bin_cover_disabled = 1.
            ##############################################################
            if 3 in self.list_of_objs:
                self.tumbler_tumbler_cover_disabled = 0.  
                self.is_tumbler = 'init'
            else:
                self.tumbler_tumbler_cover_disabled = 1. 
            ##############################################################
            if 4 in self.list_of_objs:
                self.lotion_disabled = 0.                
                self.is_lotion = 'init'
            else:
                self.lotion_disabled = 1. 
            ##############################################################
                

            self.goal_gen_ord = [0,1,2,3,4,5,6]
            
            self.laptop, self.drawer, self.bin, self.bin_cover, self.tumbler, self.tumbler_cover, self.lotion = None, None, None, None, None, None, None
            
            self.goal_desir_ed_pose(n_epi)


            if self.is_lotion == None:
                self.lotion_org_loc = None

            print('='*60)
            print("lotion_org_loc :", self.lotion_org_loc)
            print('='*60)
            
            if (not self.goal_check2) or self.done_z_goal:
                print("self.goal_check2 :", self.goal_check2)
                print("self.done_z_goal :", self.done_z_goal)
                print("Reset during goal")
                continue
            
            
            self.rgb2gray_main_goal_ori(n_epi)


            self.initial_random_pose(n_epi)
            

            if self.done_z_init or self.init_check or self.init_check_2:
                print("self.init_check :", self.init_check)   
                print("self.init_check_2 :", self.init_check_2) 
                                                                # True -> bad
                                                                # False -> good
                print("self.done_z_init :", self.done_z_init)
                print("Reset during init")
                continue
            
            
            self.rgb2gray_main_state_ori(n_epi=n_epi)
            
            
            self.done = False
            
            self.done_laptop_epi, self.done_bin_epi, self.done_drawer_epi, self.done_tumbler_epi, self.done_lotion_epi = False, False, False, False, False
            
            print('@'*30)
            print('laptop : {}'.format(self.laptop))
            print('drawer : {}'.format(self.drawer))
            print('bin : {}'.format(self.bin))
            print('tumbler : {}'.format(self.tumbler))
            print('lotion : {}'.format(self.lotion))
            print('@'*30)
            
            self.rgb_image = torch.cat((self.rgb_image_main, self.rgb_image_main_goal), 0)


            if self.laptop_disabled and self.bin_bin_cover_disabled and self.drawer_disabled and self.tumbler_tumbler_cover_disabled and self.lotion_disabled:
                print('Done all objs while RESET !') 
                self.done_all = 1.
            else:
                print('Not Yet while RESET !') 
                self.done_all = 0.

      
            print('*'*60)
            print("Final Good Reset !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print('*'*60)
                     
            break

        return self.rgb_image



    def diff_norm_quat(self, quat1, quat2):
        
        cond1 = quat1[0] * quat2[0]
        cond2 = quat1[1] * quat2[1]
        cond3 = quat1[2] * quat2[2]
        cond4 = quat1[3] * quat2[3]
        
        if (cond1 <= 0) and (cond2 <= 0) and (cond3 <= 0) and (cond4 <= 0) and (np.linalg.norm(quat1) - np.linalg.norm(quat2) < 0.05):
            diff = -np.array(quat1) - np.array(quat2)
        
        else:
            diff = np.array(quat1) - np.array(quat2)
        
        
        return np.linalg.norm(diff)




    def whether_some_other_obj_pos_ori_changes(self, except_obj):  
        if self.bin:
            dist_pos = [self.pos_bin_cover_before[0] - self.pos_bin_cover_after[0], self.pos_bin_cover_before[1] - self.pos_bin_cover_after[1], self.pos_bin_cover_before[2] - self.pos_bin_cover_after[2]]

            norm_quat_diff = self.diff_norm_quat(self.ori_bin_cover_before, self.ori_bin_cover_after)

            if abs(dist_pos[2]) > 0.01 or np.linalg.norm(dist_pos) > 0.03 or norm_quat_diff > 0.25:
                changed_bin_cover = True
            else:
                changed_bin_cover = False
        else:
            changed_bin_cover = False


        if self.tumbler:
            dist_pos = [self.pos_tumbler_before[0] - self.pos_tumbler_after[0], self.pos_tumbler_before[1] - self.pos_tumbler_after[1], self.pos_tumbler_before[2] - self.pos_tumbler_after[2]]

            norm_quat_diff = self.diff_norm_quat(self.ori_tumbler_before, self.ori_tumbler_after)

            if abs(dist_pos[2]) > 0.01 or np.linalg.norm(dist_pos) > 0.03 or norm_quat_diff > 0.25:
                changed_tumbler = True
            else:
                changed_tumbler = False

    

            dist_pos = [self.pos_tumbler_cover_before[0] - self.pos_tumbler_cover_after[0], self.pos_tumbler_cover_before[1] - self.pos_tumbler_cover_after[1], self.pos_tumbler_cover_before[2] - self.pos_tumbler_cover_after[2]]


            if abs(dist_pos[2]) > 0.01 or np.linalg.norm(dist_pos) > 0.03:
            # if abs(dist_pos[2]) > 0.005 or np.linalg.norm(dist_pos) > 0.03 or norm_quat_diff > 0.25:
                changed_tumbler_cover = True
            else:
                changed_tumbler_cover = False
    
        else:
            changed_tumbler = False
            changed_tumbler_cover = False
            
            
        if self.lotion:
            dist_pos = [self.pos_lotion_before[0] - self.pos_lotion_after[0], self.pos_lotion_before[1] - self.pos_lotion_after[1], self.pos_lotion_before[2] - self.pos_lotion_after[2]]

            if abs(dist_pos[2]) > 0.01 or np.linalg.norm(dist_pos) > 0.03:
                changed_lotion = True
            else:
                changed_lotion = False
        else:
            changed_lotion = False

        if except_obj == 'bin_cover':
            if changed_tumbler or changed_lotion or changed_tumbler_cover:
                Flag = True
            else:
                Flag = False
        
        elif except_obj == 'lotion':
            if changed_bin_cover or changed_tumbler or changed_tumbler_cover:
                Flag = True
            else:
                Flag = False
        
        elif except_obj == 'tumbler':
            if changed_bin_cover or changed_lotion:
                Flag = True
            else:
                Flag = False
                
        elif except_obj == 'tumbler_cover':
            if changed_bin_cover or changed_lotion:
                Flag = True
            else:
                Flag = False
                
        else:
            if changed_bin_cover or changed_tumbler or changed_lotion or changed_tumbler_cover:
                Flag = True
            else:
                Flag = False
        
        return Flag



    def whether_goal_pos_ori_is_reached(self, revelent_obj):
        
        if 'bin_cover' == revelent_obj:
            
            dist_pos = [self.xpos_bin_cover_goal - self.pos_bin_cover_after[0], self.ypos_bin_cover_goal - self.pos_bin_cover_after[1]]

            norm_quat_diff = self.diff_norm_quat(self.ori_bin_cover_goal, self.ori_bin_cover_after)


            if np.linalg.norm(dist_pos) < 0.03 and norm_quat_diff < 0.20:
                reached_goal = True
            else:
                reached_goal = False

        

        elif 'tumbler' == revelent_obj:
            
            dist_pos = [self.xpos_tumbler_goal - self.pos_tumbler_after[0], self.ypos_tumbler_goal - self.pos_tumbler_after[1], self.zpos_tumbler_goal - self.pos_tumbler_after[2]]

            norm_quat_diff = self.diff_norm_quat(self.ori_tumbler_goal, self.ori_tumbler_after)

            if abs(dist_pos[2]) < 0.01 and np.linalg.norm(dist_pos) < 0.03 and norm_quat_diff < 0.20:
                reached_goal = True
            else:
                reached_goal = False
                

        elif 'tumbler_cover' == revelent_obj:
            
            dist_pos = [self.xpos_tumbler_cover_goal - self.pos_tumbler_cover_after[0], self.ypos_tumbler_cover_goal - self.pos_tumbler_cover_after[1], self.zpos_tumbler_cover_goal - self.pos_tumbler_cover_after[2]]

            norm_quat_diff = self.diff_norm_quat(self.ori_tumbler_cover_goal, self.ori_tumbler_cover_after)

            if abs(dist_pos[2]) < 0.01 and np.linalg.norm(dist_pos) < 0.03 and norm_quat_diff < 0.20:
                reached_goal = True
            else:
                reached_goal = False

                

        elif 'lotion' == revelent_obj:
            
            dist_pos = [self.xpos_lotion_goal - self.pos_lotion_after[0], self.ypos_lotion_goal - self.pos_lotion_after[1], self.zpos_lotion_goal - self.pos_lotion_after[2]]


            if abs(dist_pos[2]) < 0.01 and np.linalg.norm(dist_pos) < 0.03:
            # if np.linalg.norm(dist_pos) < 0.03:
                reached_goal = True
            else:
                reached_goal = False

        return reached_goal




    def whether_reached_goal_joint_ang(self, revelent_obj, joint_idx):
        joint_pos, _, _, _ = p.getJointState(revelent_obj, joint_idx)

        if revelent_obj == self.laptop:
            return abs(joint_pos - self.joint_ang_laptop_goal) < 0.1745  
        
        elif revelent_obj == self.drawer:
            return abs(joint_pos - self.joint_ang_drawer0_goal) < 0.005 
        
        
    def get_after_pos_ori_all_obj(self):

        if self.laptop:
            self.pos_laptop_after, self.ori_laptop_after = p.getBasePositionAndOrientation(self.laptop)  
        
        if self.bin:
            self.pos_bin_after, self.ori_bin_after = p.getBasePositionAndOrientation(self.bin)  
            self.pos_bin_cover_after, self.ori_bin_cover_after = p.getBasePositionAndOrientation(self.bin_cover)   
        
        if self.drawer:
            self.pos_drawer_after, self.ori_drawer_after = p.getBasePositionAndOrientation(self.drawer)  
        
        if self.tumbler:
            self.pos_tumbler_after, self.ori_tumbler_after = p.getBasePositionAndOrientation(self.tumbler)   
            self.pos_tumbler_cover_after, self.ori_tumbler_cover_after = p.getBasePositionAndOrientation(self.tumbler_cover) 
        
        if self.lotion:
            self.pos_lotion_after, self.ori_lotion_after = p.getBasePositionAndOrientation(self.lotion) 


        
    def Fold_Laptop(self):
        self.generate_goal_laptop_step() 
        self.get_after_pos_ori_all_obj()

        check_laptop2 = self.whether_reached_goal_joint_ang(self.laptop, 0) 
                                                                            
        check_laptop3 = self.ensure_laptop_step(self.laptop)
        

        
        if (self.is_laptop == 'init') and (check_laptop2) and (check_laptop3):  
  
            self.done_laptop_epi = False
            self.is_laptop = 'goal'
            self.laptop_disabled = 1.
    
        else:
            self.reward = -1.
            self.done_laptop_epi = True




    def Close_Drawer(self):
        self.generate_goal_drawer_step()
        self.get_after_pos_ori_all_obj()


        if (self.is_drawer == 'init') and (self.is_bin == 'goal') and (not self.lotion_org_loc == "table") and ((self.is_lotion == 'drawer') or (self.is_lotion == 'bin')):
    
            self.done_drawer_epi = False
            self.is_drawer = 'goal'


            self.lotion_disabled = 1.
            self.drawer_disabled = 1.


        elif (self.is_drawer == 'init') and (self.is_bin == 'goal') and (self.lotion_org_loc == "table"):
        
            self.done_drawer_epi = False
            self.is_drawer = 'goal'
            self.drawer_disabled = 1.

    

        elif (self.is_drawer == 'init') and (self.is_bin == 'init'):
      
            self.done_drawer_epi = False
            self.is_drawer = 'goal'
            

            if self.is_lotion == 'drawer':
                self.lotion_disabled = 1.
            
            self.drawer_disabled = 1.

 
        elif (self.is_drawer == 'init') and (self.is_bin == None) and ((self.is_lotion == 'drawer') or (self.is_lotion == 'bin')):
     
            self.done_drawer_epi = False
            self.is_drawer = 'goal'
            self.drawer_disabled = 1.
            self.lotion_disabled = 1.
        
       
        elif (self.is_drawer == 'init') and (self.is_bin == None) and (self.lotion_org_loc == "table") and ((self.is_lotion == 'init') or (self.is_lotion == 'table')):

            self.done_drawer_epi = False
            self.is_drawer = 'goal'
            self.drawer_disabled = 1.

      
        elif (self.is_drawer == 'init') and (self.is_lotion == None):
      
            self.done_drawer_epi = False
            self.is_drawer = 'goal'
            self.drawer_disabled = 1.
        
        else:
            self.reward = -1.
            self.done_drawer_epi = True

    

    def Open_Drawer(self):
        
        ##################################
        self.generate_goal_drawer_step_open()
        ##################################
        

        self.get_after_pos_ori_all_obj()


        if (self.is_drawer == 'goal') and (self.is_bin == 'goal') and (not self.lotion_org_loc == "table") and ((self.is_lotion == 'init') or (self.is_lotion == 'table')):
   
            self.done_drawer_epi = False
            self.is_drawer = 'init'
            self.drawer_disabled = 0.
            self.bin_bin_cover_disabled = 1.
        
        elif (self.is_drawer == 'goal') and (self.is_bin == None) and (self.lotion_org_loc == "drawer") and ((self.is_lotion == 'init') or (self.is_lotion == 'table')):
       
            self.done_drawer_epi = False
            self.is_drawer = 'init'
            self.drawer_disabled = 0.
        
        else:
            self.reward = -1.
            self.done_drawer_epi = True
            self.drawer_disabled = 0.
    
    
    def Close_Bin(self):
        self.generate_goal_bin_bin_cover_step() 
        self.get_after_pos_ori_all_obj()

        check_bin = self.whether_some_other_obj_pos_ori_changes('bin_cover') 
        
        check_bin2 = self.whether_goal_pos_ori_is_reached('bin_cover')
        

        if (not check_bin) and (check_bin2) and (self.is_bin == 'init') and (self.is_drawer == 'goal') and (not self.lotion_org_loc == "table") and ((self.is_lotion == 'drawer') or (self.is_lotion == 'bin')):
      
            self.done_bin_epi = False
            self.is_bin = 'goal'        
            

            self.lotion_disabled = 1.
            self.bin_bin_cover_disabled = 1.

    
        elif (not check_bin) and (check_bin2) and (self.is_bin == 'init') and (self.is_drawer == 'goal') and (self.lotion_org_loc == "table"):
      
            self.done_bin_epi = False
            self.is_bin = 'goal'
            self.bin_bin_cover_disabled = 1.

    
        elif (not check_bin) and (check_bin2) and (self.is_bin == 'init') and (self.is_drawer == 'init'):
       
            self.done_bin_epi = False
            self.is_bin = 'goal'
            
  
            if self.is_lotion == 'bin':
                self.lotion_disabled = 1.

            self.bin_bin_cover_disabled = 1.

        elif (not check_bin) and (check_bin2) and (self.is_bin == 'init') and (self.is_drawer == None) and ((self.is_lotion == 'drawer') or (self.is_lotion == 'bin')):
 
            self.done_bin_epi = False
            self.is_bin = 'goal'
            self.lotion_disabled = 1
            self.bin_bin_cover_disabled = 1.

        elif (not check_bin) and (check_bin2) and (self.is_bin == 'init') and (self.is_drawer == None) and (self.lotion_org_loc == "table") and ((self.is_lotion == 'init') or (self.is_lotion == 'table')):
 
            self.done_bin_epi = False
            self.is_bin = 'goal'
            self.bin_bin_cover_disabled = 1.
        

        elif (not check_bin) and (check_bin2) and (self.is_bin == 'init') and (self.is_lotion == None):
            # self.reward = 1. 
            self.done_bin_epi = False
            self.is_bin = 'goal'
            self.bin_bin_cover_disabled = 1.
    
        else:
            self.reward = -1.
            self.done_bin_epi = True

    
    def Open_Bin(self):
        
        ############################################
        self.generate_goal_bin_bin_cover_step_open() 
        ############################################
        
        self.get_after_pos_ori_all_obj()

        if (self.is_bin == 'goal') and (self.is_drawer == 'goal') and (not self.lotion_org_loc == "table") and ((self.is_lotion == 'init') or (self.is_lotion == 'table')):

            self.done_bin_epi = False
            self.is_bin = 'init'        
            self.bin_bin_cover_disabled = 0.
            self.drawer_disabled = 1.
        
        elif (self.is_bin == 'goal') and (self.is_drawer == None) and (self.lotion_org_loc == "bin") and ((self.is_lotion == 'init') or (self.is_lotion == 'table')):

            self.done_bin_epi = False
            self.is_bin = 'init'
            self.bin_bin_cover_disabled = 0.
        
        else:
            self.reward = -1.
            self.done_bin_epi = True
            self.bin_bin_cover_disabled = 0.
    
    
    def Organize_Tumbler(self):
        self.generate_goal_tumbler_step() 
        self.get_after_pos_ori_all_obj()
        
        check_tumbler = self.whether_some_other_obj_pos_ori_changes('tumbler') 
        
        check_tumbler2 = self.whether_goal_pos_ori_is_reached('tumbler')       
        
        check_tumbler3 = self.whether_goal_pos_ori_is_reached('tumbler_cover')
        
        if (self.is_tumbler == 'init') and (not check_tumbler) and (check_tumbler2) and (check_tumbler3):
            # self.reward = 1. 
            self.done_tumbler_epi = False
            self.is_tumbler = 'goal'
            self.tumbler_tumbler_cover_disabled = 1.
    
        else:
            self.reward = -1.
            self.done_tumbler_epi = True



    def Organize_Lotion_on_the_table(self):
        
        ########################################
        self.before_is_lotion = self.is_lotion
        ########################################
        
        self.generate_goal_lotion_step(lotion_org_loc='table')                
        self.get_after_pos_ori_all_obj()

        check_lotion = self.whether_some_other_obj_pos_ori_changes('lotion')  
        check_lotion2 = self.whether_goal_pos_ori_is_reached('lotion')   

        if (self.is_lotion == 'init') and (not check_lotion) and (check_lotion2) and (self.lotion_org_loc == "table"):

            self.done_lotion_epi = False
            self.is_lotion = 'table'
            self.lotion_disabled = 1.
    
        else:
            self.reward = -1.
            self.done_lotion_epi = True
            
            ############################################################################################################################################
            if self.before_is_lotion == 'drawer' and self.is_drawer == 'goal': 
                self.lotion_disabled = 0.
                self.drawer_disabled = 0.
                self.bin_bin_cover_disabled = 0.

            if self.before_is_lotion == 'bin' and self.is_bin == 'goal': 
                self.lotion_disabled = 0.
                self.drawer_disabled = 0.
                self.bin_bin_cover_disabled = 0.
            ############################################################################################################################################



    def Organize_Lotion_in_Drawer(self):
        self.generate_goal_lotion_step(lotion_org_loc='drawer')               
        self.get_after_pos_ori_all_obj()

        check_lotion = self.whether_some_other_obj_pos_ori_changes('lotion')  
        
        if (self.is_lotion == 'init') and (self.is_drawer == 'init') and (not check_lotion) and (not self.lotion_org_loc == "table"):

            self.done_lotion_epi = False
            self.is_lotion = 'drawer'
    
        else:
            self.reward = -1.

            self.done_lotion_epi = True




    def Organize_Lotion_in_Bin(self):
        self.generate_goal_lotion_step(lotion_org_loc='bin')               
        self.get_after_pos_ori_all_obj()

        check_lotion = self.whether_some_other_obj_pos_ori_changes('lotion')  
        if (self.is_lotion == 'init') and (self.is_bin == 'init') and (not check_lotion) and (not self.lotion_org_loc == "table"):

            self.done_lotion_epi = False
            self.is_lotion = 'bin'
    
        else:
            self.reward = -1.

            self.done_lotion_epi = True




    def step(self, action, present_step, global_step, n_epi):
        print('='*60)
        
        self.done = False
        self.reward = 0.

        if self.laptop:
            self.pos_laptop_before, self.ori_laptop_before = p.getBasePositionAndOrientation(self.laptop)  

        if self.bin:
            self.pos_bin_before, self.ori_bin_before = p.getBasePositionAndOrientation(self.bin)  
            self.pos_bin_cover_before, self.ori_bin_cover_before = p.getBasePositionAndOrientation(self.bin_cover)   

        if self.drawer:
            self.pos_drawer_before, self.ori_drawer_before = p.getBasePositionAndOrientation(self.drawer)  
        
        if self.tumbler:
            self.pos_tumbler_before, self.ori_tumbler_before = p.getBasePositionAndOrientation(self.tumbler)   
            self.pos_tumbler_cover_before, self.ori_tumbler_cover_before = p.getBasePositionAndOrientation(self.tumbler_cover)

        if self.lotion:
            self.pos_lotion_before, self.ori_lotion_before = p.getBasePositionAndOrientation(self.lotion)   


        if action == 0:
            print('Action 0 : Fold Laptop !') 
            if self.laptop:
                self.Fold_Laptop()
            else:
                print('No Laptop !') 
                self.reward = -1.
                self.done_laptop_epi = True
        
        
        elif action == 1:
            print('Action 1 : Close Drawer !')
            if self.drawer:
                self.Close_Drawer()
            else:
                print('No Drawer !') 
                self.reward = -1.
                self.done_drawer_epi = True


        elif action == 2:
            print('Action 2 : Open Drawer !')
            if self.drawer:
                self.Open_Drawer()
            else:
                print('No Drawer !') 
                self.reward = -1.
                self.done_drawer_epi = True


        elif action == 3:
            print('Action 3 : Close Bin !')
            if self.bin:
                self.Close_Bin()
            else:
                print('No Bin !') 
                self.reward = -1.
                self.done_bin_epi = True


        elif action == 4:
            print('Action 4 : Open Bin !')
            if self.bin:
                self.Open_Bin()
            else:
                print('No Bin !') 
                self.reward = -1.
                self.done_bin_epi = True


        elif action == 5:
            print('Action 5 : Organize Tumbler on the table !')
            if self.tumbler:
                self.Organize_Tumbler()
            else:
                print('No Tumbler !') 
                self.reward = -1.
                self.done_tumbler_epi = True


        elif action == 6:
            print('Action 6 : Organize Lotion on the table !')
            if self.lotion:
                self.Organize_Lotion_on_the_table()
            else:
                print('No Lotion !') 
                self.reward = -1.
                self.done_lotion_epi = True


        elif action == 7:
            print('Action 7 : Organize Lotion in Drawer !')
            if self.lotion:
                self.Organize_Lotion_in_Drawer()
            else:
                print('No Lotion !') 
                self.reward = -1.
                self.done_lotion_epi = True


        elif action == 8:
            print('Action 8 : Organize Lotion in Bin!')
            if self.lotion:
                self.Organize_Lotion_in_Bin()
            else:
                print('No Lotion !') 
                self.reward = -1.
                self.done_lotion_epi = True


        if self.laptop_disabled and self.bin_bin_cover_disabled and self.drawer_disabled and self.tumbler_tumbler_cover_disabled and self.lotion_disabled:
            print('Done all objs while STEP !') 
            self.done_all = 1.
        else:
            print('Not Yet while STEP !') 
            self.done_all = 0.

                
        self.rgb2gray_main_state_ori(present_step, n_epi)
        

        self.rgb_image = torch.cat((self.rgb_image_main, self.rgb_image_main_goal), 0)


        print('+'*30)
        print('n_epi :', n_epi)
        print('present_step :', present_step)
        print('global_step :', global_step)
        print('+'*30)
        

        if not self.done_laptop_epi and not self.done_bin_epi and not self.done_drawer_epi and not self.done_tumbler_epi and not self.done_lotion_epi:
            print('NO Criticla Error occur_ed yet ! Keep going !')
            self.done_epi = False
            self.done = False
        else:
            print('Criticla Error occur_ed ! Reset Env. !')
            self.done_epi = True
            self.done = True
            self.reward = -1.



        if self.whether_contact_plane():
            self.done = True
            self.reward = -1.
            print('Bad step_pos-Z !, Epi. is terminated During STEP')
        

        print('laptop_disabled:', self.laptop_disabled)
        print('bin_bin_cover_disabled :', self.bin_bin_cover_disabled)
        print('drawer_disabled :', self.drawer_disabled)
        print('tumbler_tumbler_cover_disabled :', self.tumbler_tumbler_cover_disabled)
        print('lotion_disabled:', self.lotion_disabled)


        # REACH GOAL
        if not self.done_epi and self.done_all:
            self.done = True
            self.reward = 1.  # SPARSE REWARD
        
        
        print('self.reward :', self.reward)
        print('='*60)


        return self.rgb_image, self.reward, self.done