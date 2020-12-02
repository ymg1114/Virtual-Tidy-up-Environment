# -*- coding: utf-8 -*-
import sys
sys.path.append("..") # Adds higher directory to python modules path.
# import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from time import sleep
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
# from segment_tree import MinSegmentTree, SumSegmentTree
import cv2
import os
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import skimage.io
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from modeling.deeplab import *
from dataloaders.utils import  *
from dataloaders import custom_transforms as tr
from PIL import Image as Im
import rospy
from pynput.keyboard import Key, Listener
from math import pi, atan, tan, sin, cos, sqrt, atan2, atan
from math import degrees as deg
from math import radians as rad
from urdf_parser_py.urdf import URDF
# from pykdl_utils.kdl_kinematics import KDLKinematics
from robotiq_85_msgs.msg import GripperCmd 
from robot_control_msgs.srv import *
from mrcnn import model as modellib
# Import Mask RCNN

from mrcnn import utils
# import mrcnn.model as modellib
from mrcnn import visualize, visualize_mg
from mrcnn.config import Config
import tensorflow as tf

from math import pi, atan, tan, sin, cos, sqrt, atan2, acos
from math import degrees as deg
from math import radians as rad

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

my_path = os.path.abspath(os.path.dirname(__file__))


# # 0. ~ 6. -> 0. ~ 1.
# # "input" -> should be "torch.tensor"
# def scaler(input):
#     input = torch.tensor(input, dtype=torch.float)
#     input = input / 6.
#     return input


# # 0. ~ 6. -> 0. ~ 1. -> -1. ~ 1.
# # "input" -> should be "torch.tensor"
# def normalize(input):
#     input = torch.tensor(input, dtype=torch.float)
#     input = input / 6.
#     mean, std = torch.FloatTensor([0.5]), torch.FloatTensor([0.5])
#     input = (input - mean) / (std+1e-5)
#     return input 


# # 0~255 -> 0. ~ 1. -> -1. ~ 1.
# # "input" -> should be "torch.tensor"
# def normalize_(input):
#     input = torch.tensor(input, dtype=torch.float)
#     input = input / 6.
#     mean, std = torch.FloatTensor([0.5]), torch.FloatTensor([0.5])
#     input = (input - mean) / (std+1e-5)
#     return input 


# # decoder activation => tanh
# # -1~1 -> 0~255
# def denormalize(input):
#     mean, std = torch.cuda.FloatTensor([0.5]), torch.cuda.FloatTensor([0.5])
#     denorm_image = (((input * std) + mean) * 255).type(torch.uint8)
#     return denorm_image


# # decoder activation => sigmoid
# # 0~1 -> 0~255
# def denormalize(input):
#     denorm_image = (input * 255).type(torch.uint8)
#     return denorm_image




###################################
#       topic   실제와의 오차 
# z_ee = 0.485   #- 0.025    # >>> 0.456  
# z_ee = 0.65 # 0.58 >>> 0.65  
# z_ee = 0.53 
###################################



EVAL = True


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5,))])  

composed_transforms = transforms.Compose([tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), tr.ToTensor()])


# learning_rate = 0.001 # 0.0005 -> 0.001(for LR-scheduler)
learning_rate = 0.0005 
gamma         = 0.7
buffer_limit  = 100000  # 100000 -> 200000 -> 150000 -> 100000 -> 200000 -> 150000 -> 200000 -> 100000
size = [buffer_limit]
batch_size    = 128 # 32 -> 128 
interval = 200 # 20 -> 200
max_step = 30 # 600 -> 50 -> 30 -> 40 -> 50 -> 30
start_training_memory_size = 10000 # 2000 -> 10000 -> 20000 / 10000

###################################
###################################
reshape_ = (1,6,75,140) 
obs_dim = [6,75,140]   
###################################
# reshape_ = (1,6,60,112) 
# obs_dim = [6,60,112]   
###################################
###################################

act_dim = 9
clip_value = 1.
max_norm = 40.

alpha = 0.6
beta_start = 0.4
beta_frames = 200000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

num_atoms = 51
Vmin = -2
Vmax = 2 

n_step = 1

AEN_dim = [act_dim]

AEN_epoch = 1

DDN_epoch = 1

# sub_buffer_limit = 50000  # 50000 
# sub_batch_size = 128 # 128
sub_start_training_memory_size = 5000 # 10000 <-> 5000



device = torch.device('cuda')
print('device :', device)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
     
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features, out_features, std_init = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(x,self.weight_mu + self.weight_sigma * self.weight_epsilon,self.bias_mu + self.bias_sigma * self.bias_epsilon)
    
    @staticmethod
    def scale_noise(size):
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())



################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
class RAINBOW_Q_net(nn.Module):
    def __init__(self, in_dim, out_dim, num_atoms, Vmin, Vmax):
        super(RAINBOW_Q_net, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(device)  # 이거는 랜덤으로 값이 결정되는 그런게 아님
                                                                                        # 인터폴레이션 개넘으로 "Vmin ~ Vmax" 구간을 동일한 길이로 "num_atoms" 개수 만큼 나눔
        
        # # For LunarLanderv2
        # # For LunarLanderv2
        # # For LunarLanderv2
        # ################################################################################
        # # set common feature layer
        # self.feature_layer = nn.Sequential(nn.Linear(self.in_dim[0], 128), 
        #                                    nn.ReLU(),
        #                                    nn.Linear(128, 128),
        #                                    nn.ReLU())
        
        # # set noisy value layer
        # self.noisy_value_1 = NoisyLinear(128, 128)
        # self.noisy_value_2 = NoisyLinear(128, self.num_atoms)
        
        
        # # set noisy advantage layer
        # self.noisy_advantage_1 = NoisyLinear(128, 128)  
        # self.noisy_advantage_2 = NoisyLinear(128, self.out_dim*self.num_atoms)
        # ################################################################################
        
        
        
        ################################################################################
        # set common feature layer
        self.feature_layer = nn.Sequential(nn.Conv2d(self.in_dim[0], 32, 8, 4),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(32, 64, 4, 2),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(64, 64, 3, 1),
                                               nn.LeakyReLU(),
                                               Flatten())


        # set noisy value layer
        self.noisy_value_1 = NoisyLinear(self._feature_size(), 512)
        self.noisy_value_2 = NoisyLinear(512, self.num_atoms)
        
        
        # set noisy advantage layer
        self.noisy_advantage_1 = NoisyLinear(self._feature_size(), 512)  
        self.noisy_advantage_2 = NoisyLinear(512, self.out_dim*self.num_atoms) 
        ################################################################################
        
        
        ################################################################################
        # Linear + MSE
        self.AEN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
                                               nn.LeakyReLU(),
                                               nn.Linear(512, self.out_dim))

        # self.AEN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
        #                                        nn.LeakyReLU(),
        #                                        nn.Linear(512, 512),
        #                                        nn.LeakyReLU(),
        #                                        nn.Linear(512, self.out_dim))

        
        self.AEN_loss_fn = nn.MSELoss()
        self.AEN_optimizer = torch.optim.Adam(self.AEN_feature_layer.parameters(), lr=learning_rate)
        ################################################################################


        # ################################################################################
        # # Sigmoid + BCE
        # self.AEN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
        #                                        nn.LeakyReLU(),
        #                                        nn.Linear(512, self.out_dim),
        #                                        nn.Sigmoid())
        
        # self.AEN_loss_fn = nn.BCELoss()
        # self.AEN_optimizer = torch.optim.Adam(self.AEN_feature_layer.parameters(), lr=learning_rate)
        # ################################################################################
        
        
        ################################################################################
        # Sigmoid + BCE
        self.DDN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
                                               nn.LeakyReLU(),
                                               nn.Linear(512, 1),
                                               nn.Sigmoid())

        self.DDN_loss_fn = nn.BCELoss()
        self.DDN_optimizer = torch.optim.Adam(self.DDN_feature_layer.parameters(), lr=learning_rate)
        ################################################################################


    def _feature_size(self):
        return self.feature_layer(torch.zeros(reshape_)).view(1, -1).size(1)


    def forward(self, x):
        """Forward method implementation."""
        dist = self.dist(x)
        
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    
    def dist(self, x):
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        
        val_hid = F.relu(self.noisy_value_1(feature))
        value = self.noisy_value_2(val_hid).view(-1, 1, self.num_atoms)
        
        adv_hid = F.relu(self.noisy_advantage_1(feature))
        advantage = self.noisy_advantage_2(adv_hid).view(-1, self.out_dim, self.num_atoms)
        
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)  # softmax 이기 때문에 -> 싹다 합치면 "1" 임 // q-value의 분포
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist


    def forward_AEN(self, x):
        feature = self.feature_layer(x)
        
        xx = feature.detach()
        
        ########################################
        out = self.AEN_feature_layer(xx)
        ########################################
        
        return out

    def forward_DDN(self, x):
        feature = self.feature_layer(x)
        
        xx = feature.detach()
        
        ########################################
        out = self.DDN_feature_layer(xx)
        ########################################
        
        return out
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################



class Qnet(nn.Module):
    def __init__(self, in_dim=obs_dim, out_dim=act_dim, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax, batch_size=batch_size, n_step=n_step, EVAL=EVAL):
        super(Qnet, self).__init__()
        
        self.prior_eps    = 1e-6
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(device)  # 이거는 랜덤으로 값이 결정되는 그런게 아님
                                                                                        # 인터폴레이션 개넘으로 "Vmin ~ Vmax" 구간을 동일한 길이로 "num_atoms" 개수 만큼 나눔
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size 
    
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        self.n_step = n_step 





        ################################################################################################################################################################
        ################################################################################################################################################################
        ################################################################################################################################################################
        self.Q1 = RAINBOW_Q_net(in_dim=in_dim, out_dim=out_dim, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax).to(device)
        self.Q2 = RAINBOW_Q_net(in_dim=in_dim, out_dim=out_dim, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax).to(device)
       
        # self.aen = AEN_net().to(device)
       
        # mode: train / test
        self.evaluation = EVAL
        if not self.evaluation:
            self.summary = SummaryWriter(SUMMARY_PATH)
        else:  
            # ###############################################################################################################################################################
            # self.Q1.DDN_feature_layer.load_state_dict(torch.load('../DDN/WEIGHT/weight_DDN_test_0_ver2/sequence_RAINBOW_DQN_DDN_45000.pt'))
            # self.Q1.AEN_feature_layer.load_state_dict(torch.load('../AEN/WEIGHT/weight_AEN_test_0_ver2/sequence_RAINBOW_DQN_AEN_45000.pt'))
            # ###############################################################################################################################################################


            # ################################################################################################################################################################
            # self.Q1.DDN_feature_layer.load_state_dict(torch.load('../DDN/WEIGHT/weight_DDN_test_0_ver2_2/sequence_RAINBOW_DQN_DDN_50000.pt'))
            # self.Q1.AEN_feature_layer.load_state_dict(torch.load('../AEN/WEIGHT/weight_AEN_test_0_ver2_2/sequence_RAINBOW_DQN_AEN_50000.pt'))
            # ################################################################################################################################################################


            # ################################################################################################################################################################
            # self.Q1.DDN_feature_layer.load_state_dict(torch.load('../DDN/WEIGHT/weight_DDN_test_0_ver2_4/sequence_RAINBOW_DQN_DDN_50000.pt'))
            # self.Q1.AEN_feature_layer.load_state_dict(torch.load('../AEN/WEIGHT/weight_AEN_test_0_ver2_4/sequence_RAINBOW_DQN_AEN_50000.pt'))
            # ################################################################################################################################################################


            # ################################################################################################################################################################
            # self.Q1.DDN_feature_layer.load_state_dict(torch.load('../DDN/WEIGHT/weight_DDN_test_0_ver2_regrap/sequence_RAINBOW_DQN_DDN_50000.pt'))
            # self.Q1.AEN_feature_layer.load_state_dict(torch.load('../AEN/WEIGHT/weight_AEN_test_0_ver2_regrap/sequence_RAINBOW_DQN_AEN_50000.pt'))
            # ################################################################################################################################################################

            ################################################################################################################################################################
            self.Q1.DDN_feature_layer.load_state_dict(torch.load('/media/irobot/8FD4-3E76/201118_white_drawer_red_tumbler_cap/sequence_RAINBOW_DQN_DDN_17600.pt'))
            self.Q1.AEN_feature_layer.load_state_dict(torch.load('/media/irobot/8FD4-3E76/201118_white_drawer_red_tumbler_cap/sequence_RAINBOW_DQN_AEN_17600.pt'))
            ################################################################################################################################################################



    def organize_done_or_not(self, next_state):
        
        done_prob = self.Q1.forward_DDN(next_state.to(device))

        return done_prob



    def sample_action(self, state):
        """Select an action from the input state."""

        q1 = self.Q1(state.to(device))
        q2 = self.Q2(state.to(device))

        out = torch.min(q1, q2)
        print("out_before :", out)
        print("out_before_action :", out.argmax().item())

        # FOR TEST
        ############################################################################################################
        aen_correct = self.Q1.forward_AEN(state.to(device))
        print("aen_correct :", aen_correct)
        out = out - aen_correct
        print("out_final :", out)
        print("out_final_action :", out.argmax().item())
        ############################################################################################################
        

        return out.argmax().item(), out
        ################################################################################################################################################################
        ################################################################################################################################################################
        ################################################################################################################################################################
        







        
    #     ################################################################################################################################################################
    #     ################################################################################################################################################################
    #     ################################################################################################################################################################
    #     ################################################################################
    #     # set common feature layer
    #     self.feature_layer = nn.Sequential(nn.Conv2d(self.in_dim[0], 32, 8, 4),
    #                                         nn.LeakyReLU(),
    #                                         nn.Conv2d(32, 64, 4, 2),
    #                                         nn.LeakyReLU(),
    #                                         nn.Conv2d(64, 64, 3, 1),
    #                                         nn.LeakyReLU(),
    #                                         Flatten())


    #     # set noisy value layer
    #     self.noisy_value_1 = NoisyLinear(self._feature_size(), 512)
    #     self.noisy_value_2 = NoisyLinear(512, self.num_atoms)
        
        
    #     # set noisy advantage layer
    #     self.noisy_advantage_1 = NoisyLinear(self._feature_size(), 512)  
    #     self.noisy_advantage_2 = NoisyLinear(512, self.out_dim*self.num_atoms) 
    #     ################################################################################
        

    #     ################################################################################
    #     # Linear + MSE
    #     self.AEN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
    #                                         nn.LeakyReLU(),
    #                                         nn.Linear(512, self.out_dim))
        
    #     self.AEN_loss_fn = nn.MSELoss()
    #     self.AEN_optimizer = torch.optim.Adam(self.AEN_feature_layer.parameters(), lr=learning_rate)
    #     ################################################################################
        
        
    #     ################################################################################
    #     # Sigmoid + BCE
    #     self.DDN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
    #                                         nn.LeakyReLU(),
    #                                         nn.Linear(512, 1),
    #                                         nn.Sigmoid())

    #     self.DDN_loss_fn = nn.BCELoss()
    #     self.DDN_optimizer = torch.optim.Adam(self.DDN_feature_layer.parameters(), lr=learning_rate)
    #     ################################################################################


    #     # mode: train / test
    #     self.evaluation = EVAL
    #     if not self.evaluation:
    #         self.summary = SummaryWriter(SUMMARY_PATH)
    #     else:
    #         # print(222222222222)
    #         self.DDN_feature_layer.load_state_dict(torch.load('../DDN/WEIGHT/weight_DDN_test_0_notwin_ver2/sequence_RAINBOW_DQN_DDN_40000.pt'))
    #         # print(222222222222)
    #         self.AEN_feature_layer.load_state_dict(torch.load('../AEN/WEIGHT/weight_AEN_test_0_notwin_ver2/sequence_RAINBOW_DQN_AEN_40000.pt'))



    # def forward_AEN(self, x):
    #     feature = self.feature_layer(x)
        
    #     xx = feature.detach()
        
    #     ########################################
    #     out = self.AEN_feature_layer(xx)
    #     ########################################
        
    #     return out



    # def forward_DDN(self, x):
    #     feature = self.feature_layer(x)
        
    #     xx = feature.detach()
        
    #     ########################################
    #     out = self.DDN_feature_layer(xx)
    #     ########################################
        
    #     return out



    # def _feature_size(self):
    #     return self.feature_layer(torch.zeros(reshape_)).view(1, -1).size(1)



    # def forward(self, x):
    #     """Forward method implementation."""
    #     dist = self.dist(x)
        
    #     q = torch.sum(dist * self.support, dim=2)
        
    #     return q
    
    
    # def dist(self, x):
    #     """Get distribution for atoms."""
    #     feature = self.feature_layer(x)
        
    #     val_hid = F.relu(self.noisy_value_1(feature))
    #     value = self.noisy_value_2(val_hid).view(-1, 1, self.num_atoms)
        
    #     adv_hid = F.relu(self.noisy_advantage_1(feature))
    #     advantage = self.noisy_advantage_2(adv_hid).view(-1, self.out_dim, self.num_atoms)
        
    #     q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
    #     dist = F.softmax(q_atoms, dim=-1)  # softmax 이기 때문에 -> 싹다 합치면 "1" 임 // q-value의 분포
    #     dist = dist.clamp(min=1e-3)  # for avoiding nans

    #     return dist


    # def reset_noise(self):
    #     """Reset all noisy layers."""
    #     self.noisy_value_1.reset_noise()
    #     self.noisy_value_2.reset_noise()
    #     self.noisy_advantage_1.reset_noise()
    #     self.noisy_advantage_2.reset_noise()
    
    
    
    # def organize_done_or_not(self, next_state):
        
    #     done_prob = self.forward_DDN(next_state.to(device))

    #     return done_prob




    # def sample_action(self, state):
    #     """Select an action from the input state."""

    #     out = self.forward(state.to(device))
        
    #     print("="*60)
    #     print("out_before :", out)
    #     print("out_before_action :", out.argmax().item())
    #     print("*"*40)

    #     # # FOR TEST
    #     # ############################################################################################################
    #     # aen_correct = self.forward_AEN(state.to(device))
    #     # print("*"*40)
    #     # print("aen_correct :", aen_correct)
    #     # print("*"*40)
    #     # out = out + aen_correct
    #     # print("out_final :", out)
    #     # print("out_final_action :", out.argmax().item())
    #     # print("*"*40)
    #     # ############################################################################################################

    #     return out.argmax().item(), out
    #     ################################################################################################################################################################
    #     ################################################################################################################################################################
    #     ################################################################################################################################################################







#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
input_size = 2     # obj ~ robot-ee 상대 위치 벡터
# # input_size = 3    # obj ~ robot-ee 상대 위치 벡터 / 현재 step 
# input_size = 4   

# hidden_size_lstm = 64
hidden_size_mlp = 128
num_layers_lstm = 2
output_size = 2
# seq_len = 4

# episode_count = 4
# file_count = 0

class BC_mlp_nn(nn.Module):
    def __init__(self, input_size=input_size, hidden_size_mlp= hidden_size_mlp, output_size=output_size):
        super(BC_mlp_nn, self).__init__()

        self.action = nn.Sequential(
            nn.Linear(input_size, hidden_size_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size_mlp, hidden_size_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size_mlp, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.action(x)
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################



class IRLConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "IRL"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 5  # 'bg' + 'laptop', 'tumbler', 'drawer', 'bin', 'handcream'
    NUM_CLASSES = 1 + 7  # 'bg' + 'laptop','drawer','bin','bin_cap','tumbler','tumbler_cap','hand_cream'

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.75


class InferenceConfig(IRLConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # NUM_CLASSES = 1 + 5



class kinect_rgb_img_get():
    def __init__(self):
        rospy.init_node("YMG_Kinect_Maskrcnn")

        ############ RGB Camera Parameters ############ >>> ALSO ############ DEPTH TO RGB Camera Parameters ############ SAME
        self.ppx = 638.384765625
        self.ppy = 367.5414123535156
        self.fu = 613.1182250976562
        self.fv = 613.0377807617188
        self.camera_parameters = (self.ppx, self.ppy, self.fu, self.fv)
        ######################################################################################################################

        self.bc_agent = BC_mlp_nn()#.to(device)
        # agent = BC_RNN()#.to(device)
        print(self.bc_agent)

        # self.bc_agent.load_state_dict(torch.load(my_path + '/save_net_weight_mlp/BC_net_gan_rand_aug_200_data_epoch_212000.pt'))
        # self.bc_agent.load_state_dict(torch.load(my_path + '/save_net_weight_mlp/BC_net_epoch_49500_three_ver2.pt'))

        # self.bc_agent.load_state_dict(torch.load(my_path + '/save_net_weight_lstm/BC_net_epoch_49500_two.pt'))
        # self.bc_agent.eval()


        self.q = Qnet().to(device)

        # ######################################################################################################################
        # self.q.load_state_dict(torch.load('../weight/weights_test_0_ver2/sequence_RAINBOW_DQN_45000.pt')) # twin-Q-RAINBOW
        # ######################################################################################################################


        # ######################################################################################################################
        # self.q.load_state_dict(torch.load('../weight/weights_test_0_ver2_2/sequence_RAINBOW_DQN_50000.pt')) # twin-Q-RAINBOW
        # ######################################################################################################################

        # ######################################################################################################################
        # self.q.load_state_dict(torch.load('../weight/weights_test_0_ver2_4/sequence_RAINBOW_DQN_50000.pt')) # twin-Q-RAINBOW
        # ######################################################################################################################

        # ######################################################################################################################
        # self.q.load_state_dict(torch.load('../weight/weights_test_0_ver2_regrap/sequence_RAINBOW_DQN_50000.pt')) # twin-Q-RAINBOW
        # ######################################################################################################################

        ######################################################################################################################
        self.q.load_state_dict(torch.load('/media/irobot/8FD4-3E76/201118_white_drawer_red_tumbler_cap/sequence_RAINBOW_DQN_17600.pt')) # twin-Q-RAINBOW
        ######################################################################################################################



        self.bridge = CvBridge()
        self.image = Image()
        self.flag, self.flag_depth = False, False
        self.count = 0
        rospy.Subscriber('/kinectA/rgb/image_raw', Image, self.rgb_call_back)
        rospy.Subscriber('/kinectA/depth_to_rgb/image_raw', Image, self.depth_call_back)

        self.gripper_cmd = GripperCmd()
        self.gripper_pub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)

        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0125_201112_yellow_drawer_red_tumbler_cap.h5'
        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0200_yellow_drawer.h5'
        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0199_white_drawer.h5'
        self.IRL_MODEL_PATH = '/media/irobot/8FD4-3E76/201123_white_drawer_red_tumbler_cap/maskrcnn/mask_rcnn_irl_0380.h5'
        
        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0199_new_tumbler_cap.h5'

        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0199_1014.h5'

        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0127_1014.h5'
        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0099_new_bin.h5'
        # self.IRL_MODEL_PATH = my_path + '/mask_rcnn_irl_0099.h5'
        self.CLASS_NAMES = ['BG','laptop','drawer','bin','bin_cap','tumbler','tumbler_cap','hand_cream']

        config = InferenceConfig()

        config.display()

        self.maskrcnn_model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)

        self.maskrcnn_model.load_weights(self.IRL_MODEL_PATH, by_name=True)

        print(100)

        self.model = DeepLab(num_classes=8,                  # stefan = 7 // default: voc = 21
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=None,
                        freeze_bn=False)

        # ckpt = torch.load(my_path + '/checkpoint_201112_yellow_drawer_red_tumbler_cap.pth.tar', map_location='cpu')
        # ckpt = torch.load(my_path + '/checkpoint_yellow_drawer.pth.tar', map_location='cpu')
        # ckpt = torch.load(my_path + '/checkpoint_new_tumbler_cap.pth.tar', map_location='cpu')

        # ckpt = torch.load(my_path + '/checkpoint_201113_white_drawer.pth.tar', map_location='cpu')
        ckpt = torch.load('/media/irobot/8FD4-3E76/201123_white_drawer_red_tumbler_cap/deeplab/checkpoint.pth.tar', map_location='cpu')
        # ckpt = torch.load('/media/irobot/8FD4-3E76/201118_white_drawer_red_tumbler_cap/checkpoint.pth_201120_white_drawer_red_tumbler_cap.tar', map_location='cpu')
        # ckpt = torch.load('/media/irobot/8FD4-3E76/새 폴더 (6)/checkpoint_201112.pth.tar', map_location='cpu')
        # ckpt = torch.load(my_path + '/checkpoint_201112.pth.tar', map_location='cpu')

        # ckpt = torch.load(my_path + '/checkpoint_new_tumbler_cap.pth.tar', map_location='cpu')

        # ckpt = torch.load(my_path + '/model_best_1014.pth.tar', map_location='cpu')

        # ckpt = torch.load(my_path + '/checkpoint_7.pth.tar', map_location='cpu')

        # ckpt = torch.load(my_path + '/checkpoint_new_bin.pth.tar', map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'])


    def on_press(self, key):
        # self.on_press_key = True
        print('{0} pressed'.format(key))
        # return True
      

    def on_release(self, key):
        # self.on_press_key = False
        
        # print('{0} release'.format(
        #     key))
        print('masked image generating...')
        if key == Key.esc:
            # Stop listener
            return False
           

    def get_top_view_maskrcnn(self):

        img = self.cv2_rgb_image
       
        with sess.graph.as_default():
            pred = self.maskrcnn_model.detect([img], verbose=0)

        result = pred[0]

        pred_class = [self.CLASS_NAMES[i] for i in list(result['class_ids'])]

        img = (img * 0).astype("uint8")

        image = visualize_mg.display_instances_cv(img, result['rois'], result['masks'], result['class_ids'], self.CLASS_NAMES, result['scores'], alpha=0.)

        return image, result['rois'], result['masks'], result['class_ids'], self.CLASS_NAMES





    # ###############################################################################
    # # y : 45도 를 의미 (보정 각도) (pi/4)
    # # x : 29.5도 /  : 45도 를 의미 (보정 각도) (pi*29.5/180)
    # # 보정 각도가 의미있는게 아님 -> 잘못 생각한듯
    # #       topic   실제와의 오차   
    # # z_ee = 0.485   - 0.025    >>> 0.456
    # def correct_camera_distortion_x_y(self, what_obj, x, y, z=None):

    #     r = np.linalg.norm([x, y])
    #     # print("r : ", r)

    #     if y > 0 :
    #         theta = acos(x/r+1e-6)
    #         # print("theta : ", theta)
    #         if what_obj == 'bin_cap_bottom':
    #             z = 0.025  # bin_cap 의 바닥에서 부터의 높이

    #             x -= ((z*r) / (2*(z_ee-z))) * cos(theta)
    #             y -= ((z*r) / (2*(z_ee-z))) * sin(theta)
            

    #         elif what_obj == 'tumbler_cap':
    #             z = 0.045 # tumbler_cap의 높이

    #             x -= ((z*r) / (2*(z_ee-z))) * cos(theta)
    #             y -= ((z*r) / (2*(z_ee-z))) * sin(theta)
            

    #     elif y <= 0:
    #         theta = -acos(x/r+1e-6)
    #         # print("theta : ", theta)
    #         if what_obj == 'bin_cap_bottom':
    #             z = 0.025  # bin_cap 의 바닥에서 부터의 높이

    #             x -= ((z*r) / (2*(z_ee-z))) * cos(theta)
    #             y -= ((z*r) / (2*(z_ee-z))) * sin(theta)
                
    #         elif what_obj == 'tumbler_cap':
    #             z = 0.045 # tumbler_cap의 높이

    #             x -= ((z*r) / (2*(z_ee-z))) * cos(theta)
    #             y -= ((z*r) / (2*(z_ee-z))) * sin(theta)


    #     return x, y
    # ###############################################################################




    # ################################################################################
    # # y : 45도 를 의미 (보정 각도) (pi/4)
    # # x : 29.5도 /  : 45도 를 의미 (보정 각도) (pi*29.5/180)
    # # 보정 각도가 의미있는게 아님 -> 잘못 생각한듯
    # #       topic   실제와의 오차   
    # # z_ee = 0.485   - 0.025    >>> 0.456
    # def correct_camera_distortion_x_y(self, what_obj, x, y, z=None):

    #     r = np.linalg.norm([x, y])
    #     # print("r : ", r)

    #     theta = atan2(y, x)
    #     # theta = acos(x/r+1e-6)
    #     print("theta : ", theta)
    #     if what_obj == 'bin_cap_bottom':
    #         z = 0.025  # bin_cap 의 바닥에서 부터의 높이

    #         x += ((z*r) / (2*z_ee-z)) * cos(theta)
    #         y += ((z*r) / (2*z_ee-z)) * sin(theta)
            

    #     elif what_obj == 'tumbler_cap':
    #         z = 0.045 # tumbler_cap의 높이

    #         x += ((z*r) / (2*z_ee-z)) * cos(theta)
    #         y += ((z*r) / (2*z_ee-z)) * sin(theta)
            

    #     return x, y
    # ################################################################################


    def move_cartesian(self, pose):

        ###########################################################################################################
        maxVelocity = 0.50  # 0.310  //  0.5
        acceleration = 0.20 # # 0.310  //  0.5
        relative = True
        operatingMode = 1
        rospy.wait_for_service("/panda_target_pose")
        try:
            SetTargetpose = rospy.ServiceProxy("/panda_target_pose", SetTargetPose)
            _ = SetTargetpose(pose, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################################################
        sleep(4)


    def move_joint(self, jointPosition):

        ###########################################################################################################
        maxVelocity = 60.0  # 40  //  60
        acceleration = 30.0 # 40  //  60
        relative = False
        operatingMode = 1
        
        rospy.wait_for_service("/panda_target_joint_position")
        try:
            setTargetJointPosition = rospy.ServiceProxy("/panda_target_joint_position", SetTargetJointPosition)
            _ = setTargetJointPosition(jointPosition, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################################################
        sleep(4)



    # bias correction >>> between TCP ~ RGB-center >>> 아무래도 실제 렌즈~TCP 물리적 길이와는 좀 다른 무언가가 있는것 같음 (보통의 카메라는 이정도만 신경 썼어도 됬을거 같은데) >>> 키넥트는 너무 심하게 볼록 렌즈 / 왜곡이 너무 심해서 >>> 인위적 bias가 필요한 듯함
    # bias correction >>> between TCP ~ RGB-center >>> 아무래도 실제 렌즈~TCP 물리적 길이와는 좀 다른 무언가가 있는것 같음 (보통의 카메라는 이정도만 신경 썼어도 됬을거 같은데) >>> 키넥트는 너무 심하게 볼록 렌즈 / 왜곡이 너무 심해서 >>> 인위적 bias가 필요한 듯함
    # bias correction >>> between TCP ~ RGB-center >>> 아무래도 실제 렌즈~TCP 물리적 길이와는 좀 다른 무언가가 있는것 같음 (보통의 카메라는 이정도만 신경 썼어도 됬을거 같은데) >>> 키넥트는 너무 심하게 볼록 렌즈 / 왜곡이 너무 심해서 >>> 인위적 bias가 필요한 듯함
    def x_y_bias_correction(self, x, y):

        return x + 0.07 , y - 0.0315

        # return x + 0.065 , y - 0.0315
        # return x + 0.07 , y - 0.0315
        # return x + 0.065 , y - 0.0315

        # return x + 0.0725 , y - 0.03150

        # return x + 0.032 , y - 0.09

        # y = y_rgb #- 90.0   # 이거 맞음 거의 정확함 / 무적권 맞음 >>> TCP ~ 카메라 사이의 bias를 알고 있으니 >>> 카메라 rgb센서 중심으로 물체를 align 시키고 >>> 나중에 제어 할 때는 bias된 만큼 이동 시키면 됨
        # x = x_rgb #+ 32.0   # 이거 맞음 거의 정확함 / 무적권 맞음 >>> TCP ~ 카메라 사이의 bias를 알고 있으니 >>> 카메라 rgb센서 중심으로 물체를 align 시키고 >>> 나중에 제어 할 때는 bias된 만큼 이동 시키면 됨
 


    # 일단은 싹다 Pick&Place 로 만들었음 >>> 제대로 안되는 동작이 존재 (laptop, drawer 관련) +tumbler의 regrasping
    # 이것들은 일단은 데모영상을 찍어야 하기에 특정 경우에 대해서만 휴리스틱하게 동작을 만들어 놓고
    # 추후 로봇움직임에 대해서 뭐 통합을 하던가 해야 할 듯
    # 로보티큐85-gripper말단으로 제어 정확히 되고 있음 (말단z 실측값 이랑 ros토픽으로 측정한 z랑 동일 함)
    # 로보티큐85-gripper말단으로 제어 정확히 되고 있음 (말단z 실측값 이랑 ros토픽으로 측정한 z랑 동일 함)
    # 로보티큐85-gripper말단으로 제어 정확히 되고 있음 (말단z 실측값 이랑 ros토픽으로 측정한 z랑 동일 함)
    def step(self, action):

        if action == 0:
            print('action : 0, Close Laptop')
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 1:    # laptop의 클래스 idx : 1
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    z_obj = self.obj_position_class_goal[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                    break

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            x_0 = x_obj + 0.12          # 노트북을 닫기 위해 더 뒤로 감
            z_0 = z_obj - (0.3 + 0.06) # 0.3 차이만큼 발생함
                                       # 노트북을 닫기 위해 0.06 만큼 덜 내려감 

            self.move_to_obj(x=x_0, y=y_obj)

            self.move_up_down_obj(z=-z_0) 

            x_1 = -0.2   # 노트북을 닫기 위해 뒤로 갔던 그리퍼를 -> 앞으로 땡김 
            z_1 = 0.05   # 그와중에 약간 밑으로 당겨서 노트북 화면이 -> 닫히게 함

            self.move_to_obj(x=x_1, z=-z_1)

            self.move_up_down_obj(z=z_0+z_1) 
            self.move_to_obj(x=-(x_0+x_1), y=-y_obj)
        


        # 아래쪽 서랍 닫는 행동 임
        elif action == 1:
            print('action : 1, Close Drawer') 
            self.move_home_top_view_step() 
            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 2: # drawer의 클래스 idx : 2

                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    # z_obj = self.obj_position_class_goal[i][2]
                    # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]
                     
                    delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['drawer'] # z_obj : z = 0.41 여기서 찍은 것
                                                                                               # 원래 다른 코드들 -> z = 0.53 에서 찍음 
                    x_obj += delta_x
                    y_obj += delta_y
                    z_obj += 0.12

                    break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            for i in range(len(self.class_list_goal_cal)):
                if self.class_list_goal_cal[i] == 2: # drawer의 클래스 idx : 2

                    grasp_ang = self.obj_deg_ang_class_top_goal_calibrate[i]

                    break

            if grasp_ang > 0:
                put_donw_ang = grasp_ang+90
                # self.hold_grasp_angle(grasp_angle_deg=ang)
            else:
                put_donw_ang = grasp_ang-90
                # self.hold_grasp_angle(grasp_angle_deg=ang)

            # print("grasp_ang :", grasp_ang)
            grasp_rad = (grasp_ang/180.)*pi

            # ===================================================
            self.hold_grasp_angle(grasp_angle_deg=put_donw_ang)
            # ===================================================

            # 0.3 >>> 서랍 중심에서, 아래쪽 서랍 최대로 튀어나온곳 까지의 길이
            x_0 = x_obj - 0.33*cos(grasp_rad)
            y_0 = y_obj + 0.33*sin(grasp_rad) #+ 0.04  # 중앙 "pinch" 에 그리퍼가 충돌하는것을 방지하기 위함
            
            z_0 = 0.22  # -> 요정도는 그리퍼가 아래로 내리 꽂아야 함
            self.move_to_obj(x=x_0, y=y_0)

            self.move_up_down_obj(z=-z_0)

            # 0.16 >>> 서랍 앞우분 끝자락에서, 아래쪽 서랍 최대로 튀어나온곳 까지의 길이
            x_1 = 0.171*cos(grasp_rad)
            y_1 = -0.171*sin(grasp_rad)

            self.move_to_obj(x=x_1, y=y_1)

            self.move_up_down_obj(z=z_0)

            # ===================================================
            self.hold_grasp_angle(grasp_angle_deg=-(put_donw_ang))
            # ===================================================

            self.move_to_obj(x=-(x_0+x_1), y=-(y_0+y_1))
            
        
        #######################################################################################
        # 아직 행동 수정 안했음 -> 끝까지 안 할거 같음
        # 아직 행동 수정 안했음 -> 끝까지 안 할거 같음
        # 아직 행동 수정 안했음 -> 끝까지 안 할거 같음
        elif action == 2:
            print('action : 2, Open Drawer')
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 2: # drawer의 클래스 idx : 2
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    z_obj = self.obj_position_class_goal[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                    break

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            
            x_1 = -0.15 # 지금 아무 의미 없음
            y_1 = -0.05 # 지금 아무 의미 없음
            z_0 = 0.17  # 지금 아무 의미 없음
            self.move_to_obj(x=x_1, y=y_1)
            
            self.move_up_down_obj(z=z_0)

            self.gripper_open()

            self.move_up_down_obj(z=-z_0)

            self.move_to_obj(x=-(x_obj+x_1), y=-(y_obj+y_1))
        #######################################################################################



        #######################################################################################
        elif action == 3:
            print('action : 3, Close Bin')
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_state)):
                if self.obj_position_class_state[i][3] == 4: # bin의 클래스 idx : 3
                                                             # bin_cap의 클래스 idx : 4
                    x_obj = self.obj_position_class_state[i][0]
                    y_obj = self.obj_position_class_state[i][1]
                    z_obj = self.obj_position_class_state[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_state[i][0]

                    break

            print("state bin_cap before / x_obj : {}, y_obj : {}, z_obj : {}".format(x_obj, y_obj, z_obj))

            # ##############################################################################################
            # x_obj, y_obj = self.correct_camera_distortion_x_y(what_obj='bin_cap_bottom', x=x_obj, y=y_obj)
            # ##############################################################################################

            # print("state bin_cap after / x_obj : {}, y_obj : {}, z_obj : {}".format(x_obj, y_obj, z_obj))

            ########################################################################
            z_obj = 0.185 + 0.3 # 키넥트 depth가 튀어서 -> ㅅㅂ... / 인위적으로 내가 정했음
            ########################################################################

            self.move_to_obj(x=x_obj, y=y_obj)

            #########################################################################################################
            self.get_each_obj_mask_top_view_goal_calibrate_state(what_obj='bin_cap_bottom') # step -> x, y 보정 작업
            #########################################################################################################

            # print("state bin_cap after / x_obj : {}, y_obj : {}, z_obj : {}".format(x_obj, y_obj, z_obj))


            # ##########################################################################################################################
            # if self.bin_cap_exact_ang_step > 0 :
            #     grasp_ang = self.bin_cap_exact_ang_step
            #     # self.hold_grasp_angle(grasp_angle_deg=ang)
            # else:
            #     grasp_ang = self.bin_cap_exact_ang_step+180
            #     # self.hold_grasp_angle(grasp_angle_deg=ang)

            # print("self.bin_cap_exact_ang_step : ", self.bin_cap_exact_ang_step) # self.bin_cap_exact_ang_step :  68.60841937076145
            # ##########################################################################################################################


            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함 
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함 
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            self.move_to_obj(x=-(x_obj+self.position_x_delta), y=-(y_obj+self.position_y_delta))


            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 4: # bin의 클래스 idx : 3
                                                            # bin_cap의 클래스 idx : 4
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    # z_obj = self.obj_position_class_goal[i][2]
                    # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]
                    
                    print("goal bin_cap before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                    delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['bin_cap'] # z_obj : z = 0.41 여기서 찍은 것
                                                                                                # 원래 다른 코드들 -> z = 0.53 에서 찍음 

                    print("bin_cap calibration / delta_x : {}, delta_y : {}".format(delta_x, delta_y))

                    x_obj += delta_x
                    y_obj += delta_y
                    z_obj += 0.12

                    print("goal bin_cap after / x_obj : {}, y_obj : {}".format(x_obj, y_obj)) 

                    break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            ######################
            # x_obj = x_obj + 0.005
            # y_obj = y_obj + 0.01
            ######################

            for i in range(len(self.class_list_goal_cal)):
                if self.class_list_goal_cal[i] == 4: # bin_cap의 클래스 idx : 4

                    grasp_ang = self.obj_deg_ang_class_top_goal_calibrate[i]

                    break


            # ##########################################################################################################################
            # if grasp_ang > 0 :
            #     grasp_ang = grasp_ang
            #     # self.hold_grasp_angle(grasp_angle_deg=ang)
            # else:
            #     grasp_ang = grasp_ang+180
            #     # self.hold_grasp_angle(grasp_angle_deg=ang)

            # print("goal bin_cap grasp_ang  : ", grasp_ang) # goal bin_cap grasp_ang  :  -79.74783398792651
            # ##########################################################################################################################


            # z_obj = z_obj - 0.01   # 약간은 이격 (0.01) 을 두고 상자의 뚜겅을 상자 본체에 얹어야 함

            ########################################################################
            z_obj = 0.11 + 0.3 # 키넥트 depth가 튀어서 -> ㅅㅂ... / 인위적으로 내가 정했음
            ########################################################################
        
            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함 
            self.gripper_open()
            self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함 
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            self.move_to_obj(x=-x_obj, y=-y_obj)
        #######################################################################################



        # #######################################################################################
        # elif action == 3:
        #     print('action : 3, Close Bin')

        #     #################################
        #     self.BC_output_delta('state')
        #     #################################
            

        #     #################################
        #     self.BC_output_delta('goal')
        #     #################################
        # #######################################################################################



        elif action == 4:
            print('action : 4, Open Bin')
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 4: # bin의 클래스 idx : 3
                                                            # bin_cap의 클래스 idx : 4
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    # z_obj = self.obj_position_class_goal[i][2]
                    # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                    print("goal bin_cap before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                    delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['bin_cap'] # z_obj : z = 0.41 여기서 찍은 것
                                                                                               # 원래 다른 코드들 -> z = 0.53 에서 찍음 
                    x_obj += delta_x
                    y_obj += delta_y
                    z_obj += 0.12

                    print("goal bin_cap after / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                    break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            for i in range(len(self.class_list_goal_cal)):
                if self.class_list_goal_cal[i] == 4: # bin_cap의 클래스 idx : 4

                    grasp_ang = self.obj_deg_ang_class_top_goal_calibrate[i]

                    break

            ##########################################################################################################################
            if grasp_ang > 0 :
                grasp_ang = grasp_ang
                # self.hold_grasp_angle(grasp_angle_deg=ang)
            else:
                grasp_ang = grasp_ang+180
                # self.hold_grasp_angle(grasp_angle_deg=ang)

            print("goal bin_cap grasp_ang  : ", grasp_ang) # goal bin_cap grasp_ang  :  -79.74783398792651
            ##########################################################################################################################

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함 
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함 
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            
            ###############################################################
            x_1 = -0.23 # 닫혀있던 상자 뚜겅을 열어서 놓기 위한 위치 -> 그냥 임의로 정했음
            y_1 = 0.05  # 닫혀있던 상자 뚜겅을 열어서 놓기 위한 위치 -> 그냥 임의로 정했음

            z_1 = 0.15     # 바닥에 bin-cap을 놓기에 적절한 위치 -> 그냥 임의로 정했음
            ###############################################################

            self.move_to_obj(x=x_1, y=y_1)
            
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=(90)) # 닫혀있던 상자 뚜겅을 열어서 놓기 위한 각도 -> 그냥 임의로 정했음
            # ==============================================

            self.move_up_down_obj(z=-z_1)

            self.gripper_open()

            self.move_up_down_obj(z=z_1)

            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-(90)) # 닫혀있던 상자 뚜겅을 열어서 놓기 위한 각도 -> 그냥 임의로 정했음
            # ==============================================

            self.move_to_obj(x=-(x_obj+x_1), y=-(y_obj+y_1))



        # ##################################################################################################################################################################
        # ##################################################################################################################################################################
        # ##################################################################################################################################################################
        # elif action == 5:
        #     print('action : 5, Organize Tumbler')
        #     self.move_home_top_view_step()
        #     for i in range(len(self.obj_position_class_state)):
        #         if self.obj_position_class_state[i][3] == 6: # tumbler의 클래스 idx : 5
        #                                                      # tumbler_cap의 클래스 idx : 6
                                                            
        #             x_obj = self.obj_position_class_state[i][0]
        #             y_obj = self.obj_position_class_state[i][1]
        #             z_obj = self.obj_position_class_state[i][2]
        #             # grasp_ang = self.obj_deg_ang_class_top_state[i][0]

        #             break

        #     print("state tumbler_cap before / x_obj : {}, y_obj : {}, z_obj : {}".format(x_obj, y_obj, z_obj))

        #     # ##############################################################################################
        #     # x_obj, y_obj = self.correct_camera_distortion_x_y(what_obj='tumbler_cap', x=x_obj, y=y_obj)
        #     # ##############################################################################################

        #     # print("state tumbler_cap after / x_obj : {}, y_obj : {}, z_obj : {}".format(x_obj, y_obj, z_obj))

        #     # z_obj = z_obj + 0.01 # 텀블러 뚜껑은 좀더 깊게(0.02) 내려가야 잡을 수 있음
        #     self.move_to_obj(x=x_obj, y=y_obj)

        #     #########################################################################################################
        #     self.get_each_obj_mask_top_view_goal_calibrate_state(what_obj='tumbler_cap') # step -> x, y 보정 작업
        #     #########################################################################################################

        #     # ==============================================
        #     # self.hold_grasp_angle(grasp_angle_deg=grasp_ang) # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
        #     # ==============================================
        #     self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함
        #     self.gripper_close()
        #     self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함
        #     # ==============================================
        #     # self.hold_grasp_angle(grasp_angle_deg=-grasp_ang) # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
        #     # ==============================================
        #     self.move_to_obj(x=-(x_obj+self.position_x_delta), y=-(y_obj+self.position_y_delta))


        #     # 인식 알고리즘의 문제인데 -> goal 상태에서 -> tumbler / tumbler_cap 이 둘이 결합된 상태에서는 -> tumbler 로 통째로 인식하는거 같음
        #     for i in range(len(self.obj_position_class_goal)):
        #         if self.obj_position_class_goal[i][3] == 6: # tumbler_cap의 클래스 idx : 6                       
                                                         
        #             x_obj = self.obj_position_class_goal[i][0]
        #             y_obj = self.obj_position_class_goal[i][1]
        #             # z_obj = self.obj_position_class_goal[i][2]
        #             # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

        #             print("goal tumbler_cap before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

        #             delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['tumbler_cap'] # z_obj : z = 0.41 여기서 찍은 것
        #                                                                                             # 원래 다른 코드들 -> z = 0.53 에서 찍음 
        #                                                                                             # 첫 번째 보정
        #             print("tumbler_cap calibration first / delta_x : {}, delta_y : {}".format(delta_x, delta_y))

        #             # delta_x_sec, delta_y_sec, z_obj = self.obj_position_class_goal_calibrate_second['tumbler_cap'] # 두 번째 보정
        #             # print("tumbler_cap calibration second / delta_x : {}, delta_y : {}".format(delta_x_sec, delta_y_sec))

        #             x_obj += delta_x # 첫 번째 보정 만 적용
        #             y_obj += delta_y # 첫 번째 보정 만 적용

        #             # x_obj += delta_x+delta_x_sec # 첫 번째 보정, 두 번째 보정 모두 적용
        #             # y_obj += delta_y+delta_y_sec # 첫 번째 보정, 두 번째 보정 모두 적용
        #             z_obj += 0.12

        #             print("goal tumbler_cap after / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

        #             break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장


        #         elif self.obj_position_class_goal[i][3] == 5: # tumbler의 클래스 idx : 5                           
                                                         
        #             x_obj = self.obj_position_class_goal[i][0]
        #             y_obj = self.obj_position_class_goal[i][1]
        #             # z_obj = self.obj_position_class_goal[i][2]
        #             # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

        #             print("goal tumbler_cap before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

        #             delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['tumbler_cap'] # z_obj : z = 0.41 여기서 찍은 것
        #                                                                                             # 원래 다른 코드들 -> z = 0.53 에서 찍음 
        #                                                                                             # 첫 번째 보정
        #             print("tumbler_cap calibration first / delta_x : {}, delta_y : {}".format(delta_x, delta_y))

        #             # delta_x_sec, delta_y_sec, z_obj = self.obj_position_class_goal_calibrate_second['tumbler_cap'] # 두 번째 보정
        #             # print("tumbler_cap calibration second / delta_x : {}, delta_y : {}".format(delta_x_sec, delta_y_sec))

        #             x_obj += delta_x # 첫 번째 보정 만 적용
        #             y_obj += delta_y # 첫 번째 보정 만 적용

        #             # x_obj += delta_x+delta_x_sec # 첫 번째 보정, 두 번째 보정 모두 적용
        #             # y_obj += delta_y+delta_y_sec # 첫 번째 보정, 두 번째 보정 모두 적용
        #             z_obj += 0.12

        #             print("goal tumbler_cap after / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

        #             break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장

        #     x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

        #     # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
        #     #################################################################################
        #     # for i in range(len(self.class_list_goal_cal)):
        #     #     if self.class_list_goal_cal[i] == 6: # tumbler_cap의 클래스 idx : 6

        #     #         grasp_ang = self.obj_deg_ang_class_top_goal_calibrate[i]

        #     #         break
        #     #################################################################################

        #     z_obj = 0.08 + 0.3 # 텀블러 본체가 뻥 뚤려있어서 밑에 까지 쑥 내려감 -> 방지하기 위해 그냥 인위적으로 내려갈 길이(0.08) 를 정해줌

        #     self.move_to_obj(x=x_obj, y=y_obj)
        #     # ==============================================
        #     # self.hold_grasp_angle(grasp_angle_deg=grasp_ang)       # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
        #     # ==============================================
        #     self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함
        #     self.gripper_open()
        #     self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함
        #     # ==============================================
        #     # self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)      # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
        #     # ==============================================
        #     self.move_to_obj(x=-x_obj, y=-y_obj)
        #     ##################################################################################################################################################################
        #     ##################################################################################################################################################################
        #     ##################################################################################################################################################################



        # 카메라 RGB-센서 중심 기준으로 위치를 찾고 (물체의 중앙 위로 >>> 카메라 RGB 렌즈가 위치 하도록) >>> 마지막으로 로봇 제어 할 때만 TCP ~ RGB센서 bias를 보정해서 제어 >>> 요 작업 아직 안했음
        # 카메라 RGB-센서 중심 기준으로 위치를 찾고 (물체의 중앙 위로 >>> 카메라 RGB 렌즈가 위치 하도록) >>> 마지막으로 로봇 제어 할 때만 TCP ~ RGB센서 bias를 보정해서 제어 >>> 요 작업 아직 안했음
        # 카메라 RGB-센서 중심 기준으로 위치를 찾고 (물체의 중앙 위로 >>> 카메라 RGB 렌즈가 위치 하도록) >>> 마지막으로 로봇 제어 할 때만 TCP ~ RGB센서 bias를 보정해서 제어 >>> 요 작업 아직 안했음
        ##################################################################################################################################################################
        ##################################################################################################################################################################
        ##################################################################################################################################################################
        # 텀블러를 리그래스핑 정리 동작 (미완 / 얼추 해놓긴 했음)
        # 텀블러를 리그래스핑 정리 동작 (미완 / 얼추 해놓긴 했음)
        # 텀블러를 리그래스핑 정리 동작 (미완 / 얼추 해놓긴 했음)
        elif action == 5:
            print('action : 5, Organize Tumbler')

            # 텀블러를 리그래스핑 하고 >>> 목표 자리에 세워 놓는 것 까지 >>> 텀블러 뚜껑을 닫는 동작은 따로 또 구성 해야 함
            # 텀블러를 리그래스핑 하고 >>> 목표 자리에 세워 놓는 것 까지 >>> 텀블러 뚜껑을 닫는 동작은 따로 또 구성 해야 함
            ####################################################################################################
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_state)):
                if self.obj_position_class_state[i][3] == 5: # tumbler의 클래스 idx : 5
                                                             # tumbler_cap의 클래스 idx : 6
                    x_obj = self.obj_position_class_state[i][0]
                    y_obj = self.obj_position_class_state[i][1]
                    z_obj = self.obj_position_class_state[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_state[i][0]

            grasp_rad = (grasp_ang/180.)*pi

            # x_0 = x_obj - 0.065 * cos(grasp_rad) # 재파지 길이 -> 0.065
            # y_0 = y_obj + 0.065 * sin(grasp_rad) # 재파지 길이 -> 0.065

            self.move_to_obj(x=x_obj, y=y_obj)

            #########################################################################################################
            # self.get_each_obj_mask_top_view_goal_calibrate_state(what_obj='tumbler_cap') # step -> x, y 보정 작업
            self.get_each_obj_mask_top_view_goal_calibrate_state(what_obj='tumbler') # step -> x, y 보정 작업
            #########################################################################################################

            self.move_to_obj(x=-0.055*cos(grasp_rad), y=0.055*sin(grasp_rad))

            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            self.move_up_down_obj(z=-(z_obj+0.02-0.3)) # 0.3 차이만큼 발생함 / 0.02는 >>> 텀블러를 좀 더 내려가서 잡기 위한 것
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj+0.02-0.3))  # 0.3 차이만큼 발생함 / 0.02는 >>> 텀블러를 좀 더 내려가서 잡기 위한 것

            # 텀블러 본체 잡고 재파지 >>> 조인트 동작으로 구성함
            # 텀블러 본체 잡고 재파지 >>> 조인트 동작으로 구성함
            # 텀블러 본체 잡고 재파지 >>> 조인트 동작으로 구성함
            #############################################################################################################################################################################
            jointPosition = [4.819408795276922, -32.77849063672159, -85.93926034334815, -157.26644978616434, -55.86675163671909, 139.93437520549463, 13.058365386945379] # >>> 재파지 위치 (fixed)
            self.move_joint(jointPosition)
            #############################################################################################################################################################################

            pose1 = [0,0,0,0,35,0]      # 충돌 방지를 위해 재파지 자세 미리 어느정도 해놓음 -> 30도
            self.move_cartesian(pose1)

            pose2 = [0.095,0,0,0,0,0]    # 재파지를 하기 위해 텀블러 잡고 앞으로 이동 함 1
            self.move_cartesian(pose2)

            # pose2_1 = [0.01,0,0,0,0,0]    # 재파지를 하기 위해 텀블러 잡고 앞으로 이동 함 1_1
            # self.move_cartesian(pose2_1)

            pose2_2 = [0.02,0,0,0,0,0]    # 재파지를 하기 위해 텀블러 잡고 앞으로 이동 함 1_2
            self.move_cartesian(pose2_2)

            pose3 = [0.03,0,0.13,0,0,0]  # 재파지를 하기 위해 텀블러 잡고 앞으로 이동 함 2
            self.move_cartesian(pose3)

            pose4 = [0.1,0,-0.05,0,5,0] # 재파지를 하기 위해 텀블러 잡고 앞으로 이동 함 3 (책상위에 놓음)
            self.move_cartesian(pose4)

            self.gripper_open()         # 텀블러 내려 놓음

            pose5 = [0,0,0,0,-40,0]     # 말단 그리퍼 방위 원래 상태 (over-head) 로 다시 돌려 놓음
            self.move_cartesian(pose5)

            # pose6 = [0.005,0,0,0,0,0]    # 세워 놓은 텀블러를 파지 하기 위한 위치
            # self.move_cartesian(pose6)

            self.gripper_close()        # 텀블러 파지

            self.move_home_top_view_step() # 원상태 복귀 (회귀)


            # 잡은 텀블러-본체를 목표 위치에 놓음
            # 잡은 텀블러-본체를 목표 위치에 놓음
            # 잡은 텀블러-본체를 목표 위치에 놓음
            for i in range(len(self.obj_position_class_goal)):

                if self.obj_position_class_goal[i][3] == 6:  # tumbler_cap의 클래스 idx : 6                          
                                                            
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    # z_obj = self.obj_position_class_goal[i][2]
                    # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                    print("goal tumbler before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                    delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['tumbler_cap'] # z_obj : z = 0.41 여기서 찍은 것
                                                                                                    # 원래 다른 코드들 -> z = 0.53 에서 찍음 
                                                                                                    # 첫 번째 보정
                    print("tumbler calibration first / delta_x : {}, delta_y : {}".format(delta_x, delta_y))

                    # delta_x_sec, delta_y_sec, z_obj = self.obj_position_class_goal_calibrate_second['tumbler_cap'] # 두 번째 보정
                    # print("tumbler_cap calibration second / delta_x : {}, delta_y : {}".format(delta_x_sec, delta_y_sec))

                    x_obj += delta_x # 첫 번째 보정 만 적용
                    y_obj += delta_y # 첫 번째 보정 만 적용

                    # x_obj += delta_x+delta_x_sec # 첫 번째 보정, 두 번째 보정 모두 적용
                    # y_obj += delta_y+delta_y_sec # 첫 번째 보정, 두 번째 보정 모두 적용
                    z_obj += 0.12

                    print("goal tumbler after / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                    break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장


                ################################################################################################################
                # elif self.obj_position_class_goal[i][3] == 5:  # tumbler의 클래스 idx : 5                                     
                                                    
                #     x_obj = self.obj_position_class_goal[i][0]
                #     y_obj = self.obj_position_class_goal[i][1]
                #     # z_obj = self.obj_position_class_goal[i][2]
                #     # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                #     print("goal tumbler before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                #     delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['tumbler_cap'] # z_obj : z = 0.41 여기서 찍은 것
                #                                                                                     # 원래 다른 코드들 -> z = 0.53 에서 찍음 
                #                                                                                     # 첫 번째 보정
                #     print("tumbler calibration first / delta_x : {}, delta_y : {}".format(delta_x, delta_y))

                #     # delta_x_sec, delta_y_sec, z_obj = self.obj_position_class_goal_calibrate_second['tumbler_cap'] # 두 번째 보정
                #     # print("tumbler_cap calibration second / delta_x : {}, delta_y : {}".format(delta_x_sec, delta_y_sec))

                #     x_obj += delta_x # 첫 번째 보정 만 적용
                #     y_obj += delta_y # 첫 번째 보정 만 적용

                #     # x_obj += delta_x+delta_x_sec # 첫 번째 보정, 두 번째 보정 모두 적용
                #     # y_obj += delta_y+delta_y_sec # 첫 번째 보정, 두 번째 보정 모두 적용
                #     z_obj += 0.12

                #     print("goal tumbler after / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                #     break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장
                ################################################################################################################


            z_obj = 0.127 + 0.3 # 그냥 인위적으로 내려갈 길이(0.08) 를 정해줌

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            self.move_to_obj(x=x_obj, y=y_obj)
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함 / -0.02는 >>> 반대로 좀 여유를 두고 텀블러를 내려놓기 위함

            self.gripper_open()                        # 텀블러 내려 놓음

            self.move_up_down_obj(z=(z_obj-0.3))  # 0.3 차이만큼 발생함 / -0.02는 >>> 반대로 좀 여유를 두고 텀블러를 내려놓기 위함

            self.move_home_top_view_step()  # 원상태 복귀 (회귀)
            ####################################################################################################


            # 텀블러 뚜껑을 >>> 텀블러 본체에 닫기 위한 동작
            # 텀블러 뚜껑을 >>> 텀블러 본체에 닫기 위한 동작
            # 텀블러 뚜껑을 >>> 텀블러 본체에 닫기 위한 동작
            for i in range(len(self.obj_position_class_state)):
                if self.obj_position_class_state[i][3] == 6: # tumbler의 클래스 idx : 5
                                                             # tumbler_cap의 클래스 idx : 6
                                                            
                    x_obj = self.obj_position_class_state[i][0]
                    y_obj = self.obj_position_class_state[i][1]
                    z_obj = self.obj_position_class_state[i][2]
                    # grasp_ang = self.obj_deg_ang_class_top_state[i][0]

                    break

            print("state tumbler_cap before / x_obj : {}, y_obj : {}, z_obj : {}".format(x_obj, y_obj, z_obj))

            # print("state tumbler_cap after / x_obj : {}, y_obj : {}, z_obj : {}".format(x_obj, y_obj, z_obj))

            # z_obj = z_obj + 0.02  # 텀블러 뚜껑은 좀더 깊게(0.02) 내려가야 잡을 수 있음
            self.move_to_obj(x=x_obj, y=y_obj)

            #########################################################################################################
            self.get_each_obj_mask_top_view_goal_calibrate_state(what_obj='tumbler_cap') # step -> x, y 보정 작업
            #########################################################################################################

            # ==============================================
            # self.hold_grasp_angle(grasp_angle_deg=grasp_ang) # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함
            # ==============================================
            # self.hold_grasp_angle(grasp_angle_deg=-grasp_ang) # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
            # ==============================================
            self.move_to_obj(x=-(x_obj+self.position_x_delta), y=-(y_obj+self.position_y_delta))


            #########################################################################################################
            # # 인식 알고리즘의 문제인데 -> goal 상태에서 -> tumbler / tumbler_cap 이 둘이 결합된 상태에서는 -> tumbler 로 통째로 인식하는거 같음
            # for i in range(len(self.obj_position_class_goal)):
            #     if self.obj_position_class_goal[i][3] == 5: # tumbler의 클래스 idx : 5
            #                                                 # tumbler_cap의 클래스 idx : 6
                                                         
            #         x_obj = self.obj_position_class_goal[i][0]
            #         y_obj = self.obj_position_class_goal[i][1]
            #         # z_obj = self.obj_position_class_goal[i][2]
            #         # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

            #         print("goal tumbler_cap before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

            #         delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['tumbler_cap'] # z_obj : z = 0.41 여기서 찍은 것
            #                                                                                         # 원래 다른 코드들 -> z = 0.53 에서 찍음 
            #                                                                                         # 첫 번째 보정

            #         print("tumbler_cap calibration first / delta_x : {}, delta_y : {}".format(delta_x, delta_y))

            #         # 솔직히 두번이나 보정하는게 의미가 없는거 같다
            #         # 솔직히 두번이나 보정하는게 의미가 없는거 같다
            #         # 솔직히 두번이나 보정하는게 의미가 없는거 같다
            #         # delta_x_sec, delta_y_sec, z_obj = self.obj_position_class_goal_calibrate_second['tumbler_cap'] # 두 번째 보정
            #         # print("tumbler_cap calibration second / delta_x : {}, delta_y : {}".format(delta_x_sec, delta_y_sec))

            #         x_obj += delta_x # 첫 번째 보정 만 적용
            #         y_obj += delta_y # 첫 번째 보정 만 적용

            #         # x_obj += delta_x+delta_x_sec # 첫 번째 보정, 두 번째 보정 모두 적용
            #         # y_obj += delta_y+delta_y_sec # 첫 번째 보정, 두 번째 보정 모두 적용
            #         z_obj += 0.12

            #         print("goal tumbler_cap after / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

            #         break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장
            #########################################################################################################


            # 인식 알고리즘의 문제인데 -> goal 상태에서 -> tumbler / tumbler_cap 이 둘이 결합된 상태에서는 -> tumbler 로 통째로 인식하는거 같음
            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 6: # tumbler의 클래스 idx : 5
                                                            # tumbler_cap의 클래스 idx : 6
                                                         
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    # z_obj = self.obj_position_class_goal[i][2]
                    # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                    print("goal tumbler_cap before / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                    delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['tumbler_cap'] # z_obj : z = 0.41 여기서 찍은 것
                                                                                                    # 원래 다른 코드들 -> z = 0.53 에서 찍음 
                                                                                                    # 첫 번째 보정

                    print("tumbler_cap calibration first / delta_x : {}, delta_y : {}".format(delta_x, delta_y))

                    # 솔직히 두번이나 보정하는게 의미가 없는거 같다
                    # 솔직히 두번이나 보정하는게 의미가 없는거 같다
                    # 솔직히 두번이나 보정하는게 의미가 없는거 같다
                    # delta_x_sec, delta_y_sec, z_obj = self.obj_position_class_goal_calibrate_second['tumbler_cap'] # 두 번째 보정
                    # print("tumbler_cap calibration second / delta_x : {}, delta_y : {}".format(delta_x_sec, delta_y_sec))

                    x_obj += delta_x # 첫 번째 보정 만 적용
                    y_obj += delta_y # 첫 번째 보정 만 적용

                    # x_obj += delta_x+delta_x_sec # 첫 번째 보정, 두 번째 보정 모두 적용
                    # y_obj += delta_y+delta_y_sec # 첫 번째 보정, 두 번째 보정 모두 적용
                    z_obj += 0.12

                    print("goal tumbler_cap after / x_obj : {}, y_obj : {}".format(x_obj, y_obj))

                    break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장


            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            # #####################################
            # x_obj = x_obj + 0.005  # 인위적 보정
            # #####################################

            # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
            #################################################################################
            # for i in range(len(self.class_list_goal_cal)):
            #     if self.class_list_goal_cal[i] == 6: # tumbler_cap의 클래스 idx : 6

            #         grasp_ang = self.obj_deg_ang_class_top_goal_calibrate[i]

            #         break
            #################################################################################

            z_obj = 0.08 + 0.3 # 텀블러 본체가 뻥 뚤려있어서 밑에 까지 쑥 내려감 -> 방지하기 위해 그냥 인위적으로 내려갈 길이(0.08) 를 정해줌

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            # self.hold_grasp_angle(grasp_angle_deg=grasp_ang)       # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함
            self.gripper_open()
            self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함
            # ==============================================
            # self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)      # >>> tumbler_cap 에 대해서는 어떻게 잡는지 방위 설정을 굳이 하지 않음
            # ==============================================
            self.move_to_obj(x=-x_obj, y=-y_obj)
            ##################################################################################################################################################################
            ##################################################################################################################################################################
            ##################################################################################################################################################################


        elif action == 6:
            print('action : 6, Organize Hand_cream on the table')
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_state)):
                if self.obj_position_class_state[i][3] == 7: # hand_cream의 클래스 idx : 7
                                                         
                    x_obj = self.obj_position_class_state[i][0]
                    y_obj = self.obj_position_class_state[i][1]
                    z_obj = self.obj_position_class_state[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_state[i][0]

                    break

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3)) # 0.3 차이만큼 발생함
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj-0.3)) # 0.3 차이만큼 발생함
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            self.move_to_obj(x=-x_obj, y=-y_obj)
			

            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 7: # hand_cream의 클래스 idx : 7
                                                         
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    z_obj = self.obj_position_class_goal[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_goal[i][0]
                    
                    break

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3-0.03)) # 0.3 차이만큼 발생함 / 약간 위로 띄어놓은 채로 핸드크림을 놓아야 함 -> 0.03
            self.gripper_open()
            self.move_up_down_obj(z=(z_obj-0.3-0.03)) # 0.3 차이만큼 발생함 / 약간 위로 띄어놓은 채로 핸드크림을 놓아야 함 -> 0.03
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            self.move_to_obj(x=-x_obj, y=-y_obj)


        elif action == 7:
            print('action : 7, Organize Hand_cream in the drawer')
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_state)):
                if self.obj_position_class_state[i][3] == 7: # hand_cream의 클래스 idx : 7
                                                            
                    x_obj = self.obj_position_class_state[i][0]
                    y_obj = self.obj_position_class_state[i][1]
                    z_obj = self.obj_position_class_state[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_state[i][0]
                    
                    break

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3))  # 0.3 차이만큼 발생함
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj-0.3))  # 0.3 차이만큼 발생함
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            self.move_to_obj(x=-x_obj, y=-y_obj)
            

            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 2: # drawer의 클래스 idx : 2
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    # z_obj = self.obj_position_class_goal[i][2]
                    # grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                    delta_x, delta_y, z_obj = self.obj_position_class_goal_calibrate['drawer'] # z_obj : z = 0.41 여기서 찍은 것
                                                                                               # 원래 다른 코드들 -> z = 0.53 에서 찍음 
                    x_obj += delta_x
                    y_obj += delta_y
                    z_obj += 0.12

                    break # drawer를 중복으로 검출하는 현상을 막아줌 -> 맨 처음 current-state에서 drawer를 검출 했을때의 좌표/방위 를 저장

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            for i in range(len(self.class_list_goal_cal)):
                if self.class_list_goal_cal[i] == 2: # drawer의 클래스 idx : 2

                    grasp_ang = self.obj_deg_ang_class_top_goal_calibrate[i]

                    break

            print("drawer state grasp_ang :", grasp_ang)
            grasp_rad = (grasp_ang/180.)*pi

            # 0.21 >>> 핸드크림 놓을 장소
            x_0 = x_obj - 0.23*cos(grasp_rad)
            y_0 = y_obj + 0.23*sin(grasp_rad) 
            z_0 = 0.12  # 0.12m -> 이 정도는 내려가고 나서 핸드크림을 서랍안에 투하 해야 함 / 이것도 그냥 인위적으로 정한 길이

            self.move_to_obj(x=x_0, y=y_0)

            # ===================================================
            self.hold_grasp_angle(grasp_angle_deg=(grasp_ang))
            # ===================================================

            self.move_up_down_obj(z=-z_0)

            self.gripper_open()

            self.move_up_down_obj(z=z_0)

            # ===================================================
            self.hold_grasp_angle(grasp_angle_deg=-(grasp_ang))
            # ===================================================

            self.move_to_obj(x=-x_0, y=-y_0)



        # 요 행동은 굳이 x,y 보정 안해도 -> 잘 굴러갔으니 -> 보정 생략함
        # 요 행동은 굳이 x,y 보정 안해도 -> 잘 굴러갔으니 -> 보정 생략함
        # 요 행동은 굳이 x,y 보정 안해도 -> 잘 굴러갔으니 -> 보정 생략함
        elif action == 8:
            print('action : 8, Organize Hand_cream in the bin')
            self.move_home_top_view_step()
            for i in range(len(self.obj_position_class_state)):
                if self.obj_position_class_state[i][3] == 7: # hand_cream의 클래스 idx : 7
                                                            
                    x_obj = self.obj_position_class_state[i][0]
                    y_obj = self.obj_position_class_state[i][1]
                    z_obj = self.obj_position_class_state[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_state[i][0]

                    break

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=grasp_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.3-0.02)) # 0.3 차이만큼 발생함
            self.gripper_close()
            self.move_up_down_obj(z=(z_obj-0.3-0.02)) # 0.3 차이만큼 발생함
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-grasp_ang)
            # ==============================================
            self.move_to_obj(x=-x_obj, y=-y_obj)
            

            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 4: # bin의 클래스 idx : 3
                                                            # bin_cap의 클래스 idx : 4
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    z_obj = self.obj_position_class_goal[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_goal[i][0]

                    break

            if grasp_ang > 0 :
                put_donw_ang = grasp_ang-90
                # self.hold_grasp_angle(grasp_angle_deg=ang)
            else:
                put_donw_ang = grasp_ang+90
                # self.hold_grasp_angle(grasp_angle_deg=ang)

            x_obj, y_obj = self.x_y_bias_correction(x_obj, y_obj)

            self.move_to_obj(x=x_obj, y=y_obj)
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=put_donw_ang)
            # ==============================================
            self.move_up_down_obj(z=-(z_obj-0.015-0.3)) # 0.3 차이만큼 발생함 / 핸드크림을 너무 높은곳에서 떨어뜨리면 안됨
            self.gripper_open()
            self.move_up_down_obj(z=(z_obj-0.015-0.3)) # 0.3 차이만큼 발생함  / 핸드크림을 너무 높은곳에서 떨어뜨리면 안됨
            # ==============================================
            self.hold_grasp_angle(grasp_angle_deg=-(put_donw_ang))
            # ==============================================
            self.move_to_obj(x=-x_obj, y=-y_obj)


    def gripper_open(self):
        gripper_cmd = self.gripper_cmd
        gripper_cmd.position = 2.0
        gripper_cmd.speed = 0.5   # 1.0
        gripper_cmd.force = 15.0  # 30.0
        self.gripper_pub.publish(gripper_cmd)
        sleep(1)
 
    def gripper_close(self):
        gripper_cmd = self.gripper_cmd
        gripper_cmd.position = 0.0
        gripper_cmd.speed = 0.5   # 1.0
        gripper_cmd.force = 15.0  # 30.0
        self.gripper_pub.publish(gripper_cmd)
        sleep(1)
    
    def gripper_set(self, position):
        gripper_cmd = self.gripper_cmd
        gripper_cmd.position = position
        gripper_cmd.speed = 0.5   # 1.0
        gripper_cmd.force = 15.0  # 30.0
        self.gripper_pub.publish(gripper_cmd)
        sleep(1)


    # def is_control_finished(self, req):

    #     return IsControlFinResponse()

    # def is_control_finished_server(self):

    #     ###########################################################################################################

    #     IsControlFin = rospy.Service("/panda_control_status", IsControlFin, is_control_finished)

    #     ###########################################################################################################
    #     sleep(3)

    
    def move_home_forward_view(self):

        jointPosition = [0.0, -99.99999999999999, 0.0, -125.00000000000001, 0.0, 67.0, 45.0]

        ###########################################################################################################
        maxVelocity = 60.0  # 40 / 60
        acceleration = 60.0 # 40 / 60
        relative = False
        operatingMode = 1
        
        rospy.wait_for_service("/panda_target_joint_position")
        try:
            setTargetJointPosition = rospy.ServiceProxy("/panda_target_joint_position", SetTargetJointPosition)
            _ = setTargetJointPosition(jointPosition, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################################################
        sleep(4)

    
    def move_home_top_view(self):
        jointPosition = [13.020751514829222, 3.0911463403694452, -11.485064245530623, -67.14796249924379, 0.6543278622805229, 70.17826043419628, 46.77355870988202]
        # 13.020751514829222, 3.0911463403694452, -11.485064245530623, -67.14796249924379, 0.6543278622805229, 70.17826043419628, 46.77355870988202 >>> 0.5200015294445399, 0.020006394372710703, 0.5299980557318252, -179.99964750309863, 0.0002713087858300326, 0.0002810072073839687
        ###########################################################################################################                         
        maxVelocity = 60.0  # 40 / 60
        acceleration = 60.0 # 40 / 60
        relative = False
        operatingMode = 1

        rospy.wait_for_service("/panda_target_joint_position")
        try:
            setTargetJointPosition = rospy.ServiceProxy("/panda_target_joint_position", SetTargetJointPosition)
            _ = setTargetJointPosition(jointPosition, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################################################
        sleep(4)


    # z >>> 0.25 or 0.3 차이 -> 확인 필요 : 0.3 맞음 (10/12에 확인한 결과)
    # z >>> 0.25 or 0.3 차이 -> 확인 필요 : 0.3 맞음 (10/12에 확인한 결과)
    # z >>> 0.25 or 0.3 차이 -> 확인 필요 : 0.3 맞음 (10/12에 확인한 결과)
    def move_home_top_view_step(self):
        jointPosition = [13.511903525054915, -5.317056313255498, -11.016751553101791, -122.57277741116432, -1.1419482880011687, 117.34826134503739, 48.06578979107195]
        # 13.511903525054915, -5.317056313255498, -11.016751553101791, -122.57277741116432, -1.1419482880011687, 117.34826134503739, 48.06578979107195 >>> 0.5200015294445407, 0.02000639437271071, 0.22999805573182516, -179.99964750309863, 0.0002713087858202495, 0.0002810072073807882
                                                                                                                                              
        ###########################################################################################################                   
        maxVelocity = 60.0  # 40 / 60
        acceleration = 60.0 # 40 / 60
        relative = False
        operatingMode = 1

        rospy.wait_for_service("/panda_target_joint_position")
        try:
            setTargetJointPosition = rospy.ServiceProxy("/panda_target_joint_position", SetTargetJointPosition)
            _ = setTargetJointPosition(jointPosition, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################################################
        sleep(4)

    # 최대로 올라가서 top-view => 에서 찍었을 때 -> z = 0.53
    # goal의 x,y의 -> calibration하기 위한 -> z = 0.41
    # step 명령 도중에는 -> -> z = 0.23
    def move_home_top_view_goal_calibrate(self):
        jointPosition = [13.718612936778248, -6.475815295032881, -10.668096107169022, -97.34970068658025, -1.1961453629173349, 90.98442036576026, 48.137313942651566]
        # 13.718612936778248, -6.475815295032881, -10.668096107169022, -97.34970068658025, -1.1961453629173349, 90.98442036576026, 48.137313942651566 >>> 0.5200015294445407, 0.02000639437271075, 0.40999805573182524, -179.99964750309863, 0.00027130878580737755, 0.00028100720737442715
                                                                                                                                              
        ###########################################################################################################                   
        maxVelocity = 60.0  # 40 / 60
        acceleration = 30.0 # 40 / 60
        relative = False
        operatingMode = 1

        rospy.wait_for_service("/panda_target_joint_position")
        try:
            setTargetJointPosition = rospy.ServiceProxy("/panda_target_joint_position", SetTargetJointPosition)
            _ = setTargetJointPosition(jointPosition, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################################################
        sleep(4)


    def hold_grasp_angle(self, grasp_angle_deg):
        jointPosition = [0, 0, 0, 0, 0, 0, grasp_angle_deg]

        ###########################################################################################################
        maxVelocity = 60.0  # 40 / 60
        acceleration = 30.0 # 40 / 60
        relative = True
        operatingMode = 1

        rospy.wait_for_service("/panda_target_joint_position")
        try:
            setTargetJointPosition = rospy.ServiceProxy("/panda_target_joint_position", SetTargetJointPosition)
            _ = setTargetJointPosition(jointPosition, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################################################
        sleep(4)


    # 로봇 말단 기준으로 움직임 -> "relative = True"
    # 움직임의 방향은 판다로봇의 base 좌표계의 방향으로 (카메라 좌표계 방향으로 움직이는것이 아님)
    def move_to_obj(self, x=0, y=0, z=0):

        pose = [x, y, z, 0, 0, 0]

        ###########################################################################
        maxVelocity = 0.50  # 0.310 / 0.40
        acceleration = 0.20 # 0.310 / 0.40
        relative = True
        operatingMode = 1

        rospy.wait_for_service("/panda_target_pose")
        try:
            SetTargetpose = rospy.ServiceProxy("/panda_target_pose", SetTargetPose)
            _ = SetTargetpose(pose, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################
        sleep(4)


    # 로봇 말단 기준으로 움직임 -> "relative = True"
    # 움직임의 방향은 판다로봇의 base 좌표계의 방향으로 (카메라 좌표계 방향으로 움직이는것이 아님)
    def move_up_down_obj(self, z=0):

        pose = [0, 0, z, 0, 0, 0]

        ###########################################################################
        maxVelocity = 0.50  # 0.310 / 0.40
        acceleration = 0.20 # 0.310 / 0.40
        relative = True
        operatingMode = 1

        rospy.wait_for_service("/panda_target_pose")
        try:
            SetTargetpose = rospy.ServiceProxy("/panda_target_pose", SetTargetPose)
            _ = SetTargetpose(pose, maxVelocity, acceleration, relative, operatingMode)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ###########################################################################
        sleep(4)


    def depth_interpolation(self):
        depth = self.cv2_depth_image
        mask = np.zeros(depth.shape)
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                if depth[i,j] == 0 :
                    mask[i,j] = 255
        mask = mask.astype(np.uint8)
        # depth_tmp = depth_tmp.astype(np.uint8)
        dst = cv2.inpaint(depth ,mask, 3, flags=cv2.INPAINT_TELEA)
        self.depth = dst


    def get_pose_class_goal(self):
        self.depth_interpolation()

        self.obj_position_class_goal = []

        n = len(self.box_list_goal)
        for i in range(n):
            y1, x1, y2, x2 = self.box_list_goal[i]
            x_pix = int((x1 + x2)/2)
            y_pix = int((y1 + y2)/2)

            obj_pix = (x_pix, y_pix)
            z_cam = self.depth[y_pix, x_pix]
            # z_cam = self.cv2_depth_image[y_pix, x_pix]
            
            position_x, position_y = self.pixel_to_tool(self.camera_parameters, obj_pix, z_cam)
            
            # position_z = z_cam - (0.12 + 0.025) # 이게 정확함 / 이거 설마 키넥트 rgb센서 렌즈가 안으로 움푹 패여서 그런가;;
            position_z = z_cam - 0.12   # 이게 정확함
            # position_z = z_cam - (0.12 - 0.025)

            print('box_goal / Class : ', y1, x1, y2, x2, " / ", self.class_list_goal[i])
            print('pixel_goal / Class : ', x_pix, y_pix, " / ", self.class_list_goal[i])
            print('z_cam_goal / Class : ', z_cam, " / ", self.class_list_goal[i])
            print('POS_goal / Class : ', -position_y, -position_x, position_z, " / ", self.class_list_goal[i])

            self.obj_position_class_goal.append((-position_y, -position_x, position_z, self.class_list_goal[i])) # 판다 베이스 좌표계와 카메라 좌표계의 방향이 다르기 때문


    # -> 서랍 / 2
    # -> 상자 뚜껑 / 4
    # -> 텀블러 뚜겅 / 6 (근데 텀블러 본체(5) 를 찾고 -> 그 위치에 가서 텀블러 뚜겅(6)을 찾아야 할 듯)
    def get_pose_class_goal_calibrate(self):
        self.depth_interpolation()

        self.obj_position_class_goal_calibrate = dict()

        n = len(self.box_list_goal_cal)
        for i in range(n):
            y1, x1, y2, x2 = self.box_list_goal_cal[i]
            x_pix = int((x1 + x2)/2)
            y_pix = int((y1 + y2)/2)

            obj_pix = (x_pix, y_pix)
            z_cam = self.depth[y_pix, x_pix]
            # z_cam = self.cv2_depth_image[y_pix, x_pix]

            position_x, position_y = self.pixel_to_tool(self.camera_parameters, obj_pix, z_cam)
            
            # position_z = z_cam - (0.12 + 0.025) # 이게 정확함 / 이거 설마 키넥트 rgb센서 렌즈가 안으로 움푹 패여서 그런가;;
            position_z = z_cam - 0.12   # 이게 정확함
            # position_z = z_cam - (0.12 - 0.025)

            print('POS_goal_calibrated / Class : ', -position_y, -position_x, position_z, " / ", self.class_list_goal_cal[i])

            # drawer x,y - calibration => delta
            if self.class_list_goal_cal[i] == 2:
                self.obj_position_class_goal_calibrate['drawer'] = (-position_y, -position_x, position_z)

            # bin_cap x,y - calibration => delta
            elif self.class_list_goal_cal[i] == 4:
                self.obj_position_class_goal_calibrate['bin_cap'] = (-position_y, -position_x, position_z)


            # #################################################################################################
            # # tumbler_cap x,y - calibration => delta
            # elif self.class_list_goal_cal[i] == 5:
            #     self.obj_position_class_goal_calibrate['tumbler_cap'] = (-position_y, -position_x, position_z)
            # #################################################################################################


            # tumbler_cap x,y - calibration => delta
            elif self.class_list_goal_cal[i] == 6:
                self.obj_position_class_goal_calibrate['tumbler_cap'] = (-position_y, -position_x, position_z)



    # # -> 텀블러 뚜겅 / 6 (근데 텀블러 본체(5) 를 찾고 -> 그 위치에 가서 텀블러 뚜겅(6)을 찾아야 할 듯)
    # # 텀블러는 2번 보정해라~~
    # # 텀블러는 2번 보정해라~~
    # # 텀블러는 2번 보정해라~~
    # def get_pose_class_goal_calibrate_second(self):
    #     # self.depth_interpolation()

    #     self.obj_position_class_goal_calibrate_second = dict()

    #     n = len(self.box_list_goal_cal_second)
    #     for i in range(n):
    #         y1, x1, y2, x2 = self.box_list_goal_cal_second[i]
    #         x_pix = int((x1 + x2)/2)
    #         y_pix = int((y1 + y2)/2)

    #         obj_pix = (x_pix, y_pix)
    #         z_cam = self.depth[y_pix, x_pix]
    #         z_cam = self.cv2_depth_image[y_pix, x_pix]

    #         position_x, position_y = self.pixel_to_tool(self.camera_parameters, obj_pix, z_cam)
            
    #         position_z = z_cam - (0.12 + 0.025) # 이게 정확함 / 이거 설마 키넥트 rgb센서 렌즈가 안으로 움푹 패여서 그런가;;
    #         # position_z = z_cam - 0.12   # 이게 정확함
    #         # position_z = z_cam - (0.12 - 0.025)

    #         print('POS_goal_calibrated_second / Class : ', -position_y, -position_x, position_z, " / ", self.class_list_goal_cal_second[i])

    #         # # drawer x,y - calibration => delta
    #         # if self.class_list_goal_cal[i] == 2:
    #         #     self.obj_position_class_goal_calibrate_second['drawer'] = (-position_y, -position_x, position_z)

    #         # # bin_cap x,y - calibration => delta
    #         # elif self.class_list_goal_cal[i] == 4:
    #         #     self.obj_position_class_goal_calibrate_second['bin_cap'] = (-position_y, -position_x, position_z)

    #         # tumbler_cap x,y - calibration => delta
    #         if self.class_list_goal_cal_second[i] == 6:
    #             self.obj_position_class_goal_calibrate_second['tumbler_cap'] = (-position_y, -position_x, position_z)



    def get_pose_class_state(self):
        self.depth_interpolation()

        self.obj_position_class_state = []

        n = len(self.box_list_state)
        for i in range(n):
            y1, x1, y2, x2 = self.box_list_state[i]
            x_pix = int((x1 + x2)/2)
            y_pix = int((y1 + y2)/2)

            obj_pix = (x_pix, y_pix)
            z_cam = self.depth[y_pix, x_pix]
            # z_cam = self.cv2_depth_image[y_pix, x_pix]

            position_x, position_y = self.pixel_to_tool(self.camera_parameters, obj_pix, z_cam)

            # position_z = z_cam - (0.12 + 0.025) # 이게 정확함 / 이거 설마 키넥트 rgb센서 렌즈가 안으로 움푹 패여서 그런가;;
            position_z = z_cam - 0.12  # 이게 정확함
            # position_z = z_cam - (0.12 - 0.025)

            print('POS_state / Class : ', -position_y, -position_x, position_z, " / ", self.class_list_state[i])

            self.obj_position_class_state.append((-position_y, -position_x, position_z, self.class_list_state[i])) # 판다 베이스 좌표계와 카메라 좌표계의 방향이 다르기 때문



    # 즉, 카메라 RGB-센서를 중심으로 물체의 위치를 찾고 >>> 물체의 중앙(중심) 위치를 찾고 나서 >>> 최종적 로봇 제어 할 때만 >>> bias-correction 작업이 필요
    # x >>> x - 0.09
    # y >>> y + 0.032
    def pixel_to_tool(self, cam_param, pos, pos_z):
        ppx, ppy, fu, fv = cam_param
        x_pix = pos[0]
        y_pix = pos[1]
        z_rgb = pos_z * 1000
        x_rgb = z_rgb*(x_pix-ppx)/fu # mm
        y_rgb = z_rgb*(y_pix-ppy)/fv # mm
        y = y_rgb #- 90.0   # 이거 맞음 거의 정확함 / 무적권 맞음 >>> TCP ~ 카메라 사이의 bias를 알고 있으니 >>> 카메라 rgb센서 중심으로 물체를 align 시키고 >>> 나중에 제어 할 때는 bias된 만큼 이동 시키면 됨
        x = x_rgb #+ 32.0   # 이거 맞음 거의 정확함 / 무적권 맞음 >>> TCP ~ 카메라 사이의 bias를 알고 있으니 >>> 카메라 rgb센서 중심으로 물체를 align 시키고 >>> 나중에 제어 할 때는 bias된 만큼 이동 시키면 됨
 
        # z = z_rgb - 184.0
        return x/1000, y/1000
 

    def rgb_call_back(self, ros_img):

        self.cv2_rgb_image = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        
        self.flag = True



    def depth_call_back(self, ros_img):

        self.cv2_depth_image = self.bridge.imgmsg_to_cv2(ros_img, "passthrough")
        
        self.flag_depth = True



    def forward_deeplab(self, whether):
        self.move_home_forward_view()

        if whether == "goal":
            cv2_rgb_goal_ = self.cv2_rgb_image[47:647, 63:1183]
            cv2.imwrite('../test_experiments/test_{}/rgb/goal_img/kinect_goal_{}_img.jpg'.format(self.test_num,self.test_num), cv2_rgb_goal_)
        elif whether == "state":
            cv2_rgb_state_ = self.cv2_rgb_image[47:647, 63:1183]
            cv2.imwrite('../test_experiments/test_{}/rgb/state_img/kinect_state_{}_{}_img.jpg'.format(self.test_num,self.test_num,self.count), cv2_rgb_state_)
        
        img = cv2.cvtColor(self.cv2_rgb_image, cv2.COLOR_BGR2RGB)
        image = Im.fromarray(img).convert('RGB')
        target = Im.fromarray(img).convert('L')

        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_in)

        return output



    ########################################################################################
    ########################################################################################
    ########################################################################################
    # def get_top_view(self):
        
    #     img = cv2.cvtColor(self.cv2_rgb_image, cv2.COLOR_BGR2RGB)
    #     image = Im.fromarray(img).convert('RGB')
    #     target = Im.fromarray(img).convert('L')

    #     sample = {'image': image, 'label': target}
    #     tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

    #     self.model.eval()
    #     with torch.no_grad():
    #         output = self.model(tensor_in)
    #         # print("output.shape :", output.shape) # ('output.shape :', (1, 8, 720, 1280))
    #         # print("output[:3].shape :", output[:3].shape) # ('output[:3].shape :', (1, 8, 720, 1280))

    #     return output
    ########################################################################################
    ########################################################################################
    ########################################################################################



    ########################################################################################################################################################################################################
    ########################################################################################################################################################################################################
    ########################################################################################################################################################################################################
    # # [0  ,   0,   0],        # black         # BG
    # # [255, 255, 255],        # white         # laptop
    # # [255, 255,   0],        # yellow        # drawer
    # # [  0,   0, 255],        # blue          # bin
    # # [  0,   0, 255],        # blue          # bin_cap
    # # [255,   0,   0],        # red           # tumbler    
    # # [255,   0,   0],        # red           # tumbler_cap
    # # [  0, 255,   0]])       # green         # hand_cream
    # def get_each_obj_mask_top_view(self, whether):
    #     self.move_home_top_view()
        
    #     ########################################################
    #     # output = self.get_top_view_maskrcnn()   # mask_rcnn
    #     output = self.get_top_view()          # deeplab
    #     ########################################################

    #     cv2_rgb_top = decode_seg_map_sequence_ymg(torch.max(output[:3], 1)[1].detach().cpu().numpy(), 'known_7')
        
    #     if whether == 'goal':
    #         cv2.imwrite('../test_experiments/test_{}/seg/top_img_goal/kinect_top_{}_img.jpg'.format(self.test_num,self.test_num), cv2_rgb_top)
    #     elif whether == 'state':
    #         cv2.imwrite('../test_experiments/test_{}/seg/top_img_state/kinect_top_{}_{}_img.jpg'.format(self.test_num,self.test_num,self.count), cv2_rgb_top)

    #     # print("output.shape :", output.shape) # ('output.shape :', (1, 8, 720, 1280))
    #     # print("output[:3].shape :", output[:3].shape) # ('output[:3].shape :', (1, 8, 720, 1280))
    #     # print("torch.max(output[:3], 1)[1].detach().cpu().numpy().shape :", torch.max(output[:3], 1)[1].detach().cpu().numpy().shape) # ('torch.max(output[:3], 1)[1].detach().cpu().numpy().shape :', (1, 720, 1280)
        
    #     labeled_mask = torch.max(output[:3], 1)[1].detach().cpu().numpy() # (1, 720, 1280)
        
    #     # zero_mask = np.zeros((labeled_mask.shape[0], labeled_mask.shape[1]))

    #     # B_y, B_x                     = np.where(labeled_mask[0] == 0)     # black         # BG
    #     laptop_y, laptop_x           = np.where(labeled_mask[0] == 1)     # white         # laptop
    #     drawer_y, drawer_x           = np.where(labeled_mask[0] == 2)     # yellow        # drawer
    #     bin_y, bin_x                 = np.where(labeled_mask[0] == 3)     # blue          # bin
    #     bin_cap_y, bin_cap_x         = np.where(labeled_mask[0] == 4)     # blue          # bin_cap
    #     tumbler_y, tumbler_x         = np.where(labeled_mask[0] == 5)     # red           # tumbler 
    #     tumbler_cap_y, tumbler_cap_x = np.where(labeled_mask[0] == 6)     # red           # tumbler_cap
    #     hand_cream_y, hand_cream_x   = np.where(labeled_mask[0] == 7)     # green         # hand_cream


    #     # self.box_list = []
    #     # self.mask_list = []
    #     # self.class_list = []


    #     # top-view로 볼 때 laptop 존재
    #     if len(laptop_y) > 2 and len(laptop_x) > 2:

    #         box_y1_laptop, box_x1_laptop, box_y2_laptop, box_x2_laptop = np.min(laptop_y), np.min(laptop_x), np.max(laptop_y), np.max(laptop_x)
    #         box_laptop = box_y1_laptop, box_x1_laptop, box_y2_laptop, box_x2_laptop
            
    #         laptop_mask = np.zeros((labeled_mask.shape[1], labeled_mask.shape[2]))
    #         # print("laptop_mask[labeled_mask[0]==1].shape :", laptop_mask[labeled_mask[0]==1].shape)
    #         laptop_mask[labeled_mask[0]==1] = 1
            
    #         if whether == 'goal':
    #             self.box_list_goal.append(box_laptop)
    #             self.mask_list_goal.append(laptop_mask)
    #             self.class_list_goal.append(1)
    #         elif whether == 'state':
    #             self.box_list_state.append(box_laptop)
    #             self.mask_list_state.append(laptop_mask)
    #             self.class_list_state.append(1)


    #     # top-view로 볼 때 drawer 존재
    #     if len(drawer_y) > 2 and len(drawer_x) > 2:
    #         box_y1_drawer, box_x1_drawer, box_y2_drawer, box_x2_drawer = np.min(drawer_y), np.min(drawer_x), np.max(drawer_y), np.max(drawer_x)
    #         box_drawer = box_y1_drawer, box_x1_drawer, box_y2_drawer, box_x2_drawer
            
    #         drawer_mask = np.zeros((labeled_mask.shape[1], labeled_mask.shape[2]))
    #         drawer_mask[labeled_mask[0]==2] = 1

    #         if whether == 'goal':
    #             self.box_list_goal.append(box_drawer)
    #             self.mask_list_goal.append(drawer_mask)
    #             self.class_list_goal.append(2)
    #         elif whether == 'state':
    #             self.box_list_state.append(box_drawer)
    #             self.mask_list_state.append(drawer_mask)
    #             self.class_list_state.append(2)


    #     # top-view로 볼 때 bin 존재
    #     if len(bin_y) > 2 and len(bin_x) > 2:
    #         box_y1_bin, box_x1_bin, box_y2_bin, box_x2_bin = np.min(bin_y), np.min(bin_x), np.max(bin_y), np.max(bin_x)
    #         box_bin = box_y1_bin, box_x1_bin, box_y2_bin, box_x2_bin
            
    #         bin_mask = np.zeros((labeled_mask.shape[1], labeled_mask.shape[2]))
    #         bin_mask[labeled_mask[0]==3] = 1

    #         if whether == 'goal':
    #             self.box_list_goal.append(box_bin)
    #             self.mask_list_goal.append(bin_mask)
    #             self.class_list_goal.append(3)
    #         elif whether == 'state':
    #             self.box_list_state.append(box_bin)
    #             self.mask_list_state.append(bin_mask)
    #             self.class_list_state.append(3)


    #     # top-view로 볼 때 bin_cap 존재
    #     if len(bin_cap_y) > 2 and len(bin_cap_x) > 2:
    #         box_y1_bin_cap, box_x1_bin_cap, box_y2_bin_cap, box_x2_bin_cap = np.min(bin_cap_y), np.min(bin_cap_x), np.max(bin_cap_y), np.max(bin_cap_x)
    #         box_bin_cap = box_y1_bin_cap, box_x1_bin_cap, box_y2_bin_cap, box_x2_bin_cap
            
    #         bin_cap_mask = np.zeros((labeled_mask.shape[1], labeled_mask.shape[2]))
    #         bin_cap_mask[labeled_mask[0]==4] = 1

    #         if whether == 'goal':
    #             self.box_list_goal.append(box_bin_cap)
    #             self.mask_list_goal.append(bin_cap_mask)
    #             self.class_list_goal.append(4)
    #         elif whether == 'state':
    #             self.box_list_state.append(box_bin_cap)
    #             self.mask_list_state.append(bin_cap_mask)
    #             self.class_list_state.append(4)


    #     # top-view로 볼 때 tumbler 존재
    #     if len(tumbler_y) > 2 and len(tumbler_x) > 2:
    #         box_y1_tumbler, box_x1_tumbler, box_y2_tumbler, box_x2_tumbler = np.min(tumbler_y), np.min(tumbler_x), np.max(tumbler_y), np.max(tumbler_x)
    #         box_tumbler = box_y1_tumbler, box_x1_tumbler, box_y2_tumbler, box_x2_tumbler
            
    #         tumbler_mask = np.zeros((labeled_mask.shape[1], labeled_mask.shape[2]))
    #         tumbler_mask[labeled_mask[0]==5] = 1

    #         if whether == 'goal':
    #             self.box_list_goal.append(box_tumbler)
    #             self.mask_list_goal.append(tumbler_mask)
    #             self.class_list_goal.append(5)
    #         elif whether == 'state':
    #             self.box_list_state.append(box_tumbler)
    #             self.mask_list_state.append(tumbler_mask)
    #             self.class_list_state.append(5)


    #     # top-view로 볼 때 tumbler_cap 존재
    #     if len(tumbler_cap_y) > 2 and len(tumbler_cap_x) > 2:
    #         box_y1_tumbler_cap, box_x1_tumbler_cap, box_y2_tumbler_cap, box_x2_tumbler_cap = np.min(tumbler_cap_y), np.min(tumbler_cap_x), np.max(tumbler_cap_y), np.max(tumbler_cap_x)
    #         box_tumbler_cap = box_y1_tumbler_cap, box_x1_tumbler_cap, box_y2_tumbler_cap, box_x2_tumbler_cap
            
    #         tumbler_cap_mask = np.zeros((labeled_mask.shape[1], labeled_mask.shape[2]))
    #         tumbler_cap_mask[labeled_mask[0]==6] = 1

    #         if whether == 'goal':
    #             self.box_list_goal.append(box_tumbler_cap)
    #             self.mask_list_goal.append(tumbler_cap_mask)
    #             self.class_list_goal.append(6)
    #         elif whether == 'state':
    #             self.box_list_state.append(box_tumbler_cap)
    #             self.mask_list_state.append(tumbler_cap_mask)
    #             self.class_list_state.append(6)


    #     # top-view로 볼 때 hand_cream 존재
    #     if len(hand_cream_y) > 2 and len(hand_cream_x) > 2:
    #         box_y1_hand_cream, box_x1_hand_cream, box_y2_hand_cream, box_x2_hand_cream = np.min(hand_cream_y), np.min(hand_cream_x), np.max(hand_cream_y), np.max(hand_cream_x)
    #         box_hand_cream = box_y1_hand_cream, box_x1_hand_cream, box_y2_hand_cream, box_x2_hand_cream
            
    #         hand_cream_mask = np.zeros((labeled_mask.shape[1], labeled_mask.shape[2]))
    #         hand_cream_mask[labeled_mask[0]==7] = 1

    #         if whether == 'goal':
    #             self.box_list_goal.append(box_hand_cream)
    #             self.mask_list_goal.append(hand_cream_mask)
    #             self.class_list_goal.append(7)
    #         elif whether == 'state':
    #             self.box_list_state.append(box_hand_cream)
    #             self.mask_list_state.append(hand_cream_mask)
    #             self.class_list_state.append(7)

    #     if whether == 'goal':
    #         self.obj_deg_ang_class_top_goal, self.result_PCA_img_top_goal = self.PCA_mg(box_msg = self.box_list_goal, mask = self.mask_list_goal, rgb_img = self.cv2_rgb_image, Class= self.class_list_goal)

    #         cv2.imwrite('../test_experiments/test_{}/rgb/top_pca_goal/kinect_top_{}_img.jpg'.format(self.test_num,self.test_num), self.result_PCA_img_top_goal)
        
    #     elif whether == 'state':
    #         self.obj_deg_ang_class_top_state, self.result_PCA_img_top_state = self.PCA_mg(box_msg = self.box_list_state, mask = self.mask_list_state, rgb_img = self.cv2_rgb_image, Class= self.class_list_state)

    #         cv2.imwrite('../test_experiments/test_{}/rgb/top_pca_state/kinect_top_{}_{}_img.jpg'.format(self.test_num,self.test_num,self.count), self.result_PCA_img_top_state)
    ########################################################################################################################################################################################################
    ########################################################################################################################################################################################################
    ########################################################################################################################################################################################################





    # [0  ,   0,   0],        # black         # BG
    # [255, 255, 255],        # white         # laptop
    # [255, 255,   0],        # yellow        # drawer
    # [  0,   0, 255],        # blue          # bin
    # [  0,   0, 255],        # blue          # bin_cap
    # [255,   0,   0],        # red           # tumbler    
    # [255,   0,   0],        # red           # tumbler_cap
    # [  0, 255,   0]])       # green         # hand_cream
    def get_each_obj_mask_top_view(self, whether):

        self.box_list_state = []
        self.mask_list_state = []
        self.class_list_state  = []

        self.move_home_top_view()

        ##################################################################################################
        output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()   # mask_rcnn
        # output = self.get_top_view()          # deeplab
        ##################################################################################################

        cv2_rgb_top = output

        # cv2_rgb_top = decode_seg_map_sequence_ymg(torch.max(output[:3], 1)[1].detach().cpu().numpy(), 'known_7')
    
        if whether == 'goal':
            cv2.imwrite('../test_experiments/test_{}/seg/top_img_goal/kinect_top_{}_img.jpg'.format(self.test_num,self.test_num), cv2_rgb_top)
        elif whether == 'state':
            cv2.imwrite('../test_experiments/test_{}/seg/top_img_state/kinect_top_{}_{}_img.jpg'.format(self.test_num,self.test_num,self.count), cv2_rgb_top)

        n = boxes.shape[0]

        for i in range(n):

            class_id = class_ids[i]

            label = class_names[class_id]

            if label == 'laptop':
                if whether == 'goal':
                    self.box_list_goal.append(boxes[i])
                    self.mask_list_goal.append(masks[:, :, i])
                    self.class_list_goal.append(1)
                elif whether == 'state':
                    self.box_list_state.append(boxes[i])
                    self.mask_list_state.append(masks[:, :, i])
                    self.class_list_state.append(1)

            elif label == 'drawer':
                if whether == 'goal':
                    self.box_list_goal.append(boxes[i])
                    self.mask_list_goal.append(masks[:, :, i])
                    self.class_list_goal.append(2)
                elif whether == 'state':
                    self.box_list_state.append(boxes[i])
                    self.mask_list_state.append(masks[:, :, i])
                    self.class_list_state.append(2)

            elif label == 'bin':
                if whether == 'goal':
                    self.box_list_goal.append(boxes[i])
                    self.mask_list_goal.append(masks[:, :, i])
                    self.class_list_goal.append(3)
                elif whether == 'state':
                    self.box_list_state.append(boxes[i])
                    self.mask_list_state.append(masks[:, :, i])
                    self.class_list_state.append(3)

            elif label == 'bin_cap':
                if whether == 'goal':
                    self.box_list_goal.append(boxes[i])
                    self.mask_list_goal.append(masks[:, :, i])
                    self.class_list_goal.append(4)
                elif whether == 'state':
                    self.box_list_state.append(boxes[i])
                    self.mask_list_state.append(masks[:, :, i])
                    self.class_list_state.append(4)

            elif label == 'tumbler':
                if whether == 'goal':
                    self.box_list_goal.append(boxes[i])
                    self.mask_list_goal.append(masks[:, :, i])
                    self.class_list_goal.append(5)
                elif whether == 'state':
                    self.box_list_state.append(boxes[i])
                    self.mask_list_state.append(masks[:, :, i])
                    self.class_list_state.append(5)

            elif label == 'tumbler_cap':
                if whether == 'goal':
                    self.box_list_goal.append(boxes[i])
                    self.mask_list_goal.append(masks[:, :, i])
                    self.class_list_goal.append(6)
                elif whether == 'state':
                    self.box_list_state.append(boxes[i])
                    self.mask_list_state.append(masks[:, :, i])
                    self.class_list_state.append(6)

            elif label == 'hand_cream':
                if whether == 'goal':
                    self.box_list_goal.append(boxes[i])
                    self.mask_list_goal.append(masks[:, :, i])
                    self.class_list_goal.append(7)
                elif whether == 'state':
                    self.box_list_state.append(boxes[i])
                    self.mask_list_state.append(masks[:, :, i])
                    self.class_list_state.append(7)


        if whether == 'goal':
            self.obj_deg_ang_class_top_goal, self.result_PCA_img_top_goal = self.PCA_mg(box_msg = self.box_list_goal, mask = self.mask_list_goal, rgb_img = self.cv2_rgb_image, Class= self.class_list_goal)

            cv2.imwrite('../test_experiments/test_{}/rgb/top_pca_goal/kinect_top_{}_img.jpg'.format(self.test_num,self.test_num), self.result_PCA_img_top_goal)
        
        elif whether == 'state':
            self.obj_deg_ang_class_top_state, self.result_PCA_img_top_state = self.PCA_mg(box_msg = self.box_list_state, mask = self.mask_list_state, rgb_img = self.cv2_rgb_image, Class= self.class_list_state)

            cv2.imwrite('../test_experiments/test_{}/rgb/top_pca_state/kinect_top_{}_{}_img.jpg'.format(self.test_num,self.test_num,self.count), self.result_PCA_img_top_state)



    ############################################################################################################################################################################################################################
    ############################################################################################################################################################################################################################
    ############################################################################################################################################################################################################################
    # [0  ,   0,   0],        # black         # BG
    # [255, 255, 255],        # white         # laptop
    # [255, 255,   0],        # yellow        # drawer
    # [  0,   0, 255],        # blue          # bin
    # [  0,   0, 255],        # blue          # bin_cap
    # [255,   0,   0],        # red           # tumbler    
    # [255,   0,   0],        # red           # tumbler_cap
    # [  0, 255,   0]])       # green         # hand_cream
    # 서랍 -> 상자뚜껑 -> 텀블러 뚜겅 / 순으로 x,y 값을 calibration 함
    # 서랍 -> 상자뚜껑 -> 텀블러 뚜겅 / 순으로 x,y 값을 calibration 함
    # 서랍 -> 상자뚜껑 -> 텀블러 뚜겅 / 순으로 x,y 값을 calibration 함
    def get_each_obj_mask_top_view_goal_calibrate(self):

        self.box_list_goal_cal = []
        self.class_list_goal_cal = []
        self.obj_deg_ang_class_top_goal_calibrate = []

        self.move_home_top_view_goal_calibrate() # z = 0.53 -> z = 0.41 으로 일단 내려옴 (아래로만 내려 옴)

        ##################################################################################################
        # -> 서랍 / 2
        for i in range(len(self.obj_position_class_goal)):
            if self.obj_position_class_goal[i][3] == 2:    # drawer의 클래스 idx : 2
                x_obj = self.obj_position_class_goal[i][0]
                y_obj = self.obj_position_class_goal[i][1]

                self.move_to_obj(x=x_obj, y=y_obj)

                output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

                cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_drawer/kinect_top_rgb_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
                cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_drawer/kinect_top_seg_{}_img.jpg'.format(self.test_num, self.test_num), output)

                n = boxes.shape[0]

                for i in range(n):

                    class_id = class_ids[i]

                    label = class_names[class_id]

                    if label == 'drawer':

                        self.box_list_goal_cal.append(boxes[i])
                        self.class_list_goal_cal.append(2)  # drawer의 클래스 idx : 2

                        self.drawer_exact_ang = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

                        self.obj_deg_ang_class_top_goal_calibrate.append(self.drawer_exact_ang)

                        break

                self.move_to_obj(x=-x_obj, y=-y_obj)

                print("label : {}".format(label))

                print("서랍 x,y cal 체크 중")

                break
        ##################################################################################################


        ##################################################################################################
        # -> 상자 뚜껑 / 4
        for i in range(len(self.obj_position_class_goal)):
            if self.obj_position_class_goal[i][3] == 4:    # bin_cap의 클래스 idx : 4
                x_obj = self.obj_position_class_goal[i][0]
                y_obj = self.obj_position_class_goal[i][1]

                self.move_to_obj(x=x_obj, y=y_obj)

                output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

                cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_bin_cap/kinect_top_rgb_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
                cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_bin_cap/kinect_top_seg_{}_img.jpg'.format(self.test_num, self.test_num), output)

                n = boxes.shape[0]

                for i in range(n):

                    class_id = class_ids[i]

                    label = class_names[class_id]

                    if label == 'bin_cap':

                        self.box_list_goal_cal.append(boxes[i])
                        self.class_list_goal_cal.append(4)  # bin_cap의 클래스 idx : 4

                        self.bin_cap_exact_ang = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

                        self.obj_deg_ang_class_top_goal_calibrate.append(self.bin_cap_exact_ang)

                        break

                self.move_to_obj(x=-x_obj, y=-y_obj)

                print("label : {}".format(label))

                print("상자 뚜껑 x,y cal 체크 중")

                break
        ##################################################################################################



        ##################################################################################################
        # -> 텀블러 뚜껑 / 6
        for i in range(len(self.obj_position_class_goal)):

            if self.obj_position_class_goal[i][3] == 6:    # tumbler_cap의 클래스 idx : 6
                x_obj = self.obj_position_class_goal[i][0]
                y_obj = self.obj_position_class_goal[i][1]

                self.move_to_obj(x=x_obj, y=y_obj)

                output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

                cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_rgb_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
                cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_seg_{}_img.jpg'.format(self.test_num, self.test_num), output)

                n = boxes.shape[0]

                for i in range(n):

                    class_id = class_ids[i]

                    label = class_names[class_id]

                    if label == 'tumbler_cap':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함

                        self.box_list_goal_cal.append(boxes[i])
                        self.class_list_goal_cal.append(6) # tumbler_cap의 클래스 idx : 6    

                        self.tumbler_cap_exact_ang = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

                        self.obj_deg_ang_class_top_goal_calibrate.append(self.tumbler_cap_exact_ang)

                        break

                # 2번 한게 이상하게 더 부정확함 ㅅㅂ;
                self.move_to_obj(x=-x_obj, y=-y_obj) # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함
                                                     # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함
                                                     # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함

                print("label : {}".format(label))

                print("텀블러 뚜껑 x,y 첫 번째 cal 체크 중")

                break
        ##################################################################################################



        # ##################################################################################################
        # ##################################################################################################
        # # -> 텀블러 본체 / 5
        # # -> 텀블러 뚜껑 / 6
        # for i in range(len(self.obj_position_class_goal)):
        #     print(" self.obj_position_class_goal[:][3] : ", self.obj_position_class_goal[:][3])
        #     if self.obj_position_class_goal[i][3] == 6:    # tumbler_cap의 클래스 idx : 6
        #         x_obj = self.obj_position_class_goal[i][0]
        #         y_obj = self.obj_position_class_goal[i][1]

        #         self.move_to_obj(x=x_obj, y=y_obj)

        #         output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

        #         cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_rgb_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
        #         cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_seg_{}_img.jpg'.format(self.test_num, self.test_num), output)

        #         n = boxes.shape[0]

        #         for i in range(n):

        #             class_id = class_ids[i]

        #             label = class_names[class_id]

        #             if label == 'tumbler_cap':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함
        #             # if label == 'tumbler_cap' or label == 'tumbler':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함

        #                 self.box_list_goal_cal.append(boxes[i])
        #                 self.class_list_goal_cal.append(6) # tumbler_cap의 클래스 idx : 6    

        #                 self.tumbler_cap_exact_ang = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

        #                 self.obj_deg_ang_class_top_goal_calibrate.append(self.tumbler_cap_exact_ang)

        #                 break

        #         # 2번 한게 이상하게 더 부정확함 ㅅㅂ;
        #         self.move_to_obj(x=-x_obj, y=-y_obj) # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함
        #                                              # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함
        #                                              # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함

        #         print("label : {}".format(label))

        #         print("텀블러 뚜껑 x,y 첫 번째 cal 체크 중")

        #         break

        #     elif (self.obj_position_class_goal[i][3] == 5) and not (6 in self.obj_position_class_goal[:][3]):    # tumbler의 클래스 idx : 5
        #         x_obj = self.obj_position_class_goal[i][0]
        #         y_obj = self.obj_position_class_goal[i][1]

        #         self.move_to_obj(x=x_obj, y=y_obj)

        #         output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

        #         cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_rgb_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
        #         cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_seg_{}_img.jpg'.format(self.test_num, self.test_num), output)

        #         n = boxes.shape[0]

        #         for i in range(n):

        #             class_id = class_ids[i]

        #             label = class_names[class_id]

        #             if label == 'tumbler':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함
        #             # if label == 'tumbler_cap' or label == 'tumbler':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함

        #                 self.box_list_goal_cal.append(boxes[i])
        #                 self.class_list_goal_cal.append(5) # tumbler_cap의 클래스 idx : 6    

        #                 self.tumbler_cap_exact_ang = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

        #                 self.obj_deg_ang_class_top_goal_calibrate.append(self.tumbler_cap_exact_ang)

        #                 break

        #         # 2번 한게 이상하게 더 부정확함 ㅅㅂ;
        #         self.move_to_obj(x=-x_obj, y=-y_obj) # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함
        #                                              # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함
        #                                              # >>> 텀블러는 2번 calibration해야 해서 함수 순서 관계상 쭉 이어가기 위해 일단 주석 처리 함

        #         print("label : {}".format(label))

        #         print("텀블러 본체 x,y 첫 번째 cal 체크 중")

        #         break
        # ##################################################################################################
        # ##################################################################################################



    def get_each_obj_mask_top_view_goal_calibrate_state(self, what_obj):
        ##################################################################################################

        # 이미 얼추 tumbler_cap / bin_cap_bottom 근처 위에 올라와 있음 >>> z = 0.23
        # 이미 얼추 tumbler_cap / bin_cap_bottom 근처 위에 올라와 있음 >>> z = 0.23
        # 이미 얼추 tumbler_cap / bin_cap_bottom 근처 위에 올라와 있음 >>> z = 0.23

        # self.move_up_down_obj(z=0.1) # 잠시 위로 올라가서 제대로 보려는 행동 / z = 0.23 -> z = 0.33

        #############################################################################################################################################################
        if what_obj == "bin_cap_bottom":

            output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

            cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_bin_cap/kinect_top_rgb_step_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
            cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_bin_cap/kinect_top_seg_step_{}_img.jpg'.format(self.test_num, self.test_num), output)

            n = boxes.shape[0]

            for i in range(n):

                class_id = class_ids[i]

                label = class_names[class_id]

                if label == 'bin_cap':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함  

                    self.bin_cap_exact_ang_step = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

                    self.depth_interpolation()

                    y1, x1, y2, x2 = boxes[i]
                    x_pix = int((x1 + x2)/2)
                    y_pix = int((y1 + y2)/2)

                    obj_pix = (x_pix, y_pix)
                    z_cam = self.depth[y_pix, x_pix]
                    # z_cam = self.cv2_depth_image[y_pix, x_pix]

                    position_x_delta, position_y_delta = self.pixel_to_tool(self.camera_parameters, obj_pix, z_cam)
                    
                    position_z = z_cam - 0.12 
                    # position_z = z_cam - (0.12 + 0.025) # 이게 정확함 / 이거 설마 키넥트 rgb센서 렌즈가 안으로 움푹 패여서 그런가;;

                    break

            print("label : {}".format(label))

            self.position_x_delta, self.position_y_delta = self.x_y_bias_correction(position_x_delta, position_y_delta) # 실제로 마지막으로는 로봇을 물체 위로 움직여야(제어) 하기 때문에 >>> bias 적용

            print("상자 뚜껑 step delta / delta_x : {}, delta _y : {}".format(self.position_x_delta, self.position_y_delta))

            self.move_to_obj(x=self.position_x_delta, y=self.position_y_delta) # step-보정 / TCP가 물체 중심으로 오도록 함

            # self.move_up_down_obj(z=-0.1)  # 다시 원 위치로 내려 감

            print("상자 뚜껑 step 상황 x,y cal 보정 중")
        #############################################################################################################################################################


        #############################################################################################################################################################
        elif what_obj == "tumbler_cap":

            output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

            cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_rgb_step_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
            cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_seg_step_{}_img.jpg'.format(self.test_num, self.test_num), output)

            n = boxes.shape[0]

            for i in range(n):

                class_id = class_ids[i]

                label = class_names[class_id]

                # if label == 'tumbler_cap' or label == 'tumbler':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함  
                if label == 'tumbler_cap':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함  

                    self.tumbler_cap_exact_ang_step = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

                    self.depth_interpolation()

                    y1, x1, y2, x2 = boxes[i]
                    x_pix = int((x1 + x2)/2)
                    y_pix = int((y1 + y2)/2)

                    obj_pix = (x_pix, y_pix)
                    z_cam = self.depth[y_pix, x_pix]
                    # z_cam = self.cv2_depth_image[y_pix, x_pix]

                    position_x_delta, position_y_delta = self.pixel_to_tool(self.camera_parameters, obj_pix, z_cam)
                    
                    position_z = z_cam - 0.12
                    # position_z = z_cam - (0.12 + 0.025) # 이게 정확함 / 이거 설마 키넥트 rgb센서 렌즈가 안으로 움푹 패여서 그런가;;

                    break

            print("label : {}".format(label))

            self.position_x_delta, self.position_y_delta = self.x_y_bias_correction(position_x_delta, position_y_delta)  # 실제로 마지막으로는 로봇을 물체 위로 움직여야(제어) 하기 때문에 >>> bias 적용

            print("텀블러 뚜껑 step delta / delta_x : {}, delta _y : {}".format(self.position_x_delta, self.position_y_delta))

            self.move_to_obj(x=self.position_x_delta, y=self.position_y_delta) # step-보정 / TCP가 물체 중심으로 오도록 함

            # self.move_up_down_obj(z=-0.1)  # 다시 원 위치로 내려 감

            print("텀블러 뚜껑 step 상황 x,y cal 보정 중")
        #############################################################################################################################################################


        #############################################################################################################################################################
        elif what_obj == "tumbler":

            output, boxes, masks, class_ids, class_names = self.get_top_view_maskrcnn()

            cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_rgb_step_{}_img.jpg'.format(self.test_num, self.test_num), self.cv2_rgb_image)
            cv2.imwrite('../test_experiments/test_{}/seg/xy_cal_tumbler_cap/kinect_top_seg_step_{}_img.jpg'.format(self.test_num, self.test_num), output)

            n = boxes.shape[0]

            for i in range(n):

                class_id = class_ids[i]

                label = class_names[class_id]

                # if label == 'tumbler_cap' or label == 'tumbler':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함  
                if label == 'tumbler':  # 인식이 근접해서 찍었는데도 불구하고 병신같이 됨... / tumbler랑 tumbler_cap의 구분을 명확히 못함  

                    self.tumbler_cap_exact_ang_step = self.PCA_goal_calibrate_mg(box_msg = boxes[i], mask = masks[:, :, i])

                    self.depth_interpolation()

                    y1, x1, y2, x2 = boxes[i]
                    x_pix = int((x1 + x2)/2)
                    y_pix = int((y1 + y2)/2)

                    obj_pix = (x_pix, y_pix)
                    z_cam = self.depth[y_pix, x_pix]
                    # z_cam = self.cv2_depth_image[y_pix, x_pix]

                    position_x_delta, position_y_delta = self.pixel_to_tool(self.camera_parameters, obj_pix, z_cam)
                    
                    position_z = z_cam - 0.12
                    # position_z = z_cam - (0.12 + 0.025) # 이게 정확함 / 이거 설마 키넥트 rgb센서 렌즈가 안으로 움푹 패여서 그런가;;

                    break

            print("label : {}".format(label))

            self.position_x_delta, self.position_y_delta = self.x_y_bias_correction(position_x_delta, position_y_delta)  # 실제로 마지막으로는 로봇을 물체 위로 움직여야(제어) 하기 때문에 >>> bias 적용

            print("텀블러 본체 step delta / delta_x : {}, delta _y : {}".format(self.position_x_delta, self.position_y_delta))

            self.move_to_obj(x=self.position_x_delta, y=self.position_y_delta) # step-보정 / TCP가 물체 중심으로 오도록 함

            # self.move_up_down_obj(z=-0.1)  # 다시 원 위치로 내려 감

            print("텀블러 본체 step 상황 x,y cal 보정 중")
        #############################################################################################################################################################


    def PCA_goal_calibrate_mg(self, box_msg, mask):
   
        box_y1, box_x1, box_y2, box_x2 = box_msg
        # RoI_mask = np.copy(mask[box_y1:box_y2, box_x1:box_x2, i])
        # RoI_mask = np.copy(mask[i, box_y1:box_y2, box_x1:box_x2])
        RoI_mask = np.copy(mask[box_y1:box_y2, box_x1:box_x2])

        mean_x = int((box_x1 + box_x2)/2)
        mean_y = int((box_y1 + box_y2)/2)

        y, x = np.nonzero(RoI_mask)

        x = x - mean_x
        y = y - mean_y
        coords = np.vstack([x,y])
        cov_mat = np.cov(coords)

        evals, evecs = np.linalg.eig(cov_mat)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        tan_axis = y_v2 / x_v2
        rad_ang = atan(tan_axis)
        deg_ang = deg(rad_ang)

        return deg_ang



    def PCA_mg(self, box_msg, mask, rgb_img, Class):
        obj_deg_ang_class = []
        # n = box_msg.shape[0]
        n = len(box_msg)
        result_img = np.copy(rgb_img)
        for i in range(n):
            box_y1, box_x1, box_y2, box_x2 = box_msg[i]
            # RoI_mask = np.copy(mask[box_y1:box_y2, box_x1:box_x2, i])
            # RoI_mask = np.copy(mask[i, box_y1:box_y2, box_x1:box_x2])
            RoI_mask = np.copy(mask[i][box_y1:box_y2, box_x1:box_x2])

            mean_x = int((box_x1 + box_x2)/2)
            mean_y = int((box_y1 + box_y2)/2)

            y, x = np.nonzero(RoI_mask)

            x = x - mean_x
            y = y - mean_y
            coords = np.vstack([x,y])
            cov_mat = np.cov(coords)

            evals, evecs = np.linalg.eig(cov_mat)
            sort_indices = np.argsort(evals)[::-1]
            x_v1, y_v1 = evecs[:, sort_indices[0]]
            x_v2, y_v2 = evecs[:, sort_indices[1]]
            tan_axis = y_v2 / x_v2
            rad_ang = atan(tan_axis)
            deg_ang = deg(rad_ang)


            # if VisualResult == True:
            long_ang = atan(y_v2 / x_v2)
            short_ang = atan(y_v1 / x_v1)
            self.long_ang = long_ang
            self.short_ang = short_ang


            short_xmin = mean_x - 100 * abs(sin(short_ang))
            short_xmax = mean_x + 100 * abs(sin(short_ang))
            short_ymin = mean_y - 100 * abs(cos(short_ang))
            short_ymax = mean_y + 100 * abs(cos(short_ang))

            long_xmin = mean_x - 200 * abs(sin(long_ang))
            long_xmax = mean_x + 200 * abs(sin(long_ang))
            long_ymin = mean_y - 200 * abs(cos(long_ang))
            long_ymax = mean_y + 200 * abs(cos(long_ang))

            # result_img = np.copy(rgb_img)
            # result_img = cv2.cvtColor(result_img,cv2.COLOR_GRAY2RGB)
            # print((int(short_xmin),int(short_ymin)), (int(short_xmax),int(short_ymax)))
            if long_ang > 0:
                result_img = cv2.line(result_img, (int(short_xmin),int(short_ymin)), (int(short_xmax),int(short_ymax)), (255,0,0), 2)
                result_img = cv2.line(result_img, (int(long_xmin),int(long_ymax)), (int(long_xmax),int(long_ymin)), (0,0,255), 2)
            else:
                result_img = cv2.line(result_img, (int(short_xmin),int(short_ymax)), (int(short_xmax),int(short_ymin)), (255,0,0), 2)
                result_img = cv2.line(result_img, (int(long_xmin),int(long_ymin)), (int(long_xmax),int(long_ymax)), (0,0,255), 2)
			
            obj_deg_ang_class.append((deg_ang, Class))
	

        return obj_deg_ang_class, result_img


    #################################################################################################################
    #################################################################################################################
    #################################################################################################################
    def BC_output_delta(self, whether):

        print(1111)
        self.move_home_top_view_step()
        print(2222)

        # self.x, self.z = [], []
        if whether == "state":
            for i in range(len(self.obj_position_class_state)):
                if self.obj_position_class_state[i][3] == 4: # bin의 클래스 idx : 3
                                                            # bin_cap의 클래스 idx : 4
                    x_obj = self.obj_position_class_state[i][0]
                    y_obj = self.obj_position_class_state[i][1]
                    z_obj = self.obj_position_class_state[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_state[i][0]
                    z_obj = 0.195

        elif whether == "goal":
            for i in range(len(self.obj_position_class_goal)):
                if self.obj_position_class_goal[i][3] == 4: # bin의 클래스 idx : 3
                                                            # bin_cap의 클래스 idx : 4
                    x_obj = self.obj_position_class_goal[i][0]
                    y_obj = self.obj_position_class_goal[i][1]
                    z_obj = self.obj_position_class_goal[i][2]
                    grasp_ang = self.obj_deg_ang_class_top_goal[i][0]
                    z_obj = 0.09

        if grasp_ang > 90 :
            ang = -(180-grasp_ang)
            # self.hold_grasp_angle(grasp_angle_deg=ang)
        else:
            ang = grasp_ang
            # self.hold_grasp_angle(grasp_angle_deg=ang)


        # print("ang :", ang)

        # print("deg(ang) :", deg(ang))
        # 높이 차이가 0.3 >>> 0.22999999999999932
        # 높이 차이가 0.3 >>> 0.22999999999999932
        # 높이 차이가 0.3 >>> 0.22999999999999932
        # print("x_obj : {}, y_obj : {}, (z_obj-0.3) : {}, grasp_ang : {} ".format(x_obj, y_obj, (z_obj-0.3), grasp_ang))  # from ee ~ obj // base 좌표 기준
        print("x_obj : {}, y_obj : {}, z_obj : {}, ang : {} ".format(x_obj, y_obj, z_obj, ang))  # from ee ~ obj // base 좌표 기준

        r = np.linalg.norm([x_obj, y_obj])

        ######################################
        if y_obj > 0:
            theta = acos(x_obj/r+1e-10)
        else:
            theta = -acos(x_obj/r+1e-10)
        ######################################

        ######################################
        # if x_obj*y_obj >= 0:
        # theta = atan(y_obj/x_obj+1e-10)
        # else:
        #     theta = -atan(y_obj/x_obj+1e-10)
        ######################################

        # print("r : {}, deg(theta) : {} ".format(r, deg(theta)))

        self.obj_r = r               # for test -> 나중에 어느정도 유동적인 상자뚜껑 위치로 바꿔야 함  / 일단 지금은 fix  
        # self.obj_z = -(z_obj-0.3)    # for test -> 나중에 어느정도 유동적인 상자뚜껑 위치로 바꿔야 함  / 일단 지금은 fix 
        self.obj_z = -z_obj    # for test -> 나중에 어느정도 유동적인 상자뚜껑 위치로 바꿔야 함  / 일단 지금은 fix 
                                          
        # print("self.obj_z :", -(z_obj-0.3))

        # 로봇이 움직이기 시작하는 지점에서 딱 한번만 획득  
        ##################################################################################################################################################################
        # self.init_r_ee_fix = self.panda_end_pose[0] # 고정, 로봇 말단 위치 x (원점 같은 역할)
        # self.init_z_ee_fix = self.panda_end_pose[2] # 고정, 로봇 말단 위치 z (원점 같은 역할)

        # out_pose_ee_fix = list(self.panda_end_pose)  # 고정, 로봇 말단 위치 6d-pose   //  from base ~ ee
        # print("out_pose_ee_fix : ", out_pose_ee_fix) # out_pose_ee_fix :  [0.5200015294445408, 0.020006394372710762, 0.22999805573182497, -179.99964750309863, 0.00027130878582658035, 0.00028100720738078816]
        ##################################################################################################################################################################

        self.init_r = 0. # init, 현재 로봇 말단 위치 x  //  from ee ~ obj 방향  //  ee 를 원점 취급 하기 떄문에 이렇게 함  
        self.init_z = 0. # init, 현재 로봇 말단 위치 z  //  from ee ~ obj 방향  //  ee 를 원점 취급 하기 떄문에 이렇게 함  

        self.state_r, self.state_z = self.obj_r-self.init_r, self.obj_z-self.init_z
        
        sleep(0.03)

        out_delta_pose = [0.5200015294445408, 0.020006394372710762, 0.22999805573182497, -180, 0., 0]

        while True:
        # for i in range(122): 

            bc_in = [[self.state_r] + [self.state_z]]
            bc_in_ = torch.Tensor(bc_in)
            out_delta = self.bc_agent(bc_in_)
            out_delta = (out_delta.detach().numpy())/100.
            # print("out_delta : ", out_delta) # [[ 0.00252049 -0.00121385]]

            self.init_r += out_delta[0,0]  # from ee ~ obj 방향 
            self.init_z += out_delta[0,1]  # from ee ~ obj 방향 

            self.state_r, self.state_z = self.obj_r-self.init_r, self.obj_z-self.init_z

            out_delta_pose[0] += out_delta[0,0]*cos(theta)
            out_delta_pose[1] += out_delta[0,0]*sin(theta)
            out_delta_pose[2] += out_delta[0,1] 

            # print("out_delta_pose: ", out_delta_pose)

            print("self.state_r : {}, self.state_z : {} ".format(self.state_r, self.state_z))


            # 로봇 말단 기준으로 움직임 -> "relative = True"
            # 움직임의 방향은 판다로봇의 base 좌표계의 방향으로 (카메라 좌표계 방향으로 움직이는것이 아님)
            ###########################################################################
            # pose = [x, y, z, -180., 0., 0.]  # from base ~ obj 방향  // base 기준 절대 좌표
            pose = out_delta_pose # from base ~ obj 방향  // base 기준 절대 좌표

            maxVelocity = 0.50   # 0.310
            acceleration = 0.20
            relative = False
            operatingMode = 1

            rospy.wait_for_service("/panda_target_pose")
            try:
                SetTargetpose = rospy.ServiceProxy("/panda_target_pose", SetTargetPose)
                _ = SetTargetpose(pose, maxVelocity, acceleration, relative, operatingMode)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
            ###########################################################################
            sleep(0.01)

            ######################################################################################################################################################
            # if np.linalg.norm([self.state_r, self.state_z]) < 0.005:
            if whether == "state":
                if abs(self.state_r) < 0.004 and abs(self.state_z) < 0.004:
                    print("r : {}, x : {}, y : {}, z : {}".format(self.init_r, out_delta_pose[0]-0.5200015294445408, out_delta_pose[1]-0.020006394372710762, out_delta_pose[2]-0.22999805573182497))
                    sleep(1)
                    break

            elif whether == "goal":
                if abs(self.state_r) < 0.002:
                    print("r : {}, x : {}, y : {}, z : {}".format(self.init_r, out_delta_pose[0]-0.5200015294445408, out_delta_pose[1]-0.020006394372710762, out_delta_pose[2]-0.22999805573182497))
                    sleep(1)
                    break
            ######################################################################################################################################################
            
        ###############################################################
        if whether == "state":
            sleep(1)
            self.hold_grasp_angle(grasp_angle_deg=ang) # 가서 그리퍼 각도 돌리고
            self.gripper_close()                       # 잡고
            sleep(1) 
            self.move_up_down_obj(z=-0.15)                  # 다시 위로 어느정도 올라오고
            self.move_home_top_view_step()             # 행동 하기 바로 전 초기 상태로 롤백

        elif whether == "goal":
            sleep(1)
            self.hold_grasp_angle(grasp_angle_deg=ang) # 가서 그리퍼 각도 돌리고
            self.gripper_open() 
            sleep(1) 
            self.move_up_down_obj(z=-0.1)                   # 다시 위로 어느정도 올라오고
            self.move_home_top_view_step()             # 행동 하기 바로 전 초기 상태로 롤백
        ###############################################################
    #################################################################################################################
    #################################################################################################################
    #################################################################################################################



    def run_test(self):

	    ######################################
        self.test_num = 1000       # need to set
        ######################################

        self.count = 0  # 0 --> reset_goal / reset_state
		
        if self.flag and self.flag_depth:

            self.box_list_goal = []
            self.mask_list_goal = []
            self.class_list_goal = []

            # 앞에서 찍고-goal
            #######################################################################################################################################
            output_goal = self.forward_deeplab('goal')
            cv2_rgb_goal = decode_seg_map_sequence_ymg(torch.max(output_goal[:3], 1)[1].detach().cpu().numpy(), 'known_7')
            cv2_rgb_goal = cv2_rgb_goal[47:647, 63:1183]
            cv2.imwrite('../test_experiments/test_{}/seg/goal_img/kinect_goal_{}_img.jpg'.format(self.test_num,self.test_num), cv2_rgb_goal)

            # self.goal = cv2_rgb_goal[47:647, 63:1183]
            self.goal = cv2_rgb_goal
            self.goal = cv2.resize(self.goal, dsize=(140, 75), interpolation=cv2.INTER_CUBIC)
            self.goal = transform(self.goal)
            print('get_goal_img')
            #######################################################################################################################################

            # 위에서 찍고-goal
            ##############################################
            self.get_each_obj_mask_top_view(whether='goal')
            self.get_pose_class_goal()

            # 특정 물체의 정확한 -> x,y 좌표를 찾기 위한 -> calibration 
            self.get_each_obj_mask_top_view_goal_calibrate()
            self.get_pose_class_goal_calibrate()

            # # 특정 물체의 정확한 -> x,y 좌표를 찾기 위한 -> calibration / 한번 더 보정 / tumbler가 좆같이 위치가 안맞아 >>> 2번 calibration한게 이상하게 위치가 더 안맞음 뭐지... ㅅㅂ
            # self.get_each_obj_mask_top_view_goal_calibrate_second()
            # self.get_pose_class_goal_calibrate_second()
            ##############################################


            while not rospy.is_shutdown():

                print('bbb')
                with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                    listener.join()
                print('ccc')

                # STEP - state
                # STEP - state
                # STEP - state
                # if self.count >= 0:
                    
                # 앞에서 찍고-state
                #######################################################################################################################################
                output_state = self.forward_deeplab('state')
                cv2_rgb_state = decode_seg_map_sequence_ymg(torch.max(output_state[:3], 1)[1].detach().cpu().numpy(), 'known_7')
                cv2_rgb_state = cv2_rgb_state[47:647, 63:1183]
                cv2.imwrite('../test_experiments/test_{}/seg/state_img/kinect_state_{}_{}_img.jpg'.format(self.test_num,self.test_num,self.count), cv2_rgb_state)

                # self.state = cv2_rgb_state[47:647, 63:1183]
                self.state = cv2_rgb_state
                self.state = cv2.resize(self.state, dsize=(140, 75), interpolation=cv2.INTER_CUBIC)
                self.state = transform(self.state)
                print('get_state_{}_img'.format(self.count))
                #######################################################################################################################################


                # 위에서 찍고-state
                ##############################################
                self.get_each_obj_mask_top_view(whether='state')
                self.get_pose_class_state()
                ##############################################
    
                self.s_ = torch.cat((self.state, self.goal), 0)
                self.s_ = np.array(self.s_)
                self.s_ = np.reshape(self.s_, reshape_)

                a, out = self.q.sample_action(torch.from_numpy(self.s_).float())

                ####################################################################################
                done_prob = self.q.organize_done_or_not(torch.from_numpy(self.s_).float())
                print("done_prob :", done_prob)
                print("="*60)
                ####################################################################################


                # STEP - RL 정리행동명령에 기반한 --> 로봇 움직임
                # # #########################################
                self.step(action=a)
                # # #########################################

                self.count += 1



if __name__ == '__main__':
    # main()
    test = kinect_rgb_img_get()
    test.run_test()
