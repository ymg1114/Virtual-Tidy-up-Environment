# -*- coding: utf-8 -*-
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import gym
import collections
import random
import numpy as npasd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from segment_tree import MinSegmentTree, SumSegmentTree
# from sequence_Env.sequence_colorfree_allrand_mg_sparse_reward import ymg_sequence_env
# from sequence_Env.sequence_colorfree_mg import ymg_sequence_env
from sequence_Env.test_4_4_aen_ddn_back_diff_obj_scale_sparse_0_ver2_1 import ymg_sequence_env
from tensorboardX import SummaryWriter
# from AEN.Action_Elimination_Network_for_test4 import AEN_net
# from RDN.Reset_distinguish_Network_for_test4_2 import RDN_net
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle


SAVE_PATH = "../weight/weights_test_0_ver2_1_sub"
# SAVE_PATH = "../weight/weights_RAINBOW_test_44_AEN_DDN_diff_obj_scale_lunar"

SUMMARY_PATH = "../summary/summary_test_0_ver2_1_sub"
# SUMMARY_PATH = "../summary/summary_RAINBOW_test_44_AEN_DDN_diff_obj_scale_lunar"

DDN_SAVE_PATH = "../DDN/WEIGHT/weight_DDN_test_0_ver2_1_sub"

AEN_SAVE_PATH = "../AEN/WEIGHT/weight_AEN_test_0_ver2_1_sub"

# PLOT_SAVE_PATH = "../PLOT/plot_test_0_ver2_"


ENV = 'Mine' # 'LunarLander-v2' -> 'Mine'
EVAL = False


# LR_step_size = 100
# LR_gamma = 0.9998
#Hyperparameters
if ENV == 'LunarLander-v2':
    # learning_rate = 0.001 # 0.0005 -> 0.001(for LR-scheduler)
    learning_rate = 0.0005 
    gamma         = 0.98 # 0.98 -> 0.7
    buffer_limit  = 50000 # 100000 -> 200000 -> 150000
    size = [buffer_limit]
    batch_size    = 128 # 32 -> 128
    interval = 50 # 20 -> 200
    max_step = 600 # 600 -> 50
    start_training_memory_size = 2000 # 2000 -> 5000
    reshape_ = (1,8) # (1,8) -> (1,6,100,100) 
    obs_dim = [8]   # [6,100,100] -> [2,84,84]
    act_dim = 4
    clip_value = 1.
    max_norm = 40.
    
    alpha = 0.6
    beta_start = 0.4
    beta_frames = 100000 # 100000 -> 150000 -> 200000
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    num_atoms = 51
    Vmin = -200 # -200 -> -5
    Vmax = 200 # 200 -> 5

    n_step = 1

else:
    # learning_rate = 0.001 # 0.0005 -> 0.001(for LR-scheduler)
    learning_rate = 0.0005 
    gamma         = 0.7
    buffer_limit  = 100000  # 100000 -> 200000 -> 150000 -> 100000 -> 200000 -> 150000 -> 200000 -> 100000
    size = [buffer_limit]
    batch_size    = 128 # 32 -> 128 
    interval = 200 # 20 -> 200
    max_step = 30 # 600 -> 50 -> 30 -> 40 -> 50 -> 30
    start_training_memory_size = 10000 # 2000 -> 10000 -> 20000 / 10000
    reshape_ = (1,6,75,140) 
    obs_dim = [6,75,140]   
    act_dim = 9 # 10 -> 9
    clip_value = 1.
    max_norm = 40.

    alpha = 0.6
    beta_start = 0.4
    beta_frames = 200000 # 100000 -> 300000 -> 200000 -> 400000 -> 200000
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    num_atoms = 51
    Vmin = -2
    Vmax = 2 
    
    n_step = 1

    AEN_dim = [act_dim]
    
    AEN_epoch = 1
    
    DDN_epoch = 1

    sub_buffer_limit = 50000  # 50000 
    sub_batch_size = batch_size
    # sub_start_training_memory_size = 5000 # 10000 -> 5000



#Seed
seed = 666
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if ENV == 'LunarLander-v2':
    env = gym.make('LunarLander-v2')
    env.seed(seed)
else:
    env = ymg_sequence_env(seed)



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









# # For LunarLanderv2
# # For LunarLanderv2
# # For LunarLanderv2
# ################################################################################################################################################
# ################################################################################################################################################
# ################################################################################################################################################
# ################################################################################################################################################
# ################################################################################################################################################
# class ReplayBuffer:
#     """A simple numpy replay buffer."""

#     def __init__(self, obs_dim=obs_dim, size=size, batch_size = batch_size , n_step = n_step, gamma = gamma):
#         self.obs_buf = np.zeros(size + obs_dim, dtype=np.float32)       # shape -> (N, 6, 84, 84) 
#         self.next_obs_buf = np.zeros(size + obs_dim, dtype=np.float32)  # shape -> (N, 6, 84, 84) 
#         self.acts_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
#         self.rews_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
#         self.done_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
#         self.max_size, self.batch_size = size[0], batch_size            # N, 128(mini-batch-size)
#         self.ptr, self.size, = 0, 0
        
#         # for N-step Learning
#         self.n_step_buffer = collections.deque(maxlen=n_step)
#         self.n_step = n_step
#         self.gamma = gamma

#     def store(self, obs, act, rew, next_obs, done):
#         transition = (obs, act, rew, next_obs, done)
#         self.n_step_buffer.append(transition)

#         # single step transition is not ready
#         if len(self.n_step_buffer) < self.n_step: # 지정한 n-step(예를들어 3개)의 크기보다 "deque"로 선언된 n-step 버퍼에 들어있는 transition의 개수가 적으면 
#                                                   # 아예 "()" 를 반환
#                                                   # 밑에 코드 까지 가지도 않음
#             return ()
        
#         # make a n-step transition
#         rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)   # 여긴 n-step  (r, s' d) => (r_n, s_n, d_n)
#         obs, act = self.n_step_buffer[0][:2]                                          # 여긴 1-step  (s, a)
        
#         self.obs_buf[self.ptr] = obs
#         self.next_obs_buf[self.ptr] = next_obs
#         self.acts_buf[self.ptr] = act
#         self.rews_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size        # 버퍼내의 데이터 idx
#         self.size = min(self.size + 1, self.max_size)    # 버퍼의 실시간 크기 (transition-데이터 가 몇개나 들어가 있는지)
        
#         return self.n_step_buffer[0]

    
#     def sample_batch_from_idxs(self, idxs):
#         # for N-step Learning
#         return dict(
#             obs=self.obs_buf[idxs],
#             next_obs=self.next_obs_buf[idxs],
#             acts=self.acts_buf[idxs],
#             rews=self.rews_buf[idxs],
#             done=self.done_buf[idxs],
#         )
    
    
#     # (예시) n-step : 3
#     # s, a, r, s', d
#     #          s', a', r', s'', d'
#     #                      s'', a'', r'', s''', d''
#     def _get_n_step_info(self, n_step_buffer, gamma):
#         """Return n step rew, next_obs, and done."""
#         # info of the last transition                            # n_step_buffer --> 에 들어갈수 있는 transition-데이터의 개수 --> "n-step의 수" 만큼
#         rew, next_obs, done = n_step_buffer[-1][-3:]             # [-1] >>> "n_step_buffer" 에 실시간으로 저장하고 있는 "맨 마지막" transition-데이터
#                                                                  # transition = (obs, act, rew, next_obs, done)
#                                                                  #               -5   -4   -3   -2        -1     

#         for transition in reversed(list(n_step_buffer)[:-1]):    # "n_step_buffer"의 "맨 마지막-1" 인 transition-데이터 부터 한개씩 빼옴
#             r, n_o, d = transition[-3:]

#             rew = r + gamma * rew * (1 - d)                      # n-step 리워드 (감마가 누적 적용됨) (거꾸로 가면서) (당연한 거지만 뒷 부분 transition 일 수록 감마가 더 많이 곱해짐)
#                                                                  # d : True 면 -> 밀림 (더 아래 step 으로) // 맨 마지막 "transition" 의 rew -> n-step 보상에 포함되지 않음
#                                                                  # d, d', d'' (n-step : 3) 싹다 True --> n-step 리워드 : 그냥 r (1-step 과 동일)


#                                                                  # 사실 d'' 가 True / False 여부는 관계 없음
#             next_obs, done = (n_o, d) if d else (next_obs, done) # d : True 면 -> 밀림 (더 아래 step 으로)
#                                                                  # d, d' (n-step : 3) 싹다 True --> s', d >>> s', d (1-step 과 동일)
                                                                 
#                                                                  # d' (n-step : 3) True  -->  s', d >>> s'', d' (2-step 과 동일)
#                                                                  # d : False
        
#                                                                  # d, d' (n-step : 3) 싹다 False --> s', d >>> s''', d'' (3-step 과 동일)
        
#                                                                  # n-step 보상도 이와 동일하게 움직임
        

#                                                                  # 결국 n-step-transition은 (n-step : 3 을 예시로 함)
#                                                                  # d, d', d''  <-- 얘들이 T/F 긴지 아닌지에 따라 바뀜
#                                                                  # d -> T 이면 // d' 이 T/F 뭐든 상관없이 -> 그냥 1-step-transition 과 동일
#                                                                  # d'' <-- 은 원래 상관 없었음
        
#                                                                  # 따라서, 구현 할 때는 "n-step 버퍼" 이긴 해도 
#                                                                  # d, d', d'' --> 들의 F/T 유무에 따라서 각각의 버퍼내에 들어 있는 "transition" 들의 형태가 항상 n-step은 아닐 수도 있음
#                                                                  # 1-step 일 수도 / 2-step 일 수도 / 3-step 일 수도  있음
#         return rew, next_obs, done

#     def __len__(self):
#         return self.size




# # 자세히 보는거는 시간낭비 일거같고, -> 의미만 파악해 보자
# # 자세히 보는거는 시간낭비 일거같고, -> 의미만 파악해 보자
# # 자세히 보는거는 시간낭비 일거같고, -> 의미만 파악해 보자
# class PrioritizedReplayBuffer(ReplayBuffer):
#     """Prioritized Replay buffer.
    
#     Attributes:
#         max_priority (float): max priority
#         tree_ptr (int): next index of tree
#         alpha (float): alpha parameter for prioritized replay buffer
#         sum_tree (SumSegmentTree): sum tree for prior
#         min_tree (MinSegmentTree): min tree for min prior to get max weight
        
#     """
    
#     def __init__(self, obs_dim=obs_dim, size=size, batch_size = batch_size, alpha = alpha, n_step= n_step, gamma = gamma):
#         """Initialization."""
#         assert alpha >= 0
        
#         super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size, n_step, gamma)
#         self.max_priority, self.tree_ptr = 1.0, 0
#         self.alpha = alpha
        
#         # capacity must be positive and a power of 2.
#         tree_capacity = 1
#         while tree_capacity < self.max_size:
#             tree_capacity *= 2

#         self.sum_tree = SumSegmentTree(tree_capacity)  # 씨그마_k [ (p_k)^alpha ] 를 구하기 위한 것
#         self.min_tree = MinSegmentTree(tree_capacity)  # (p_i)^alpha 를 구하기 위한 것
#                                                        # 얘네 둘다 기본적으로 경험데이터(인덱스) / 경험데아터(우선순위) 는 동기화된 상태로 공유하는듯
         
#     def store(self, obs, act, rew, next_obs, done):
#         """Store experience and priority."""
#         transition = super().store(obs, act, rew, next_obs, done)
        
#         if transition:                                                      # self.tree_ptr -> 현재 들어온 경험데이터의 PER 버퍼 내에서의 index
#             self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha  # 둘 다 저장하는 건 일단 -> (p_i)^alpha 를 저장하는듯
#             self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha  # 둘 다 저장하는 건 일단 -> (p_i)^alpha 를 저장하는듯
#             self.tree_ptr = (self.tree_ptr + 1) % self.max_size             # "update_priorities" -> 함수가 호출되지 않으면 가장 최근에 PER 버퍼로 들어와 저장된 경험데이터의 우선순위가 가장 높게 책정됨
        
#         return transition

#     def sample_batch(self, beta = 0.4):
#         """Sample a batch of experiences."""
#         assert len(self) >= self.batch_size
#         assert beta > 0
        
#         indices = self._sample_proportional()
        
#         obs = self.obs_buf[indices]
#         next_obs = self.next_obs_buf[indices]
#         acts = self.acts_buf[indices]
#         rews = self.rews_buf[indices]
#         done = self.done_buf[indices]
#         weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
#         return dict(
#             obs=obs,
#             next_obs=next_obs,
#             acts=acts,
#             rews=rews,
#             done=done,
#             weights=weights,
#             indices=indices,
#         )
        
#     def update_priorities(self, indices, priorities):
#         """Update priorities of sampled transitions."""
#         assert len(indices) == len(priorities)

#         for idx, priority in zip(indices, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self)

#             self.sum_tree[idx] = priority ** self.alpha
#             self.min_tree[idx] = priority ** self.alpha

#             self.max_priority = max(self.max_priority, priority)
            
#     def _sample_proportional(self):
#         """Sample indices based on proportions."""  # PER 버퍼의 TD-Error 를 기준으로 bias된 확률 분포를 따라 
#                                                     # PER 버퍼에서 경험데이터를 샘플 (원래 모분포 -> uniform 샘플링)
#                                                     # 모분포 -> TD-Error 를 기준으로 bias된 확률 분포로 바뀌었기 때문에 -> importance sampling 기법으로 보정 작업이 필요함
#         indices = []
#         p_total = self.sum_tree.sum(0, len(self) - 1)
#         segment = p_total / self.batch_size
        
#         for i in range(self.batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             upperbound = random.uniform(a, b)
#             idx = self.sum_tree.retrieve(upperbound)
#             indices.append(idx)
            
#         return indices
    
#     def _calculate_weight(self, idx, beta):
#         """Calculate the weight of the experience at idx."""
#         # get max weight 
#         p_min = self.min_tree.min() / self.sum_tree.sum() # to get max-weight --> need to get min "(p_i)^alpha"
#         max_weight = (p_min * len(self)) ** (-beta)       # len(self) >>> "N" // 계산 편의상 (-beta)를 쓰기 때문에 // 의미는 원래 정상적인 Replay Buffer의 확률분포인 uniform-sampling을 의미 함
        
#         # calculate weights
#         p_sample = self.sum_tree[idx] / self.sum_tree.sum()
#         weight = (p_sample * len(self)) ** (-beta)
#         weight = weight / max_weight
        
#         return weight   # 최종적인 importance sampling을 위한 가중치 보정값
# ################################################################################################################################################
# ################################################################################################################################################
# ################################################################################################################################################
# ################################################################################################################################################
# ################################################################################################################################################














class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim=obs_dim, size=size, batch_size = batch_size , n_step = n_step, gamma = gamma):
        self.obs_buf = np.zeros(size + obs_dim, dtype=np.float32)       # shape -> (N, 6, 84, 84) 
        self.next_obs_buf = np.zeros(size + obs_dim, dtype=np.float32)  # shape -> (N, 6, 84, 84) 
        self.acts_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
        self.rews_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
        self.done_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
        self.aen_buf = np.zeros(size + AEN_dim, dtype=np.float32)       # shape -> (N, 7)
        self.ddn_buf = np.zeros(size, dtype=np.float32)                 # shape -> (N,)
        
        self.max_size, self.batch_size = size[0], batch_size            # N, 128(mini-batch-size)
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = collections.deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    # def store(self, obs, act, AEN_label, rew, next_obs, done, DDN_label_prime):
    def store(self, obs, act, rew, next_obs, done):
        transition = (obs, act, rew, next_obs, done)
        # transition = (obs, act, AEN_label, rew, next_obs, done, DDN_label_prime)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step: # 지정한 n-step(예를들어 3개)의 크기보다 "deque"로 선언된 n-step 버퍼에 들어있는 transition의 개수가 적으면 
                                                  # 아예 "()" 를 반환
                                                  # 밑에 코드 까지 가지도 않음
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)   # 여긴 n-step  (r, s' d) => (r_n, s_n, d_n)
        obs, act = self.n_step_buffer[0][:2] 
        
        # rew, next_obs, done, DDN_label_prime = self._get_n_step_info(self.n_step_buffer, self.gamma)    # t+1 or t+N  
        # obs, act, AEN_label = self.n_step_buffer[0][:3]                                                 # t 즉, 여긴 1-step  (s, a)
         
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        # self.aen_buf[self.ptr] = AEN_label
        # self.ddn_buf[self.ptr] = DDN_label_prime
        
        self.ptr = (self.ptr + 1) % self.max_size        # 버퍼내의 데이터 idx
        self.size = min(self.size + 1, self.max_size)    # 버퍼의 실시간 크기 (transition-데이터 가 몇개나 들어가 있는지)
        
        return self.n_step_buffer[0]

    
    def sample_batch_from_idxs(self, idxs):
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
    
  
        # #  for N-step Learning
        # return dict(
        #     obs=self.obs_buf[idxs],
        #     next_obs=self.next_obs_buf[idxs],
        #     acts=self.acts_buf[idxs],
        #     rews=self.rews_buf[idxs],
        #     done=self.done_buf[idxs],
        #     AEN_labels=self.aen_buf[idxs],
        #     DDN_labels_prime=self.ddn_buf[idxs]
        # )
        
  
    
    # (예시) n-step : 3
    # s, a, r, s', d
    #          s', a', r', s'', d'
    #                      s'', a'', r'', s''', d''
    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step rew, next_obs, and done."""
        # info of the last transition                            # n_step_buffer --> 에 들어갈수 있는 transition-데이터의 개수 --> "n-step의 수" 만큼
        rew, next_obs, done = n_step_buffer[-1][-3:]
        # rew, next_obs, done, DDN_label_prime = n_step_buffer[-1][-4:]             # [-1] >>> "n_step_buffer" 에 실시간으로 저장하고 있는 "맨 마지막" transition-데이터
                                                                 # transition = (obs, act, rew, next_obs, done)
                                                                 #               -5   -4   -3   -2        -1     

        for transition in reversed(list(n_step_buffer)[:-1]):    # "n_step_buffer"의 "맨 마지막-1" 인 transition-데이터 부터 한개씩 빼옴
            r, n_o, d = transition[-3:]
            # r, n_o, d, ddn_l = transition[-4:]
            
            rew = r + gamma * rew * (1 - d)                      # n-step 리워드 (감마가 누적 적용됨) (거꾸로 가면서) (당연한 거지만 뒷 부분 transition 일 수록 감마가 더 많이 곱해짐)
                                                                 # d : True 면 -> 밀림 (더 아래 step 으로) // 맨 마지막 "transition" 의 rew -> n-step 보상에 포함되지 않음
                                                                 # d, d', d'' (n-step : 3) 싹다 True --> n-step 리워드 : 그냥 r (1-step 과 동일)


                                                                 # 사실 d'' 가 True / False 여부는 관계 없음
            # next_obs, done, DDN_label_prime = (n_o, d, ddn_l) if d else (next_obs, done, DDN_label_prime)
            next_obs, done = (n_o, d) if d else (next_obs, done) # d : True 면 -> 밀림 (더 아래 step 으로)
                                                                 # d, d' (n-step : 3) 싹다 True --> s', d >>> s', d (1-step 과 동일)
                                                                 
                                                                 # d' (n-step : 3) True  -->  s', d >>> s'', d' (2-step 과 동일)
                                                                 # d : False
        
                                                                 # d, d' (n-step : 3) 싹다 False --> s', d >>> s''', d'' (3-step 과 동일)
        
                                                                 # n-step 보상도 이와 동일하게 움직임
        

                                                                 # 결국 n-step-transition은 (n-step : 3 을 예시로 함)
                                                                 # d, d', d''  <-- 얘들이 T/F 긴지 아닌지에 따라 바뀜
                                                                 # d -> T 이면 // d' 이 T/F 뭐든 상관없이 -> 그냥 1-step-transition 과 동일
                                                                 # d'' <-- 은 원래 상관 없었음
        
                                                                 # 따라서, 구현 할 때는 "n-step 버퍼" 이긴 해도 
                                                                 # d, d', d'' --> 들의 F/T 유무에 따라서 각각의 버퍼내에 들어 있는 "transition" 들의 형태가 항상 n-step은 아닐 수도 있음
                                                                 # 1-step 일 수도 / 2-step 일 수도 / 3-step 일 수도  있음
        return rew, next_obs, done
        # return rew, next_obs, done, DDN_label_prime
    
    
    def __len__(self):
        return self.size




# 자세히 보는거는 시간낭비 일거같고, -> 의미만 파악해 보자
# 자세히 보는거는 시간낭비 일거같고, -> 의미만 파악해 보자
# 자세히 보는거는 시간낭비 일거같고, -> 의미만 파악해 보자
class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(self, obs_dim=obs_dim, size=size, batch_size = batch_size, alpha = alpha, n_step= n_step, gamma = gamma):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size, n_step, gamma)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)  # 씨그마_k [ (p_k)^alpha ] 를 구하기 위한 것
        self.min_tree = MinSegmentTree(tree_capacity)  # (p_i)^alpha 를 구하기 위한 것
                                                       # 얘네 둘다 기본적으로 경험데이터(인덱스) / 경험데아터(우선순위) 는 동기화된 상태로 공유하는듯
    
    # def store(self, obs, act, AEN_label, rew, next_obs, done, DDN_label_prime):
    def store(self, obs, act, rew, next_obs, done):
        """Store experience and priority."""
        # transition = super().store(obs, act, AEN_label, rew, next_obs, done, DDN_label_prime)
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:                                                      # self.tree_ptr -> 현재 들어온 경험데이터의 PER 버퍼 내에서의 index
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha  # 둘 다 저장하는 건 일단 -> (p_i)^alpha 를 저장하는듯
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha  # 둘 다 저장하는 건 일단 -> (p_i)^alpha 를 저장하는듯
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size             # "update_priorities" -> 함수가 호출되지 않으면 가장 최근에 PER 버퍼로 들어와 저장된 경험데이터의 우선순위가 가장 높게 책정됨
        
        return transition

    def sample_batch(self, beta = 0.4):
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        # aen_labels = self.aen_buf[indices]
        # ddn_labels_prime = self.ddn_buf[indices]
                                  
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        # return dict(
        #     obs=obs,
        #     next_obs=next_obs,
        #     acts=acts,
        #     rews=rews,
        #     done=done,
        #     weights=weights,
        #     indices=indices,
        #     aen_labels=aen_labels,
        #     ddn_labels_prime=ddn_labels_prime
        # )
        
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices
        )
        
        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self):
        """Sample indices based on proportions."""  # PER 버퍼의 TD-Error 를 기준으로 bias된 확률 분포를 따라 
                                                    # PER 버퍼에서 경험데이터를 샘플 (원래 모분포 -> uniform 샘플링)
                                                    # 모분포 -> TD-Error 를 기준으로 bias된 확률 분포로 바뀌었기 때문에 -> importance sampling 기법으로 보정 작업이 필요함
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight 
        p_min = self.min_tree.min() / self.sum_tree.sum() # to get max-weight --> need to get min "(p_i)^alpha"
        max_weight = (p_min * len(self)) ** (-beta)       # len(self) >>> "N" // 계산 편의상 (-beta)를 쓰기 때문에 // 의미는 원래 정상적인 Replay Buffer의 확률분포인 uniform-sampling을 의미 함
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight   # 최종적인 importance sampling을 위한 가중치 보정값



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
        #                                    nn.LeakyReLU(),
        #                                    nn.Linear(128, 128),
        #                                    nn.LeakyReLU())
        
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
                                               nn.Linear(512, 512),
                                               nn.LeakyReLU(),
                                               nn.Linear(512, self.out_dim))
        
        self.AEN_loss_fn = nn.MSELoss()
        self.AEN_optimizer = torch.optim.Adam(self.AEN_feature_layer.parameters(), lr=learning_rate)
        
        
        # # Sigmoid + BCE
        # self.AEN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
        #                                        nn.LeakyReLU(),
        #                                        nn.Linear(512, self.out_dim),
        #                                        nn.Sigmoid())
        
        # self.AEN_loss_fn = nn.BCELoss()
        # self.AEN_optimizer = torch.optim.Adam(self.AEN_feature_layer.parameters(), lr=learning_rate)
        ################################################################################
        
        
        ################################################################################
        # Sigmoid + BCE
        self.DDN_feature_layer = nn.Sequential(nn.Linear(self._feature_size(), 512), 
                                               nn.LeakyReLU(),
                                               nn.Linear(512, 1),
                                               nn.Sigmoid())

        self.DDN_loss_fn = nn.BCELoss()
        self.DDN_optimizer = torch.optim.Adam(self.DDN_feature_layer.parameters(), lr=learning_rate)
        ################################################################################


    def forward_AEN(self, x):
        feature = self.feature_layer(x)
        
        xx = feature.detach()
        
        ########################################
        out = self.AEN_feature_layer(xx)
        ########################################
        
        return out


    # mini_batch_sample -> "self.aen.train((state, aen_labels))"
    def train_AEN(self, mini_batch_sample):
        epoch_loss = 0 
        for _ in range(AEN_epoch):    

            # Forward pass
            state = mini_batch_sample[0]
            # print('state.shape', state.shape)   # state.shape torch.Size([N, 6, 45, 96])

            y_true = mini_batch_sample[1]
            # print('y_true.shape', y_true.shape) # y_true.shape torch.Size([N, 7])

            y_pred = self.forward_AEN(state)
            # print('y_pred.shape', y_pred.shape) # y_pred.shape torch.Size([N, 7])


            # Calculate mse-loss
            loss = self.AEN_loss_fn(y_pred, y_true)
            

            # Backward and optimize
            self.AEN_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.AEN_feature_layer.parameters(), max_norm)
            self.AEN_optimizer.step()
            
            # running_loss += loss.item()  # accumulating loss
            epoch_loss += loss.item()  # accumulating loss

        return epoch_loss




    def forward_DDN(self, x):
        feature = self.feature_layer(x)
        
        xx = feature.detach()
        
        ########################################
        out = self.DDN_feature_layer(xx)
        ########################################
        
        return out


    # mini_batch_sample -> "self.aen.train((next_state, ddn_labels))"
    def train_DDN(self, mini_batch_sample):
        epoch_loss = 0 
        for _ in range(DDN_epoch):    

            # Forward pass
            next_state = mini_batch_sample[0]
            # print('state.shape', state.shape)   # state.shape torch.Size([N, 6, 45, 96])

            y_true = mini_batch_sample[1]
            # print('y_true.shape', y_true.shape) # y_true.shape torch.Size([N, 7])

            y_pred = self.forward_DDN(next_state)
            # print('y_pred.shape', y_pred.shape) # y_pred.shape torch.Size([N, 7])


            # Calculate mse-loss
            loss = self.DDN_loss_fn(y_pred, y_true)
            

            # Backward and optimize
            self.DDN_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.DDN_feature_layer.parameters(), max_norm)
            self.DDN_optimizer.step()
            
            # running_loss += loss.item()  # accumulating loss
            epoch_loss += loss.item()  # accumulating loss

        return epoch_loss



    def save_AEN(self, n_epi):
        torch.save(self.AEN_feature_layer.state_dict(), AEN_SAVE_PATH + "/" + "sequence_RAINBOW_DQN_AEN_%.f.pt"%n_epi)


    def save_DDN(self, n_epi):
        torch.save(self.DDN_feature_layer.state_dict(), DDN_SAVE_PATH + "/" + "sequence_RAINBOW_DQN_DDN_%.f.pt"%n_epi)


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


    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_value_1.reset_noise()
        self.noisy_value_2.reset_noise()
        self.noisy_advantage_1.reset_noise()
        self.noisy_advantage_2.reset_noise()




class Qnet(nn.Module):
    def __init__(self, in_dim=obs_dim, out_dim=act_dim, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax, batch_size=batch_size, n_step=n_step, EVAL=EVAL):
        super(Qnet, self).__init__()
        
        self.prior_eps    = 1e-6
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size 
       
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        self.n_step = n_step 

        self.Q1 = RAINBOW_Q_net(in_dim=in_dim, out_dim=out_dim, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax).to(device)
        self.Q2 = RAINBOW_Q_net(in_dim=in_dim, out_dim=out_dim, num_atoms=num_atoms, Vmin=Vmin, Vmax=Vmax).to(device)
       
        # self.aen = AEN_net().to(device)
       
        # mode: train / test
        self.evaluation = EVAL
        if not self.evaluation:
            self.summary = SummaryWriter(SUMMARY_PATH)
        else:
            # print(222222222222)
            self.Q1.DDN_feature_layer.load_state_dict(torch.load('../DDN/WEIGHT/weight_DDN_test_0/sequence_RAINBOW_DQN_DDN_53200.pt'))
            # print(222222222222)
            self.Q1.AEN_feature_layer.load_state_dict(torch.load('../AEN/WEIGHT/weight_AEN_test_0/sequence_RAINBOW_DQN_AEN_53200.pt'))

    
    def organize_done_or_not(self, next_state):
    # def organize_done_or_not(self, state, next_state):
        
        done_prob = self.Q1.forward_DDN(next_state.to(device))

        aen_prrdict = self.Q1.forward_AEN(next_state.to(device))

        return done_prob, aen_prrdict



    # # For LunarLanderv2
    # # For LunarLanderv2
    # # For LunarLanderv2
    # def sample_action(self, state, global_step):
    #     """Select an action from the input state."""

    #     q1 = self.Q1(state.to(device))
    #     q2 = self.Q2(state.to(device))

    #     out = torch.min(q1, q2)

    #     return out.argmax().item(), out




    def sample_action(self, state):
        """Select an action from the input state."""

        q1 = self.Q1(state.to(device))
        q2 = self.Q2(state.to(device))

        out = torch.min(q1, q2)
        
        
        # # FOR TRAIN
        # ############################################################################################################
        # # aen_correct = self.aen(state.to(device))
        # if global_step > sub_start_training_memory_size:
        #     aen_correct = self.Q1.forward_AEN(state.to(device))
        #     out = out + aen_correct
        # # 딱 이자리에만 AEN 이 사용됨 / RAINBOW DQN의 학습에 -> AEN이 관여 하는 것이 아님 / 어찌보면 탐사를 도와준다고 볼 수 있음
        # ############################################################################################################
       
       
        # # FOR TEST
        # ############################################################################################################
        # aen_correct = self.Q1.forward_AEN(state.to(device))
        # out = out + aen_correct
        # ############################################################################################################
        # print("aen_correct :", aen_correct)
       
       
        # ############################################################################################################
        # aen_label = torch.tensor(aen_label, dtype=torch.float).to(device)
        # out = q + aen_label
        # # out = q * aen_label
        # ############################################################################################################
        
        return out.argmax().item(), out



    # def _compute_dqn_loss(self, q, q_target, samples, gamma, n_epi):
    def _compute_dqn_loss(self, q, q_target, samples, samples_aen, samples_ddn, gamma, n_epi):
        
        """Return categorical dqn loss."""

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        
        ################################################################################################
        # aen_labels = torch.FloatTensor(samples["aen_labels"]).to(device)
        # ddn_labels_prime = torch.FloatTensor(samples["ddn_labels_prime"]).to(device)
        
        state_aen, aen_labels = samples_aen
        next_state_ddn, ddn_labels_prime = samples_ddn

        # print("state_aen.shape :", state_aen.shape)      # state_aen.shape : torch.Size([N, 6, 75, 140])
        # print("aen_labels.shape :", aen_labels.shape)    # aen_labels.shape : torch.Size([N, 9])


        # print("next_state_ddn.shape :", next_state_ddn.shape)                                  # next_state_ddn.shape : torch.Size([N, 6, 75, 140])
        # print("ddn_labels_prime.unsqueeze(1).shape :", ddn_labels_prime.unsqueeze(1).shape)    # ddn_labels_prime.unsqueeze(1).shape : torch.Size([N, 1])

        # print("state.shape :", state.shape)              # state.shape : torch.Size([N, 6, 45, 96])
        # print("aen_labels.shape :", aen_labels.shape)    # aen_labels.shape : torch.Size([N, 9])

        # print("next_state.shape :", next_state.shape)              # next_state.shape : torch.Size([N, 6, 45, 96])
        # print("ddn_labels_prime.shape :", ddn_labels_prime.shape)  # ddn_labels_prime.shape : torch.Size([N])

        
        aen_loss = self.Q1.train_AEN((state_aen.to(device), aen_labels.to(device)))
        ddn_loss = self.Q1.train_DDN((next_state_ddn.to(device), ddn_labels_prime.unsqueeze(1).to(device)))

        if (n_epi%interval == 0):
            self.Q1.save_DDN(n_epi)
            self.Q1.save_AEN(n_epi)
        ################################################################################################
        
        
        
        # aen-mini_batch-accuracy
        ############################################################################
        success_aen = collections.deque(maxlen=batch_size)
        idx_aen = []

        aen_predict = self.Q1.forward_AEN(state_aen.to(device))

        for i in range(len(aen_labels)):
        
            for j in range(len(aen_labels[0])):

                if aen_labels[i][j] == 1. and aen_predict[i][j] > 0.8:
                    idx_aen.append(1.)
                    
                elif aen_labels[i][j] == 0. and aen_predict[i][j] < 0.2:
                    idx_aen.append(1.)
                    
            if len(idx_aen) == len(aen_labels[0]):
                success_aen.append(1.)
            else:
                success_aen.append(0.)
                
            idx_aen = []
        
        aen_mini_batch_accuracy = success_aen.count(1.)/(len(success_aen)+ 1e-4)
        ############################################################################


        # ddn-mini_batch-accuracy
        ############################################################################
        success_ddn = collections.deque(maxlen=batch_size)

        ddn_predict = self.Q1.forward_DDN(next_state_ddn.to(device))

        for i in range(len(ddn_labels_prime)):
        
            if ddn_labels_prime[i] == 1. and ddn_predict[i] > 0.8:
                success_ddn.append(1.)
                
            elif ddn_labels_prime[i] == 0. and ddn_predict[i] < 0.2:
                success_ddn.append(1.)
                    
            else:
                success_ddn.append(0.)
                
        
        ddn_mini_batch_accuracy = success_ddn.count(1.)/(len(success_ddn)+ 1e-4)
        ############################################################################
        
        
        
        # Categorical DQN algorithm
        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)

        with torch.no_grad():
            # Double DQN
            next_action_1 = q.Q1(next_state).argmax(1)
            next_dist_1 = q_target.Q1.dist(next_state)
            next_dist_1 = next_dist_1[range(self.batch_size), next_action_1]

            t_z_1 = reward + (1 - done) * gamma * q.Q1.support
            t_z_1 = t_z_1.clamp(min=self.Vmin, max=self.Vmax)
            b_1 = (t_z_1 - self.Vmin) / delta_z
            l_1 = b_1.floor().long()
            u_1 = b_1.ceil().long()

            offset_1 = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).to(device))

            proj_dist_1 = torch.zeros(next_dist_1.size(), device=device)
            proj_dist_1.view(-1).index_add_(0, (l_1 + offset_1).view(-1), (next_dist_1 * (u_1.float() - b_1)).view(-1))
            proj_dist_1.view(-1).index_add_(0, (u_1 + offset_1).view(-1), (next_dist_1 * (b_1 - l_1.float())).view(-1))


            # Double DQN
            next_action_2 = q.Q2(next_state).argmax(1)
            next_dist_2 = q_target.Q2.dist(next_state)
            next_dist_2 = next_dist_2[range(self.batch_size), next_action_2]  # next_dist_2 -> 여기가 분기점 1
                                                                              # torch.min(next_dist_1, next_dist_2 ) -> 이걸 해서 하나를 고를건지 말 건지

            t_z_2 = reward + (1 - done) * gamma * q.Q2.support
            t_z_2 = t_z_2.clamp(min=self.Vmin, max=self.Vmax)
            b_2 = (t_z_2 - self.Vmin) / delta_z
            l_2 = b_2.floor().long()
            u_2 = b_2.ceil().long()

            offset_2 = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).to(device))

            proj_dist_2 = torch.zeros(next_dist_2.size(), device=device)
            proj_dist_2.view(-1).index_add_(0, (l_2 + offset_2).view(-1), (next_dist_2 * (u_2.float() - b_2)).view(-1))
            proj_dist_2.view(-1).index_add_(0, (u_2 + offset_2).view(-1), (next_dist_2 * (b_2 - l_2.float())).view(-1))


        dist_1 = q.Q1.dist(state)
        log_p_1 = torch.log(dist_1[range(self.batch_size), action])
        elementwise_loss_1 = -(proj_dist_1 * log_p_1).sum(1)

        dist_2 = q.Q2.dist(state)
        log_p_2 = torch.log(dist_2[range(self.batch_size), action])
        elementwise_loss_2 = -(proj_dist_2 * log_p_2).sum(1)


        # return elementwise_loss_1, elementwise_loss_2
        return elementwise_loss_1, elementwise_loss_2, aen_loss, ddn_loss, aen_mini_batch_accuracy, ddn_mini_batch_accuracy



    def train(self, q, q_target, memory, memory_aen, memory_ddn, optimizer_1, optimizer_2, beta, gamma, n_epi): 
    # def train(self, q, q_target, memory, optimizer_1, optimizer_2, beta, gamma, n_epi): 

        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = memory.sample_batch(beta)
        
        samples_aen = memory_aen.sample()
        samples_ddn = memory_ddn.sample()
        
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(device)
        indices = samples["indices"]
        
        
        # 1-step Learning loss
        # elementwise_loss_1, elementwise_loss_2 = self._compute_dqn_loss(q, q_target, samples, gamma, n_epi)
        # elementwise_loss_1, elementwise_loss_2, aen_loss, ddn_loss, aen_mini_batch_accuracy, ddn_mini_batch_accuracy = self._compute_dqn_loss(q, q_target, samples, gamma, n_epi)
        
        elementwise_loss_1, elementwise_loss_2, aen_loss, ddn_loss, aen_mini_batch_accuracy, ddn_mini_batch_accuracy = self._compute_dqn_loss(q, q_target, samples, samples_aen, samples_ddn, gamma, n_epi)
        
        # PER: importance sampling before average
        loss_1 = torch.mean(elementwise_loss_1 * weights)
        loss_2 = torch.mean(elementwise_loss_2 * weights)
        
        
        
        #########################################################################################################################
        # # N-step Learning loss
        # # we are gonna combine 1-step loss and n-step loss so as to
        # # prevent high-variance. The original rainbow employs n-step loss only.
        # if self.use_n_step:
        #     gamma = gamma ** self.n_step
        #     samples = memory_n.sample_batch_from_idxs(indices)
        #     elementwise_loss_n_loss_1, elementwise_loss_n_loss_2 = self._compute_dqn_loss(q, q_target, samples, gamma)
        #     elementwise_loss_1 += elementwise_loss_n_loss_1
        #     elementwise_loss_2 += elementwise_loss_n_loss_2
            
        #     # PER: importance sampling before average
        #     loss_1 = torch.mean(elementwise_loss_1 * weights)
        #     loss_2 = torch.mean(elementwise_loss_2 * weights)
        #########################################################################################################################
     
     
        optimizer_1.zero_grad()
        loss_1.backward()
        # clip_grad_value_(q.parameters(), clip_value)
        clip_grad_norm_(q.Q1.parameters(), max_norm)
        optimizer_1.step()
        
        optimizer_2.zero_grad()
        loss_2.backward()
        # clip_grad_value_(q.parameters(), clip_value)
        clip_grad_norm_(q.Q2.parameters(), max_norm)
        optimizer_2.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss_1.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        q.Q1.reset_noise()
        q.Q2.reset_noise()
        q_target.Q1.reset_noise()
        q_target.Q2.reset_noise()

        ##################
        # scheduler.step()
        ##################
        
        # return loss_1.item(), loss_2.item()
        return loss_1.item(), loss_2.item(), aen_loss, ddn_loss, aen_mini_batch_accuracy, ddn_mini_batch_accuracy
    
    
class ReplayBuffer_AEN():
    def __init__(self):
        self.buffer = collections.deque(maxlen=sub_buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n=sub_batch_size):
        mini_batch = random.sample(self.buffer, n)

        s_lst, aen_label_lst, = [], []
        
        for transition in mini_batch:

            s, aen_label = transition
            
            s_lst.append(s)
            aen_label_lst.append(aen_label)
            
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(aen_label_lst)
    
    def size(self):
        return len(self.buffer)


class ReplayBuffer_DDN():
    def __init__(self):
        self.buffer = collections.deque(maxlen=sub_buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n=sub_batch_size):
        mini_batch = random.sample(self.buffer, n)

        s_prime_lst, ddn_label_lst, = [], []
        
        for transition in mini_batch:

            s_prime, ddn_label = transition
            
            s_prime_lst.append(s_prime)
            ddn_label_lst.append(ddn_label)
            
        return torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(ddn_label_lst)
    
    def size(self):
        return len(self.buffer)


def main(env):
    
    q = Qnet().to(device)
    # dummy_input = (torch.rand(reshape_, device=device),)
    # q.summary.add_graph(q, dummy_input, True)
 

    if q.evaluation:
        print(111111111111)
        q.load_state_dict(torch.load('../weight/weights_test_0/sequence_RAINBOW_DQN_53200.pt'))
    
    
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())

    memory = PrioritizedReplayBuffer()
    memory_aen = ReplayBuffer_AEN()
    memory_ddn = ReplayBuffer_DDN()

    optimizer_1 = optim.Adam(q.Q1.parameters(), lr=learning_rate)
    optimizer_2 = optim.Adam(q.Q2.parameters(), lr=learning_rate)
    
    ##########################################################################################
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=LR_gamma)
    ##########################################################################################
    
    
    epi_score = 0 
    loss_1 = 0
    loss_2 = 0
    aen_loss = 0
    ddn_loss = 0
    epi_loss_1 = 0
    epi_loss_2 = 0
    epi_aen_loss = 0
    epi_ddn_loss = 0
    beta = 0.
    n_epi = 0
    global_step = 0
    
    aen_mini_batch_accuracy = 0
    ddn_mini_batch_accuracy = 0
    
    accuracy = collections.deque(maxlen=100)
    
    # ######################################
    # epi_list = []
    # score_list = []
    # accuracy_list = []
    # aen_accuracy_list = []
    # ddn_accuracy_list = []
    # ######################################
    
    while True:
        
        ########################################################
        # s = env.reset()
        s, AEN_label, DDN_label = env.reset(n_epi)
        ########################################################

        
        s = np.array(s)
        s = np.reshape(s, reshape_)
        done = False
        
        # while not done:
        for cur_step in range(max_step):


            a, out = q.sample_action(torch.from_numpy(s).float())
            # print("out :", out)
            
            # ################################################
            # a = random.randint(0,8)
            # ################################################
            
            ####################################################################################
            # s_prime, r, done, _ = env.step(a)
            s_prime, r, done, AEN_label_prime, DDN_label_prime = env.step(a, cur_step+1, global_step, n_epi)
            ####################################################################################
            
            s_prime = np.array(s_prime)
            s_prime = np.reshape(s_prime, reshape_)


            ####################################################################################
            done_prob, aen_prrdict = q.organize_done_or_not(torch.from_numpy(s_prime).float())
            # done_prob, aen_prrdict = q.organize_done_or_not(torch.from_numpy(s).float(), torch.from_numpy(s_prime).float())
            print("aen_prrdict :", aen_prrdict)
            print("done_prob :", done_prob)
            print('#'*30)
            ####################################################################################

            # print('s.shape : ', s.shape) # s.shape :  (1, 6, 75, 140)
            # print('s_prime.shape : ', s_prime.shape) # s_prime.shape :  (1, 6, 75, 140)
            # print('AEN_label.shape : ', np.array(AEN_label).shape) # AEN_label.shape :  (9,)
            # print('DDN_label_prime.shape : ', np.array(DDN_label_prime).shape) # DDN_label_prime.shape :  ()

            ####################################################################################
            memory.store(s, a, r, s_prime, done)
            # memory.store(s, a, AEN_label, r, s_prime, done, DDN_label_prime)
            memory_aen.put((s[0], AEN_label))
            memory_ddn.put((s_prime[0], DDN_label_prime))
            ####################################################################################
            
            beta = beta_by_frame(global_step)
            
            s = s_prime
            AEN_label = AEN_label_prime
            DDN_label = DDN_label_prime
            

            
            if (len(memory)>start_training_memory_size) and (not q.evaluation):
                # loss_1, loss_2 = q.train(q, q_target, memory, optimizer_1, optimizer_2, beta, gamma, n_epi)
                # loss_1, loss_2, aen_loss, ddn_loss, aen_mini_batch_accuracy, ddn_mini_batch_accuracy = q.train(q, q_target, memory, optimizer_1, optimizer_2, beta, gamma, n_epi)
                loss_1, loss_2, aen_loss, ddn_loss, aen_mini_batch_accuracy, ddn_mini_batch_accuracy = q.train(q, q_target, memory, memory_aen, memory_ddn, optimizer_1, optimizer_2, beta, gamma, n_epi)

            ####################################################################################
            # # for name, param in q.parameters():
            # for name, param in q.named_parameters():
            #     q.summary.add_histogram(name, param.data.cpu().numpy(), global_step)
            #     # q.summary.add_histogram('grad of '+name, param.grad.data.cpu().numpy(), global_step)
            #     # q.summary.add_histogram('norm grad of '+name, l2_norm(param.grad.data.cpu().numpy()), global_step)
            ####################################################################################

            epi_score += r
            epi_loss_1 += loss_1
            epi_loss_2 += loss_2
            epi_aen_loss += aen_loss
            epi_ddn_loss += ddn_loss
            
            cur_step += 1
            global_step += 1

            if done:
                break
        
            
        accuracy.append(r)
        
        # ########################################################
        # epi_list.append(n_epi+1)
        # score_list.append(epi_score)
        # accuracy_list.append(accuracy.count(1.)/(len(accuracy)+ 1e-4))
        # aen_accuracy_list.append(aen_mini_batch_accuracy)
        # ddn_accuracy_list.append(ddn_mini_batch_accuracy)
        # ########################################################
        
        
        if (n_epi%interval == 0) and (not q.evaluation):
            q_target.load_state_dict(q.state_dict())
            torch.save(q.state_dict(), SAVE_PATH + "/" + "sequence_RAINBOW_DQN_%.f.pt"%n_epi)
        
            # ################################################################################################
            # with open(PLOT_SAVE_PATH + "/" + "epi_data_%.f.pickle"%(n_epi+1), 'wb') as ff:
            #     pickle.dump(epi_list, ff)

            # with open(PLOT_SAVE_PATH + "/" + "epi_score_data_%.f.pickle"%(n_epi+1), 'wb') as ff:
            #     pickle.dump(score_list, ff)
                
            # with open(PLOT_SAVE_PATH + "/" + "epi_accuracy_data_%.f.pickle"%(n_epi+1), 'wb') as ff:
            #     pickle.dump(accuracy_list, ff)
        
            # with open(PLOT_SAVE_PATH + "/" + "epi_aen_accuracy_data_%.f.pickle"%(n_epi+1), 'wb') as ff:
            #     pickle.dump(aen_accuracy_list, ff)
                
            # with open(PLOT_SAVE_PATH + "/" + "epi_ddn_accuracy_data_%.f.pickle"%(n_epi+1), 'wb') as ff:
            #     pickle.dump(ddn_accuracy_list, ff)
            # #################################################################################################
        
        if (not q.evaluation):
            q.summary.add_scalar('score_every_Epi', epi_score, n_epi)
            # q.summary.add_scalar('average_score_every_Epi', epi_score/((cur_step) + 1e-4), n_epi)
            # q.summary.add_scalar('average_loss_1_every_Epi', epi_loss_1/((cur_step) + 1e-4), n_epi)
            # q.summary.add_scalar('average_loss_2_every_Epi', epi_loss_2/((cur_step) + 1e-4), n_epi)
            q.summary.add_scalar('average_AEN_loss_every_Epi', epi_aen_loss/((cur_step) + 1e-4), n_epi)
            q.summary.add_scalar('average_DDN_loss_every_Epi', epi_ddn_loss/((cur_step) + 1e-4), n_epi)

            q.summary.add_scalar('taken_steps_every_Epi', cur_step, n_epi)
            
            q.summary.add_scalar('accuracy_every_Epi', accuracy.count(1.)/(len(accuracy)+ 1e-4), n_epi)
            
            q.summary.add_scalar('AEN_accuracy_every_Epi', aen_mini_batch_accuracy, n_epi)
            q.summary.add_scalar('DDN_accuracy_every_Epi', ddn_mini_batch_accuracy, n_epi)
            
            print('@'*30)
            print('@@@@@@@@@ Writing data in Tensorboard @@@@@@@@@ ')
            print('n_epi : {}'.format(n_epi))
            print('epi_score : {}'.format(epi_score))
            print('accuracy : {}/{}'.format(accuracy.count(1.),(len(accuracy))))
            print('AEN_accuracy : {}'.format(aen_mini_batch_accuracy))
            print('DDN_accuracy : {}'.format(ddn_mini_batch_accuracy))
            print('@'*30)
            
            epi_score = 0
            epi_loss_1 = 0
            epi_loss_2 = 0
            epi_aen_loss = 0
            epi_ddn_loss = 0

        
        n_epi += 1

if __name__ == '__main__':
    main(env)