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

from Virtual_tidy_up_env import sequence_env
from tensorboardX import SummaryWriter

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
my_path = os.path.abspath(os.path.dirname(__file__))

EVAL = False


SAVE_PATH = my_path + "/weight"

SUMMARY_PATH = my_path + "/summary"


learning_rate = 0.0005 
gamma         = 0.7
buffer_limit  = 100000 
size = [buffer_limit]
batch_size    = 128
interval = 200
max_step = 30 
start_training_memory_size = 10000
reshape_ = (1,6,75,140) 
obs_dim = [6,75,140]   
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


#Seed
seed = 666
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
env = sequence_env(seed)


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


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim=obs_dim, size=size, batch_size = batch_size , n_step = n_step, gamma = gamma):
        self.obs_buf = np.zeros(size + obs_dim, dtype=np.float32)       # shape -> (N, 6, 84, 84) 
        self.next_obs_buf = np.zeros(size + obs_dim, dtype=np.float32)  # shape -> (N, 6, 84, 84) 
        self.acts_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
        self.rews_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 
        self.done_buf = np.zeros(size, dtype=np.float32)                # shape -> (N,) 

        
        self.max_size, self.batch_size = size[0], batch_size            # N, 128(mini-batch-size)
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = collections.deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, obs, act, rew, next_obs, done):
        transition = (obs, act, rew, next_obs, done)

        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step: 
                                                 
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)   # 여긴 n-step  (r, s' d) => (r_n, s_n, d_n)
        obs, act = self.n_step_buffer[0][:2] 
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        
        self.ptr = (self.ptr + 1) % self.max_size        
        self.size = min(self.size + 1, self.max_size)   
        
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
    

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step rew, next_obs, and done."""
        # info of the last transition                            
        rew, next_obs, done = n_step_buffer[-1][-3:]


        for transition in reversed(list(n_step_buffer)[:-1]):    
            r, n_o, d = transition[-3:]
     
            rew = r + gamma * rew * (1 - d)                      

            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done
    
    
    def __len__(self):
        return self.size


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

        self.sum_tree = SumSegmentTree(tree_capacity)  
        self.min_tree = MinSegmentTree(tree_capacity)  
                                                       

    def store(self, obs, act, rew, next_obs, done):
        """Store experience and priority."""

        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:                                                      
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha  
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha  
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size             
        
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

                                  
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        
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
        """Sample indices based on proportions."""  
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
        p_min = self.min_tree.min() / self.sum_tree.sum() 
        max_weight = (p_min * len(self)) ** (-beta)       
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight   



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
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(device)  


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
       
       
        # mode: train / test
        self.evaluation = EVAL
        if not self.evaluation:
            self.summary = SummaryWriter(SUMMARY_PATH)






    def sample_action(self, state):
        """Select an action from the input state."""

        q1 = self.Q1(state.to(device))
        q2 = self.Q2(state.to(device))

        out = torch.min(q1, q2)
        
        return out.argmax().item(), out


    def _compute_dqn_loss(self, q, q_target, samples, gamma, n_epi):
        
        """Return categorical dqn loss."""

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        

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
            next_dist_2 = next_dist_2[range(self.batch_size), next_action_2]  
                                                                             

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


        return elementwise_loss_1, elementwise_loss_2



    def train(self, q, q_target, memory, optimizer_1, optimizer_2, beta, gamma, n_epi): 

        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = memory.sample_batch(beta)
        
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(device)
        indices = samples["indices"]
        
        
        # 1-step Learning loss
        elementwise_loss_1, elementwise_loss_2 = self._compute_dqn_loss(q, q_target, samples, gamma, n_epi)
        
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

        return loss_1.item(), loss_2.item()
    
    

def main(env):
    
    q = Qnet().to(device)
    # dummy_input = (torch.rand(reshape_, device=device),)
    # q.summary.add_graph(q, dummy_input, True)
 

    if q.evaluation:
        print(111111111111)
        q.load_state_dict(torch.load('...'))
    
    
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())

    memory = PrioritizedReplayBuffer()

    optimizer_1 = optim.Adam(q.Q1.parameters(), lr=learning_rate)
    optimizer_2 = optim.Adam(q.Q2.parameters(), lr=learning_rate)
    
    ##########################################################################################
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=LR_gamma)
    ##########################################################################################
    
    
    epi_score = 0 
    loss_1 = 0
    loss_2 = 0

    epi_loss_1 = 0
    epi_loss_2 = 0

    beta = 0.
    n_epi = 0
    global_step = 0
    
    accuracy = collections.deque(maxlen=100)
    
    while True:
        
        ########################################################
        s = env.reset(n_epi)
        ########################################################
        
        s = np.array(s)
        s = np.reshape(s, reshape_)
        done = False

        for cur_step in range(max_step):


            a, out = q.sample_action(torch.from_numpy(s).float())

            ####################################################################################
            s_prime, r, done = env.step(a, cur_step+1, global_step, n_epi)
            ####################################################################################
            
            s_prime = np.array(s_prime)
            s_prime = np.reshape(s_prime, reshape_)

            ####################################################################################
            memory.store(s, a, r, s_prime, done)
            ####################################################################################
            
            beta = beta_by_frame(global_step)
            
            s = s_prime

            if (len(memory)>start_training_memory_size) and (not q.evaluation):
                loss_1, loss_2 = q.train(q, q_target, memory, optimizer_1, optimizer_2, beta, gamma, n_epi)


            epi_score += r
            epi_loss_1 += loss_1
            epi_loss_2 += loss_2

            cur_step += 1
            global_step += 1

            if done:
                break
        
            
        accuracy.append(r)
        
        
        if (n_epi%interval == 0) and (not q.evaluation):
            q_target.load_state_dict(q.state_dict())
            torch.save(q.state_dict(), SAVE_PATH + "/" + "sequence_RAINBOW_DQN_%.f.pt"%n_epi)

        
        if (not q.evaluation):
            q.summary.add_scalar('average_loss_1_every_Epi', epi_loss_1/((cur_step) + 1e-4), n_epi)
            q.summary.add_scalar('average_loss_2_every_Epi', epi_loss_2/((cur_step) + 1e-4), n_epi)
            q.summary.add_scalar('score_every_Epi', epi_score, n_epi)
            q.summary.add_scalar('taken_steps_every_Epi', cur_step, n_epi)
            q.summary.add_scalar('accuracy_every_Epi', accuracy.count(1.)/(len(accuracy)+ 1e-4), n_epi)

            print('@'*30)
            print('@@@@@@@@@ Writing data in Tensorboard @@@@@@@@@ ')
            print('n_epi : {}'.format(n_epi))
            print('epi_score : {}'.format(epi_score))
            print('accuracy : {}/{}'.format(accuracy.count(1.),(len(accuracy))))
            print('@'*30)
            
            epi_score = 0
            epi_loss_1 = 0
            epi_loss_2 = 0
        
        n_epi += 1

if __name__ == '__main__':
    main(env)