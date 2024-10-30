# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:34:31 2024
Reinforcement learning for simultaneous optimization of fluid allocation to all
casualties.
@author: msubramaniyan
"""
#%% Imports
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from itertools import combinations
from math import comb
from sklearn.preprocessing import OneHotEncoder
#%% Classes
   
class ReplayBuffer(object):
    pass
class Environment():
    def __init__(self, n_subjects, n_fluid_units):
        self.n_subjects = n_subjects
        self.n_fluid_units = n_fluid_units
        # Load excel files containing state info
        fn = r'C:\Users\msubramaniyan\Documents\projects\fluid_allocation\data\20240810\feature_matrix_hr_sbp.csv'
        df = pd.read_csv(fn)
        df.rename(columns={'y_hr':'x_hr_70'},inplace=True)
        df.rename(columns={'y_sbp':'x_sbp_70'},inplace=True)
        # Select subjects randomly
        sel_sub_ids = np.random.choice(df.subject_id.unique(),n_subjects).astype(int)
        sel_df = []
        for sub_id in sel_sub_ids:
            sdf = df[df.subject_id==sub_id]
            # Select a random bleeding scenario and keep it for all 4 treatment
            # conditions
            rand_bleed_cond = np.random.choice(sdf.bleed_tour_cond.unique())
            sel_df.append(sdf[sdf.bleed_tour_cond==rand_bleed_cond])
        self.df = pd.concat(sel_df)
        
        # Binary bins to code the infusion windows
        self.n_states = 3
        # iwin - infusion window        
        # code 0 - end of 10 min
        # code 1 - end of 40 min
        # code 2 - end of 70 min (also terminal state)
        self.onehotencoder = OneHotEncoder().fit([[x] for x in 
                                            range(self.n_states)])
        # End of infusion window 0 - that is, the initial 10 min period end
        iwin0_end_code = self.onehotencoder.transform([[0]]).toarray()
        self.iwin_end_code = np.repeat(iwin0_end_code,n_subjects,axis=0)
        self.iwin_end_code_width = iwin0_end_code.shape[1]
        # Infusion window times
        self.t_iwin_1_start = 11 
        self.t_iwin_1_end = 40
        self.t_iwin_2_start = 41
        self.t_iwin_2_end = 70
        self.n_history_points = 10 # number of minutes of history to use as part of state
        assert np.remainder(self.n_history_points,2)==0, \
                                'n_history_points must be an even number'
        self.history_step_size_iwin1 = np.floor((self.t_iwin_1_end - 
                                self.t_iwin_1_start + 1)/(self.n_history_points/2))
        self.history_step_size_iwin2 = np.floor((self.t_iwin_2_end - 
                                self.t_iwin_2_start + 1)/(self.n_history_points/2))
        self.n_history_actions = 2 # how many actions to include in the state vector
        
        # Action related        
        self.n_possible_actions = self.get_num_possible_actions()
        self.possible_actions = self.get_all_possible_actions()
        self.last_action_index = 0 # 0 - no subject received any infusion
        self.last_action = self.possible_actions[:,self.last_action_index]
        self.action_history_for_s = np.zeros((n_subjects,self.n_history_actions)) #
        # Create initial state
        # [a,b] in the following: [fluid in first win, fluid in second win]
        # treat_cond: 1 - [0,0], 2 - [1,0], 3 - [0,1], and 4 - [1,1] 
        tc = 1        
        hr = self.df.loc[self.df.treat_cond==tc,[f'x_hr_{x}' for x in
                        range(self.t_iwin_1_start - self.n_history_points, 
                              self.t_iwin_1_start)]]
        sbp = self.df.loc[self.df.treat_cond==tc,[f'x_sbp_{x}' for x in 
                        range(self.t_iwin_1_start - self.n_history_points,
                              self.t_iwin_1_start)]]
        self.initial_state = np.hstack((hr, sbp, self.iwin_end_code, 
                                        self.action_history_for_s)) # nSub x k
        
               
    def get_next_state(self,curr_state,action):
        """ Get next state given action
        Inputs:
            curr_state - 2d numpy array of shape (n_subjects,k) where k is 
            self.n_history_points*2(hr and sbp) + self.iwin_code_bin_size + 
            self.n_history_actions; 
            order of elements: the following set of values from each subject is
            concatenated:
            n_history_points of hr, n_history_points of sbp, code indicating
            in which infusion window we are currently in, 1 or 0 indicating 
            if fluid was infused in the first infusion window, 1 or 0 indicating
            if fluid was inflused in the second infusion window
            action - 1d numpy array of size n_subjects; each element of action
                     is 1 or 0
        Outputs:
            sp - next state; shape same as curr_state
            r - reward
        """
        sp = []
        for i_sub,(sr,a) in enumerate(zip(curr_state,action)): # iterate through rows (subjects)
            non_phys = np.copy(sr[2*self.n_history_points:])
            phys_curr = np.copy(np.reshape(sr[:2*self.n_history_points],(1,-1)))
            # First get the infusion window and last action from the curr_state
            w = self.iwin_end_code_width
            iwin = self.onehotencoder.inverse_transform([non_phys[0:w]])[0,0]
            pa = np.reshape(sr[-self.n_history_actions:],(1,-1)) # past actions
            match iwin:
                case 0:
                    # Ignore past actions as this is the initial state in the
                    # episode
                    # treat_cond: 1 - [0,0], 2 - [1,0], 3 - [0,1], and 4 - [1,1] 
                    if a==0: # no infusion
                        tc = 1                        
                    elif a==1: # infuse one unit
                        tc = 2
                    else:
                        raise ValueError('a must be 0 or 1')
                    t_start = self.t_iwin_1_start
                    t_end = self.t_iwin_1_end
                    step_size = self.history_step_size_iwin1                    
                case 1:
                    # treat_cond: 1 - [0,0], 2 - [1,0], 3 - [0,1], and 4 - [1,1] 
                    if a==0: # no infusion
                        if pa[iwin-1]==0:
                            tc = 1
                        elif pa[iwin-1]==1:
                            tc = 2
                    elif a==1: # infuse one unit
                        if pa[iwin-1]==0:
                            tc = 3
                        elif pa[iwin-1]==1:
                            tc = 4
                    else:
                        raise ValueError('a must be 0 or 1')                        
                    t_start = self.t_iwin_2_start
                    t_end = self.t_iwin_2_end
                    step_size = self.history_step_size_iwin2
                case _:
                    raise ValueError('t_step must be 0, 1, or 2')
                    
            sel_t = np.flip(np.arange(t_end, t_start, -step_size))\
                                [0:int(self.n_history_points/2)].astype(int)
            hr_next = np.array(self.df.loc[self.df.treat_cond==tc,[f'x_hr_{x}'
                                                        for x in sel_t]])[[i_sub],:]
            sbp_next = np.array(self.df.loc[self.df.treat_cond==tc,[f'x_sbp_{x}' 
                                                for x in sel_t]])[[i_sub],:]                  
            # Combine with the current state
            iwin_next = iwin+1
            act_hist = np.copy(pa) # n_history_actions elements
            act_hist[iwin] = a
            iwin_code = self.onehotencoder.transform(
                                        [[iwin_next]]).toarray()
            sp.append(np.hstack((phys_curr, hr_next, sbp_next, 
                                        iwin_code, act_hist)))
            
        return np.vstack(sp)
                    
    def get_init_state(self):
      s = self.get_state(t_step=0)
    
    def get_num_possible_actions(self):
        nc = 0
        for k in range(1,self.n_fluid_units+1):
           nc += comb(self.n_subjects, k)
        return nc+1 # 1 for not giving any fluids to anybody
    
    def get_all_possible_actions(self):
        all_actions = np.zeros((self.n_subjects, 
                                self.get_num_possible_actions()))
        sub_ind = np.arange(self.n_subjects)
        i = 0
        for k in range(self.n_fluid_units):
            ind_comb = combinations(sub_ind,k)
            for cc in ind_comb:
                all_actions[cc,i] = 1
                i += 1
        return all_actions
    
class DDQNAgent():
    pass       
    
#%%
if __name__ == '__main__':
    np.random.seed(100)
    env = Environment(8,6)
    env.get_next_state(env.initial_state,env.last_action)
    a = env.get_all_possible_actions()
    ddqn_agent = DDQNAgent()
    n_episodes = 100
    n_timesteps = 2
    for i_episode in range(n_episodes):
        # Pick starting state
        s = env.get_init_state() # current state s
        for i_timestep in range(n_timesteps):
            a = ddqn_agent.choose_action(s) # a - action
            sp, r, is_terminal = env.step(a) # next state s-prime or sp, r-reward
            s = sp
            # Store transition in replay buffer 
            ddqn_agent.store(s, a, r, sp, is_terminal)
            # Agent will start learning once enough experiences accumulate
            ddqn_agent.learn()
            
            
            
   
             


