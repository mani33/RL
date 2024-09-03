# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:02:05 2024
Toy examples of RL
@author: msubramaniyan
"""
#%% Import
import numpy as np
import matplotlib.pyplot as plt
#%% Functions
# 4x4 grid
# state space
class Params():
    def __init__(self,toy,n_row=4,n_col=4):
        # based on a 4-by-4 grid, we define edges
        n_row,n_col = 4,4
        self.grid = dict(n_row=n_row, n_col=n_col)
        self.As = np.array([[-1,0],[1,0],[0,-1],[0,1]],dtype=int) # 'up','down','left','right'       
        self.gamma = 1
        self.alpha = 0.5
        self.n_val_iter = 10
        self.toy = toy
        match toy:
            case 'two_targets':
                self.s_targets = [np.array([0,0]),np.array([n_row,n_col])-1] # target states
                self.v_targets = [0,0] # target's value-function entry
            case 'one_target':
                self.s_targets = [np.array([0,0])] # target states
                self.v_targets = [0] # target's value-function entry
        self.rand_act_p = np.ones((4))*0.25 # random action probability pi(a|s)
        
def init_r_fun(p):
    rf = np.ones((p.grid['n_row'],p.grid['n_col']))*-1
    if p.toy=='one_target':
        # Blue colored blocks
        rf[0,1:] = -10
        rf[2,0:3] = -10
    
    return rf

def init_v_fun(p):   
    vf = np.zeros((p.grid['n_row'],p.grid['n_col']))
    # Apply targets' values
    for rc,v in zip(p.s_targets,p.v_targets):
        vf[rc[0],rc[1]]=v
        
    return vf

def get_next_state(s,v_fn,r_fun,policy,p):
    # s - state - a list specifying row and column in the grid. Example: (0,1)
   
    nA = p.As.shape[0]

    match policy:
        case 'random':
            a = p.As[np.random.permutation(nA)[0]]
            # s_prime
            sp = s+a
            sp = fix_exits(sp, s, p.grid)           
        case 'greedy':
            rc = get_neighbor_locs(s, p.grid)
            # Get values of neighboring states
            best_ind = np.argmax([v_fn[loc[0],loc[1]] for loc in rc])
            sp = rc[best_ind]
            
    r = r_fun[sp[0],sp[1]]
    
    return r,sp

def get_neighbor_locs(s,grid):
    # grid - dict containing 'n_row' and 'n_col' keys
    # location indices order: up, down, left, right
    r = np.array([-1,1,0,0],dtype=int)+s[0]
    c = np.array([0,0,-1,1],dtype=int)+s[1]
    rc = [fix_exits([ri,ci],s,grid) for ri,ci in zip(r,c)]
    
    return rc

def fix_exits(new_loc,old_loc,grid):
    # grid - dict containing 'n_row' and 'n_col' keys
    if (new_loc[1] < 0) | (new_loc[1] > (grid['n_col']-1)): # going out the left or right of the grid
        new_loc[1] = old_loc[1]
    if (new_loc[0] < 0) | (new_loc[0] > (grid['n_row']-1)): # going out on above or below the grid
        new_loc[0] = old_loc[0]
        
    return new_loc

def perform_value_iteration(v_fun_old,r_fun,p,policy,update_meth='synchronous'):
    v_fun = np.copy(v_fun_old)
    best_actions = np.zeros_like(v_fun_old)
    match update_meth:
        case 'asynchronous':
            for i in range(p.n_val_iter):
                s_rand = np.array([np.random.permutation(p.grid['n_row'])[0],
                          np.random.permutation(p.grid['n_col'])[0]])
                # Exclude target states from getting updated
                if np.any([np.all(s_rand==x) for x in p.s_targets]):
                    continue
                vs = compute_state_value(s_rand,r_fun,v_fun,p,policy)
                v_fun[s_rand[0],s_rand[1]] = vs
        case 'synchronous':
            # Go sequentially
            for i in range(p.grid['n_row']):
                for j in range(p.grid['n_col']):
                    s = np.array([i,j])
                    # Exclude target states from getting updated
                    if np.any([np.all(s==x) for x in p.s_targets]):
                        continue
                    
                    vs,a = compute_state_value(s,r_fun,v_fun_old,p,policy)
                    v_fun[s[0],s[1]] = vs
                    best_actions[s[0],s[1]] = a
        case _:
            raise ValueError(f'unknown update meth method: {update_meth} ')
               
    return v_fun,best_actions
        
def compute_state_value(s,r_fun,v_fun,p,policy):
    # Update the value of this selected random state
    # Get possible next states
    
    s_nei = get_neighbor_locs(s, p.grid)
    # Get rewards and values of next states
    r_nei = np.array([r_fun[rc[0],rc[1]] for rc in s_nei])
    v_nei = np.array([v_fun[rc[0],rc[1]] for rc in s_nei])
    # Get action values (q value) 
    q_as = r_nei+p.gamma*v_nei
    match policy:
        case 'random':
            vs = np.sum(p.rand_act_p*q_as)
            a = 0
        case 'greedy':
            vs = np.max(q_as)
            a = np.argmax(q_as)
            
    return vs,a
#%% Test
if __name__=='__main__':
    # toy = 'two_targets' # 'two_targets' or 'one_target'
    toy = 'one_target'
    p = Params(toy)
    rf = init_r_fun(p)
    vf = init_v_fun(p)
    vf_old = np.copy(vf)
    d = 0.11
    d_min = 0.001
    c = 0
    while (d > d_min):
        vf_new,ba = perform_value_iteration(vf_old, rf, p, 'random')
        d = np.max(np.abs(vf_old-vf_new))
        vf_old = vf_new
        c+=1
        print(c,d)
        print(np.round(vf_new,1))
#%%
    d = 0.11
    c = 0
    while (d > d_min):
        vf_new,ba = perform_value_iteration(vf_old, rf, p, 'greedy')
        d = np.max(np.abs(vf_old-vf_new))
        vf_old = vf_new
        c+=1
        print(c,d)
        print(np.round(vf_new,3))
        print(ba)