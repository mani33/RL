# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:23:58 2024

State value estimation using function approximation
Toy examples - 500 state random walk using Gradient Monte-carlo and semi-gradient
TD0

@author: msubramaniyan

"""
#%% Import
import numpy as np
import matplotlib.pyplot as plt
#%%
class Params():
    def __init__(self, toy_name, learning_meth='Monte_carlo', n_states=500,
                                                 max_jump=100, n_bins=5):
        match toy_name:
            case 'random_walk':
                self.learning_meth = learning_meth # Monte_carlo or TD0
                self.n_states = n_states
                self.n_bins = n_bins
                self.neighbors_left = np.arange(-max_jump,0)
                self.neighbors_right = np.arange(1,max_jump+1)
                self.max_jump = max_jump #
                self.alpha = 0.01
                self.gamma = 1.0
                self.s_goal = np.array([0,n_states+1],dtype=int)
                self.s_begin = int(n_states/2)
                self.toy = toy_name
                self.bin_size = int(n_states/n_bins)
                self.features = np.eye(self.n_bins)
            case _:
                raise NotImplementedError()

def init_wt(p):
    match p.toy:
        case 'random_walk':
            wt = np.zeros((p.n_bins,1))
        case _:
            raise NotImplementedError()
    return wt

def step(s,a,p):
    """
    Take one step of action, reach next state and obtain reward
    Inputs:
        s - current state, integer index
        a - action, integer index
        p - Params object
    Outputs:
        sp - next state, integer index
        r - reward (scalar)
        goal_reached - boolean, indicates if agent stepped into the goal state
    """
    # Get the next state and reward for the selected action

    # Get the neighboring group of states to which agent may jump to.
    goal_reached = False
    r = 0
    match p.toy:
        case 'random_walk':
            if a==0: # Left
                new_neighbors = p.neighbors_left + s
            else: # Right
                new_neighbors = p.neighbors_right + s
            # Pick one state to go to.
            sp = np.random.permutation(new_neighbors)[0]
            # Terminal states
            if sp <=0:
                sp = 0
                r = -1.0
                goal_reached = True
            elif sp >= (p.n_states+1):
                sp = p.n_states+1
                r = 1.0
                goal_reached = True
        case _:
            raise NotImplementedError()

    return sp, r, goal_reached

def get_aggregate_state(s,p):
    return int(np.ceil(s/p.bin_size)-1)

def update_wt(s,sp,Gt,r,wt_old,p):
    s_agg = get_aggregate_state(s,p)
    sp_agg = get_aggregate_state(sp,p)
    xs = p.features[:,[s_agg]] # need [] around s to keep xs as 2d
    vs = np.matmul(wt_old.T,xs)[0,0]
    match p.learning_meth:
        case 'Monte_carlo':
            err = Gt - vs
        case 'TD0':
            xsp = p.features[:,[sp_agg]]
            vsp = np.matmul(wt_old.T,xsp)[0,0]
            err = r + p.gamma*vsp - vs
        case _:
            raise NotImplementedError()
    # Update
    grad = xs
    wt_new = wt_old + (p.alpha * err * grad)
    return wt_new

def compute_vs(wt,X):
    v = np.matmul(wt.T,X)
    return v

def update_wt_final(s,Gt,r,wt_old,p):
    # When the next state is terminal state
    s_agg = get_aggregate_state(s,p)
    xs = p.features[:,[s_agg]] # need [] around s to keep grad as 2d
    vs = np.matmul(wt_old.T,xs)
    match p.learning_meth:
        case 'Monte_carlo':
            err = Gt - vs
        case 'TD0':
            vsp = 0 # next state is goal, and goal state's value is always 0
            err = r + (p.gamma * vsp) - vs
        case _:
            raise NotImplementedError()
    # Update
    grad = xs
    wt_new = wt_old + (p.alpha * err * grad)
    return wt_new

def run_one_episode(p):
    goal_reached = False
    s = p.s_begin
    s_epi = []
    r_epi = []
    while not goal_reached:
        a = np.random.randint(0,2)
        s_epi.append(s)
        sp, r, goal_reached = step(s,a,p)
        r_epi.append(r)
        s = sp
    Gt = r
    # print(s,r)
    return s_epi, r_epi, Gt

def update_wt_one_episode(s_epi, r_epi, Gt, wt, p):
    for i,(s,r) in enumerate(zip(s_epi, r_epi)):
        if i < (len(s_epi)-1):
            sp = s_epi[i+1]
            wt = update_wt(s, sp, Gt, r, wt, p)
        else: # goal reached
            wt = update_wt_final(s, Gt, r, wt, p)
    return wt

#%% Monte-carlo

p = Params('random_walk')
p.learning_meth = 'Monte_carlo' # 'Monte_carlo' or 'TD0'
wt = init_wt(p)
p.alpha = 0.01
n_epi = 1000
for _ in range(n_epi):
    s_epi,r_epi,Gt = run_one_episode(p)
    wt = update_wt_one_episode(s_epi, r_epi, Gt, wt, p)

# print(np.ravel(wt))
plt.plot(wt,marker='o')
plt.ylim([-1,1])
plt.grid()
print(wt)