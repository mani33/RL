# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:42:11 2024
Function approximation for state value estimation using neural networks
Coursera Reinforcement learning specialization
C3M2 Assignment2: random walk state value estimation - my own implementation
Author: Mani Subramaniyan
"""
#%% Import
import numpy as np
import matplotlib.pyplot as plt
#%%
class Params():
    def __init__(self):
        self.n_states = 500
        self.max_jump = 100
        self.neighbors_left = np.arange(-self.max_jump,0)
        self.neighbors_right = np.arange(1,self.max_jump+1)
        self.alpha = 0.001
        self.gamma = 1.0
        self.s_goal = np.array([0,self.n_states+1],dtype=int)
        self.s_begin = int(self.n_states/2)
        self.features = np.eye(self.n_states)
        # NN params
        self.n_hidden_units = 100
        self.beta_m = 0.9
        self.beta_s = 0.999
        self.epsilon = 0.0001

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
    return sp, r, goal_reached

def init_wt(p):
    # Initialize 4 weights: w0, w1, b0, and b1
    # w0: 500x100
    sd0 = np.sqrt(2/(p.n_states+1))
    w0 = np.random.normal(0,sd0,size=(p.n_states,p.n_hidden_units))
    # w1: 100x1
    sd1 = np.sqrt(2/(p.n_hidden_units+1))
    w1 = np.random.normal(0,sd1,size=(p.n_hidden_units,1))
    # b0: 100x1
    b0 = np.random.normal(0,1,size=(p.n_hidden_units,1))
    # b1: 1x1
    b1 = np.random.normal(0,1,size=(1,1))
    wt = dict(w0=w0, w1=w1, b0=b0, b1=b1)
    return wt

def forward_pass(s, wt, p):
    # Get one hot version of s
    S = p.features[:,[s-1]] # shape: (n_states,1): 500x1
    # psi: (1x500.500x100)-->(1x100)T-->(100,1) + (100,1)
    psi = np.matmul(S.T, wt['w0']).T + wt['b0'] # final dim: 100x1
    # ReLU output
    relu_in = np.hstack((np.zeros_like(psi),psi)) # 100x2
    X = np.max(relu_in,axis=1,keepdims=True) # 100x1
    # v: (1x100.100x1) + 1
    v = (np.matmul(X.T,wt['w1']) + wt['b1'])[0,0] # extract the float out
    return S, X, v

def get_gradients(S, X, wt):
    db1 = 1
    dw1 = X # 100x1
    Ix = (X > 0).astype(float) # 100x1
    db0 = np.multiply(wt['w1'],Ix) # element-wise matrix multiplication: 100x1
    dw0 = np.matmul(S,db0.T) # 500x1.1x100
    dw = dict(b0=db0, b1=db1, w0=dw0, w1=dw1)
    return dw

def init_adam_params(p):
    wp = ['w0', 'w1', 'b0', 'b1']
    adam = dict(mt={}, st={})
    for w in wp:
        adam['mt'][w] = 0
        adam['st'][w] = 0
    return adam

def update_weights(s, sp, r, wt, adam, p, t, goal_reached):
    # Get current and next state values
    S,X,v = forward_pass(s, wt, p)
    if goal_reached:
        vp = 0 # goal state's value is always zero
    else:
        vp = forward_pass(sp, wt, p)[2]
    grad = get_gradients(S, X, wt)
    td_err = r + (p.gamma*vp)-v
    for key, dw in grad.items():
        gt = td_err * dw
        adam['mt'][key] = p.beta_m*adam['mt'][key] + (1-p.beta_m)*gt
        adam['st'][key] = p.beta_s*adam['st'][key] + (1-p.beta_s)*np.square(gt)
        # # Bias correction - created issues so not going to use it
        # dn_m = 1-(p.beta_m**t)
        # dn_s = 1-(p.beta_s**t)
        # # print(f'deno of m and s: {dn_m, dn_s}')
        # adam['mt'][key] = adam['mt'][key]/dn_m
        # adam['st'][key] = adam['st'][key]/dn_s
        # Adam
        a = adam['mt'][key]/(np.sqrt(adam['st'][key])+p.epsilon)
        # Weight update
        # a = gt
        wt[key] = wt[key] + (p.alpha * a)
    return wt, adam

def run_one_episode(p, wt, t, adam):
    goal_reached = False
    s = p.s_begin
    while not goal_reached:
        # Select an action
        a = np.random.randint(0,2)
        # Make a step
        sp, r, goal_reached = step(s, a, p)
        # Update weights
        wt, adam = update_weights(s, sp, r, wt, adam, p, t, goal_reached)
        s = sp
        t += 1
    return wt, adam, t

#%%
np.random.seed(15)

p = Params()
p.alpha = 0.001
p.beta_m = 0.9
p.beta_s = 0.999
n_runs = 1
n_epi = 5000
v = np.zeros((p.n_states,n_runs))
for n in range(n_runs):
    wt = init_wt(p)
    adam = init_adam_params(p)
    t = 1
    for i in range(n_epi):
        wt, adam, t  = run_one_episode(p, wt, t, adam)
        print(n+1,i+1)
    for s in range(p.n_states):
        v[s,n] = forward_pass(s+1, wt, p)[2]

#%%
plt.plot(np.mean(v,axis=1))
plt.ylim([-1,1])
plt.plot(plt.xlim(),[-1,1],color='r')