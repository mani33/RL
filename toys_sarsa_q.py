# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:11:02 2024

Toy examples for SARSA and Q-learning

@author: msubramaniyan
"""
#%% Import
import numpy as np
import copy
import matplotlib.pyplot as plt

#%% Functions
class Params():
    def __init__(self,n_row=4,n_col=4):
        # based on a 4-by-12 grid, we define edges
        n_row,n_col = 4,12
        self.grid = dict(n_row=n_row, n_col=n_col)
        self.n_a = 4 # number of actions
        self.As = np.array([[-1,0],[1,0],[0,-1],[0,1]],dtype=int) # 'up','down','left','right'       
        self.alpha = 0.1
        self.epsilon = 0.5
        self.gamma = 1.0
        self.s_goal = np.array([n_row,n_col],dtype=int)-1       
        self.r_cliff_off = -100
        self.ep_greedy_prob = 1.0-self.epsilon + (self.epsilon/self.n_a)
        self.s_begin = np.array([3,0],dtype=int)
        self.s_cliff = np.array([[int(3),int(i)] for i in range(1,11)])
        
def init_r_fun(p):
    rf = np.ones((p.grid['n_row'],p.grid['n_col']))*-1
    # Cliff grids
    for rc in p.s_cliff:
        rf[rc[0],rc[1]] = p.r_cliff_off
        
    return rf

def init_q_fun(p):   
    qf = np.random.uniform(0,1,size=(p.grid['n_row'],p.grid['n_col'],p.n_a))
    # Goal grid location's q value is zero
    qf[p.s_goal[0],p.s_goal[1],:] = 0
    
    return qf

def get_neighbor_locs(s,p):
    # location indices order: up, down, left, right
    r = np.array([-1,1,0,0],dtype=int)+s[0]
    c = np.array([0,0,-1,1],dtype=int)+s[1]
    rc = [fix_exits(np.array([ri,ci]),s,p)[0] for ri,ci in zip(r,c)]
    
    return rc

def fix_exits(new_loc,old_loc,p):
    # sp: 2-element array [row,col]
    if (new_loc[1] < 0) | (new_loc[1] > (p.grid['n_col']-1)): # going out the left or right of the grid
        new_loc[1] = old_loc[1]
    if (new_loc[0] < 0) | (new_loc[0] > (p.grid['n_row']-1)): # going out on above or below the grid
        new_loc[0] = old_loc[0]
    # Check for cliff location
    fell_off_cliff = False
    if np.any([np.all(new_loc==x) for x in p.s_cliff]):
        new_loc=np.copy(p.s_begin)        
        fell_off_cliff = True
    # Keep the agent in the goal
    if np.all(old_loc==p.s_goal):
        new_loc = p.s_goal
    return new_loc,fell_off_cliff

def get_act_prob_ep_greedy(q_as,p):
    # Get probability values based on epsilon-greedy policy for the given sequence
    # of Q(S,A) values
    p_nongreedy = p.epsilon/p.n_a
    p_greedy = 1-p.epsilon + p_nongreedy
    ap = p_nongreedy*np.ones_like(q_as)
    # Break ties randomly
    m = np.nonzero(q_as==np.max(q_as))[0]
    if m.size > 1:
        m = np.random.permutation(m)[0]
    ap[m] = p_greedy
    
    return ap

def select_ep_greedy_act(q_as,p):
    """ Select an action based on epsilon greedy action
    Inputs:
        q_as - 1d array of action values
        p - Params object
    Outputs:
        a - index of selected action            
    """
    x = np.random.uniform()
    if x < p.ep_greedy_prob:
        a = np.argmax(q_as)
    else:
        # Pick totally randomly
        a = np.random.randint(p.n_a)
    
    return a
        
    
def step(s,a,r_fn,p):
    """
    Take one step of action, reach next state and obtain reward
    Inputs:
        s - state, 1d np array, [row,col] indices
        a - action index
        r_fn - 2d np array of reward value for each grid position
        p - Params object
    Outputs:
        sp - next state, [row,col]
        r - reward (scalar)
        goal_reached - boolean, indicates if agent stepped into the goal state
    """
    
    # Get the next state and reward for the selected action
    sp = s+p.As[a]
    sp,fell_off_cliff = fix_exits(sp,s,p)
    r = r_fn[sp[0],sp[1]]
    if fell_off_cliff:
        r = p.r_cliff_off
    goal_reached = is_terminal_state(sp,p)    
    
    return sp, r, goal_reached

def update_q(s,sp,a,ap,r,Q_fn,learn_meth,p):
    # learn_meth - sarsa, expected_sarsa, or q_learning
    
    # Current state
    q_curr = np.copy(Q_fn[s[0],s[1],a])
    # Next state
    q_sa_next = np.copy(Q_fn[sp[0],sp[1]])
    
    match learn_meth:
        case 'sarsa':
            # Get next state and action pair's q value 
            next_q_sa_estimate = q_sa_next[ap] 
        case 'expected_sarsa':
            # From the next state, get action probabilities according to the 
            # agent's behavior policy
            ap = get_act_prob_ep_greedy(q_sa_next,p)
            # Get expectecd action value when taking the next action from the
            # next state
            expected_q_sa = np.sum(q_sa_next*ap)
            next_q_sa_estimate = expected_q_sa
        case 'q_learning':
            # Get state-action value of the next action that leads to the maximum
            # state-action value
            next_q_sa_estimate = np.max(q_sa_next)
            
    # Update current state-action pair's value
    q_curr = q_curr + p.alpha*(r + (p.gamma*next_q_sa_estimate) - q_curr)
    Q_fn[s[0],s[1],a] = q_curr
          
    return Q_fn

def is_terminal_state(s,p):
    g = False
    if np.all(s==p.s_goal):
        g = True
        
    return g
#%%
if __name__ == '__main__':
    np.random.seed(10)
    p = Params()
    Q_fn = init_q_fun(p)
    r_fn = init_r_fun(p) 
    policy = 'epsilon_greedy'
    learn_meth = 'expected_sarsa' # 'sarsa','expected_sarsa','q_learning'
    
    n_episodes = 5000
    Q = copy.deepcopy(Q_fn)
    
    for iEpi in range(n_episodes):
        ssx,ssy = [],[]
        c = 0
        goal_reached = False
        # First choose an initial state and action
        s = p.s_begin
        a = select_ep_greedy_act(Q_fn[s[0],s[1]], p)
        while (not goal_reached) & (c < 5000):
            # Act and get next state and reward
            sp,r,goal_reached = step(s,a,r_fn,p)            
            q_sap = np.copy(Q_fn[sp[0],sp[1]])
            # Select an action from the next state
            ap = select_ep_greedy_act(q_sap,p)
            Q_fn = update_q(s,sp,a,ap,r,Q_fn,learn_meth,p)
            # Prepare for the next iteration
            s = sp
            a = ap
            c+=1
            # Terminate loop        
            if goal_reached:
                print(iEpi,c)
        
    pol = np.zeros((4,12))
    for i in range(4):
        for j in range(12):
            pol[i,j] = np.argmax(Q_fn[i,j])
        
    print(pol)
    
    
    
    