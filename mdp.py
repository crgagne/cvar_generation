import numpy as np
from scipy.stats import norm

def get_reward_dist_per_state(state, states_all, rewards_all):
    episode_sel = (states_all==state).sum(axis=1).astype('bool')
    return(rewards_all[episode_sel])

def get_mdp(mdp_name):
    print(mdp_name)
    if mdp_name=='9tree':
        # create an MDP
        n_states = 9
        states = np.arange(n_states)

        P = np.zeros((n_states,n_states))
        P[0,[1,2]]=0.5
        P[[1],[3,4]]=0.5
        P[[2],[5,6]]=0.5
        P[6,[7,8]]=0.5

        R = np.zeros((n_states))
        terminal_states = [3,4,5,7,8]

        std = 0.5
        means = [-2,-1,2,3,4]
        R = {ts:norm(mean,std) for ts,mean in zip(terminal_states, means)}

    elif mdp_name=='9tree_mix':

        # create an MDP
        n_states = 9
        states = np.arange(n_states)

        P = np.zeros((n_states,n_states))
        P[0,[1,2]]=0.5
        P[[1],[3,4]]=0.5
        P[[2],[5,6]]=0.5
        P[6,[7,8]]=0.5

        R = np.zeros((n_states))
        terminal_states = [3,4,5,7,8]

        std = 0.5
        means = [-2,2,-2,-2,2]
        R = {ts:norm(mean,std) for ts,mean in zip(terminal_states, means)}

    elif mdp_name=='9tree_mix2':

        # create an MDP
        n_states = 9
        states = np.arange(n_states)

        P = np.zeros((n_states,n_states))
        P[0,[1,2]]=0.5
        P[[1],[3,4]]=0.5
        P[[2],[5,6]]=0.5
        P[6,[7,8]]=0.5

        R = np.zeros((n_states))
        terminal_states = [3,4,5,7,8]

        std = 0.25
        means = [-3,2,-2,-5,2]
        R = {ts:norm(mean,std) for ts,mean in zip(terminal_states, means)}

    return(P, R, n_states, states)

def simulate_from_mdp(seed, P, R, n_states, states, n_episodes = 10000, max_seq_len = 4):

    # sample experiences
    np.random.seed(seed)

    states_all = []
    states_onehot_all = []
    rewards_all = []
    for ep in range(n_episodes):
        s = 0; r=0; done = False
        s_onehot = np.zeros(n_states)
        s_onehot[s] = 1
        states_ep = []
        rewards_ep = []
        states_onehot_ep=[]

        for i in range(max_seq_len):

            states_ep.append(s)
            rewards_ep.append(r)
            states_onehot_ep.append(s_onehot)

            p = P[s,:].squeeze() # transition probs

            if np.sum(p)==0: # end of a branch
                sp = -1
                r = 0
                s_onehot = np.zeros(n_states)
            else:
                sp = np.random.choice(states, p=p, size=1)[0]
                if sp in R.keys():
                    r = R[sp].rvs()
                else:
                    r = 0
                s_onehot = np.zeros(n_states)
                s_onehot[sp]=1

            s=sp

        states_all.append(states_ep)
        rewards_all.append(rewards_ep)
        states_onehot_all.append(states_onehot_ep)
    states_all = np.array(states_all)
    rewards_all = np.array(rewards_all).sum(axis=1)[:,np.newaxis] # only take last dim but preserve shape
    states_onehot_all = np.array(states_onehot_all)
    mask = (states_all!=-1).astype('int')

    return(states_all, rewards_all, states_onehot_all, mask)
