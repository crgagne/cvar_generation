import numpy as np
from scipy.optimize import minimize

def _cvar_loss(var,R,alpha):
    Rpos = R-var
    Rpos[Rpos<0]=0
    cvar = var + (1/alpha)*np.mean(Rpos) # trying to maximize that threshold.
    return(cvar)

# calculating CVaR for a distribution
def calc_cvar_from_samples(R,alpha):
    '''
        Args:
        R (np.array): samples from the distribution (these will be reward probabilities [0.4,0.3,0.5,0.4,etc...])
        alpha (float): alpha to calculate CVaR at

    '''
    if alpha==0:
        return(np.nan, np.min(R))

    var0 = [np.mean(-1*R)]
    bnds = [(np.min(-1*R),np.max(-1*R))]

    results = minimize(_cvar_loss, var0, args=(-1*R,alpha),bounds=bnds,method='SLSQP')

    if results.success:
        cvar =-1*results.fun
        var = -1*results.x
        return(var[0],cvar)
    else:
        return(np.nan, np.nan)

def calc_cvar_from_quantiles(thetas, taus, alphas):

    # assume alphas are the same as taus
    cvar_ests = []
    taus_w_zero = np.insert(taus, 0, 0) # so that it's. [0, 0.05, 0.15, ] etc.
    # this helps with summing for the first alpha, so that we can say the dtau is 0.05.
    # and then for the other alphas, the dtau are 0.05, 0.1, 0.1, etc.
    # but it seems to over estimate the alpha=1 case in low samples
    # but in the limit it seems to do well

    # eg.
    # rewards_per_state = get_reward_dist_per_state(state, states_all, rewards_all)
    # thetas = np.array([np.quantile(rewards_per_state.squeeze(), q) for q in taus])
    # CVaRs = np.array([float(calc_cvar_from_samples(rewards_per_state, alpha)[1]) for alpha in taus])
    # cvar_ests = calculuate_cvar_from_quantile_estimates(thetas, taus)
    # plt.scatter(cvar_ests, CVaRs, alpha=0.5)
    # plt.plot([-2.5,0.5],[-2.5,0.5], c='k')

    for alpha in taus:
        cvar = (1 / alpha)*np.sum(np.diff(taus_w_zero)[taus<=alpha]*thetas[taus<=alpha])
        cvar_ests.append(cvar)

    # add 0 and 1 estimates
    cvar_ests = np.insert(cvar_ests, 0, thetas[0])
    cvar_ests = np.append(cvar_ests, np.mean(thetas))

    return(np.array(cvar_ests))
