from scipy.optimize import minimize
import numpy as np
from pcvar import distorted_exp
from mdp import get_reward_dist_per_state
from cvar_helpers import calc_cvar_from_quantiles

def get_quantile_estimates(next_state, taus, states_all, rewards_all):

    # get quantiles (estimates) for next states
    rewards_per_state = get_reward_dist_per_state(next_state, states_all, rewards_all)
    thetas = np.array([np.quantile(rewards_per_state.squeeze(), tau) for tau in taus])

    return(thetas)

def calculate_next_CVaR(next_states, taus, alphas, states_all=None, rewards_all=None):
    V = []
    for next_state in next_states:
        thetas = get_quantile_estimates(next_state, taus, states_all=states_all, rewards_all=rewards_all)
        cvars = calc_cvar_from_quantiles(thetas, taus, alphas)
        V.append(cvars)
    return(np.array(V))

def distort_probabilities(p, alpha, alphas, V, max_inner_iters=50, multi_starts_N=10, same_answer_ns=3, same_answer_tol=1e-4):

    def sum_to_1_constraint(xi):
        zero = np.dot(xi,p)-1
        return(zero)

    xps = np.arange(len(p))
    cons = ({'type': 'eq', 'fun': sum_to_1_constraint})
    bnds = tuple(((0.0,1.0/alpha) for i in range(len(p))))

    succeeded_at_least_once = False
    results_list = []
    fun_mins = []
    for iter in range(max_inner_iters):

        xis_init = np.random.uniform(alphas[1],1/alpha,len(p))
        #dist_probs_init = np.random.dirichlet(np.ones(len(p)),size=1)[0]
        #xis_init = dist_probs_init/np.array(p)
        #if np.abs(sum_to_1_constraint(xis_init))>1e-4:
        #    import ipdb; ipdb.set_trace()
        #assert np.abs(sum_to_1_constraint(xis_init))<1e-4

        results = minimize(distorted_exp,
                           xis_init,
                           args=(V, xps, p, alpha, alphas),
                           method='SLSQP',
                           bounds=bnds,
                           constraints=cons)

        if results.success:
            succeeded_at_least_once=True
            results_list.append(results)
            fun_mins.append(results.fun)

        # exit early if all N minimums are the same
        if len(fun_mins)>same_answer_ns:
            minn = np.min(np.array(fun_mins))
            num_within_tol = np.sum((np.array(fun_mins)-minn)<same_answer_tol)
            if num_within_tol>=same_answer_ns:
                break

        # or exit after max number of multi-starts have been exceeded
        if len(fun_mins)>multi_starts_N:
            break

        if len(fun_mins)==0:
            print('inner optimization failed')
        else:
            # find minimum over multi-starts
            argmin_funs = np.argmin(np.array(fun_mins))
            results = results_list[argmin_funs]

    xis = results.x
    p_distorted = p*xis
    if not results.success:
        print('inner optimization failed')

    extra = {}
    extra['fun_mins']=fun_mins
    extra['success']=results.success

    return(p_distorted, xis, extra)


def cvar_forward_sampler(P, R, s0, alpha0, n_eps=1, n_quantiles = 10, verbose = False,
                 states_all_prev=None, rewards_all_prev=None, max_seq_len=4):

    taus = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
    alphas = np.append(np.insert(taus, 0, 0), 1) # add zero, one
    states_all = []
    rewards_all = []
    alphas_all = []

    for ep in range(n_eps):
        states_ep = []
        rewards_ep = []
        alphas_ep = []
        s = s0; r=0; alpha=alpha0
        done = False

        for i in range(max_seq_len):

            states_ep.append(s)
            rewards_ep.append(r)
            alphas_ep.append(alpha)

            # get next states, probs
            p = P[s,:]
            next_states = np.where(p!=0)[0]
            p = p[next_states]

            if np.sum(p)==0: # end of a branch
                sp = -1
                r = 0

            else:

                # distort probs
                V = calculate_next_CVaR(next_states, taus, alphas, states_all=states_all_prev, rewards_all=rewards_all_prev)
                p_distorted, xis, _ = distort_probabilities(p, alpha, alphas, V)
                if np.abs(p_distorted.sum()-1.)>0.05:
                    import ipdb; ipdb.set_trace()
                p_distorted = p_distorted/np.sum(p_distorted)

                # sample next state
                sp = np.random.choice(next_states, p=p_distorted, size=1)[0]
                alpha = xis[next_states==sp]*alpha
                alpha = np.max(np.min((alpha,1)),0)

                if sp in R.keys():
                    r = R[sp].rvs()
                else:
                    r = 0

            s = sp

        states_all.append(states_ep)
        rewards_all.append(rewards_ep)
        alphas_all.append(alphas_ep)

    states_all = np.array(states_all)
    rewards_all = np.array(rewards_all).sum(axis=1)
    alphas_all = np.array(alphas_all)

    return(states_all, rewards_all, alphas_all)
