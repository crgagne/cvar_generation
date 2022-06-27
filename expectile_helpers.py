import numpy as np
import scipy
import scipy.stats





def expectile_loss(expectiles,taus,samples):
    '''
    delta = z - q
    loss = E[I(z>q)*tau + (1-I(z>q))*(1-tau)]*(delta^2)
    see Rowland et al. 2019 Definition 3.3.
    '''
    delta = (samples[None, :] - expectiles[:, None])
    indic = np.array(delta >= 0., dtype=np.float32)
    loss = np.mean((indic*taus[:,None] + (1-indic)*(1-taus[:,None]))*delta**2)
    return(loss)

def grad_expectile_loss(expectiles,taus,samples):
    '''
    Grad of above loss function
    '''
    delta = (samples[None, :] - expectiles[:, None])
    indic = np.array(delta <= 0., dtype=np.float32)
    grad =  np.abs(taus[:, None] - indic) * delta * -0.5
    return(grad)


def expectile_loss_fn(expectiles, taus, samples):
    '''
    From Nature Paper Online Code (their notes below)
    It's the Expectile loss function for finding samples this time instead of expectiles.
    see Rowland et al. 2019 eqn. 8

    '''

    # distributional TD model: delta_t = (r + \gamma V*) - V_i
    # expectile loss: delta = sample - expectile
    delta = (samples[None, :] - expectiles[:, None])

    # distributional TD model: alpha^+ delta if delta > 0, alpha^- delta otherwise
    # expectile loss: |taus - I_{delta <= 0}| * delta^2

    # Note: When used to decode we take the gradient of this loss,
    # and then evaluate the mean-squared gradient. That is because *samples* must
    # trade-off errors with all expectiles to zero out the gradient of the
    # expectile loss.
    indic = np.array(delta <= 0., dtype=np.float32)
    grad =  np.abs(taus[:, None] - indic) * delta * -0.5
    return np.mean(np.square(np.mean(grad, axis=-1)))


def find_expectiles(expectiles0,taus,samples,method='grad_desc',alpha=0.01,
                    max_iters=10000,precision=0.0001,printt=False):

    '''
    Given samples, calculate best fitting expectiles.

    Notes on gradient descent:
        - learning rate needs to be low (0.01 or less) for distributions with rare events
        - in general, I don't the grad descent function, but it's nice to know it works

    '''
    if method=='grad_desc':
        expectiles_n = expectiles0.copy()
        for _i in range(max_iters):
            if printt:
                print(_i)
            expectiles_c = expectiles_n

            # grad over entire dataset (versus at a sample)
            grad = np.mean(grad_expectile_loss(expectiles_c,taus,samples),axis=1)
            expectiles_n = expectiles_c - alpha*np.squeeze(grad)

            step = expectiles_c-expectiles_n
            if np.all(np.abs(step)<=precision):
                expectiles = expectiles_n
                print('here')
                break

        expectiles = expectiles_n

    elif method=='optimize_scipy':
        #import pdb; pdb.set_trace()
        fn_to_minimize = lambda x: expectile_loss(x, taus, samples)
        result = scipy.optimize.minimize(
                    fn_to_minimize, method=None, x0=expectiles0)['x']
        expectiles = result

    return(expectiles)


def impute_distribution(expectiles, taus, minv=-10, maxv=10, method=None,
                 max_samples=100, max_epochs=5, N=25):
    """
    From Nature Paper Online Code (their notes below)
    They say 'Run decoding given reversal points and asymmetries (taus).'

    expectiles were reversal points in their code
    Reversal points = (but they were estimated from neurons)

    """

    ind = list(np.argsort(expectiles))
    points = expectiles[ind]
    tau = taus[ind]

    # Robustified optimization to infer distribution
    # Generate max_epochs sets of samples,
    # each starting the optimization at the best of max_samples initial points.
    sampled_dist = []
    for _ in range(max_epochs):
        # Randomly search for good initial conditions
        # This significantly improves the minima found
        #import pdb; pdb.set_trace()
        samples = np.random.uniform(minv, maxv, size=(max_samples, N))
        fvalues = np.array([expectile_loss_fn(points, tau, x0) for x0 in samples])
        #np.array([x0 for x0 in samples])

        # Perform loss minimizing on expectile loss (w.r.t samples)
        x0 = np.array(sorted(samples[fvalues.argmin()]))
        fn_to_minimize = lambda x: expectile_loss_fn(points, tau, x)
        result = scipy.optimize.minimize(
            fn_to_minimize, method=method,
            bounds=[(minv, maxv) for _ in x0], x0=x0)['x']

        #import pdb; pdb.set_trace()
        sampled_dist.extend(result.tolist())

    return np.array(sampled_dist), expectile_loss_fn(points, tau, np.array(sampled_dist))
