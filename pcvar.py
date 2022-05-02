import numpy as np
from scipy.optimize import minimize
import time


def interpolate_yV(V,xp,yp,y_set):
    '''interpolate the value function times the risk preference:  y*V(x,y)'''

    yp_i_nearest = np.argmin(np.abs(y_set-yp))
    yp_nearest = y_set[yp_i_nearest]

    assert y_set[0]==0.0

    if yp>1:
        # this is possible
        return(yp*V[xp,len(y_set)-1])
    elif yp<0:
        # should never be less than 0, so break if it does happen
        import pdb; pdb.set_trace()
    elif yp==yp_nearest:
        # no need for interpolation.
        return(yp*V[xp,yp_i_nearest])
    else:
        # find lower and upper y_nearest.
        if yp_nearest<yp:
            yp_i_upper = yp_i_nearest+1
            yp_upper = y_set[yp_i_upper]
            yp_i_lower = yp_i_nearest
            yp_lower = yp_nearest
        elif yp_nearest>yp:
            yp_i_upper = yp_i_nearest
            yp_upper = yp_nearest
            yp_i_lower = yp_i_nearest-1
            yp_lower = y_set[yp_i_lower]

        # Slope
        slope = (yp_upper*V[xp,yp_i_upper] - yp_lower*V[xp,yp_i_lower]) / (yp_upper - yp_lower)

        # Start at lower and difference times the slope
        V_interp = yp_lower*V[xp,yp_i_lower] + slope*(yp-yp_lower)

        return(V_interp)


def distorted_exp(xis,V,xps,p,y,y_set,verbose=False):
    '''
    Function which minimizes next states' summed values by choosing distortion weights.

    Inputs:
    xis = weights (usually 4)
    V = full value function (x,y) (changed to costs, rewards)
    xps = next states (usually 4)
    p = next state probabilities (usually 4)
    y = current threshold
    y_set = set of y's interpolation points, passing to the interpolation function.
    '''
    if np.any(np.isnan(xis)):
        return(np.inf)

    # get interpolated value for each next state
    V_interp = np.empty(len(xis))
    yps = np.empty(len(xis))

    # loop over next states and weights
    for i,(xp,xi) in enumerate(zip(xps,xis)):
        yp = y*xi # get next state y
        yps[i]=yp
        V_interp[i] = interpolate_yV(V,xp,yp,y_set) # interpolate

    distorted_exp = np.sum(p*V_interp/y) 

    if verbose:
        print('next states='+str(np.round(xps,3)))
        print('weights='+str(np.round(xis,3)))
        print('probs='+str(np.round(p,3)))
        print('weights*probs='+str(np.round(xis*p,3)))
        print('adjusted alpha='+str(np.round(yps,3)))
        print('interpolated value (undiscounted)='+str(np.round(V_interp/y,3)))

    # maximize distorted expectation which is positive for costs.
    return(distorted_exp)
