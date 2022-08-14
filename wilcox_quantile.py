'''Compare quantiles of 2 samples using Wilcox et al. 2013 method

Reference:
     Rand R. Wilcox , David M. Erceg-Hurn , Florence Clark & Michael Carlson ,
Journal of Statistical Computation and Simulation (2013): Comparing two independent groups
via the lower and upper quantiles, Journal of Statistical Computation and Simulation, DOI:
10.1080/00949655.2012.754026

Reference R code:
    https://github.com/nicebread/WRS

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-08-14 20:12:21.
'''

import numpy as np
from scipy.stats import beta


def getMissingMask(slab):
    '''Get a bindary denoting missing (masked or nan).

    Args:
        slab (ndarray): ndarray, possibly contains masked values or nans.
    Returns:
        mask (ndarray): nd bindary, 1s for missing, 0s otherwise.
    '''

    nan_mask=np.where(np.isnan(slab),1,0)

    if not hasattr(slab,'mask'):
        mask_mask=np.zeros(slab.shape)
    else:
        if slab.mask.size==1 and slab.mask==False:
            mask_mask=np.zeros(slab.shape)
        else:
            mask_mask=np.where(slab.mask,1,0)

    mask=np.where(mask_mask+nan_mask>0,1,0)

    return mask

def preprocess(x):
    '''Preprocessing sample array by flattening and removing missings'''
    x = x.flatten()
    mask_x = getMissingMask(x)
    xval = x[mask_x==0]
    nx = len(xval)
    return xval, nx

def get_hd_weights(n, quantile):
    '''Compute weights for quantile estimate using Harrel-Davis method

    Args:
        n (int): number of samples.
        quantile (float): quantile in [0, 1].
    Returns:
        weights (1darray): 1d array, weights.
    '''
    a = quantile * (n + 1)
    b = (1 - quantile) * (n + 1)
    pos = np.arange(n)
    weights = beta.cdf((pos + 1) / n, a, b) - beta.cdf(pos / n, a, b)
    return weights


def bootstrap_est(x, y, n_boot, quantile, alpha=0.05, seed=30):
    '''Compare quantiles from 2 groups using bootstrap resampling

    Args:
        x (ndarray): 1d array, data from group 1.
        y (ndarray): 1d array, data from group 2.
        n_boot (int): number of random bootstrap resampling.
        quantile (float): quantile to compare.
    Keyward Args:
        alpha (float): confidence level.
        seed (int or None): seed to use for random sampling.
    Returns:
        ci_lower (float): lower bound of confidence interval.
        ci_upper (float): upper bound of confidence interval.
        pvalue (float): p-value of H1.
        se (float): standard error of difference estimate.
    '''

    nx = len(x)
    ny = len(y)

    if seed is not None:
        np.random.seed(seed)

    # compute estimates for x
    # random samples with replacement
    #datax = np.vstack([np.random.choice(x, size=len(x), replace=True) for _ in range(n_boot)])
    xidx = np.random.randint(0, nx, size=nx*n_boot)
    datax = np.vstack([x[xidx[i*nx:(i+1)*nx]] for i in range(n_boot)])
    # estimate quantile using Harrell-Davis
    estx = np.array([estimate_quantile(xii, quantile) for xii in datax])
    del datax

    # compute estimates for y
    #datay = np.vstack([np.random.choice(y, size=len(y), replace=True) for _ in range(n_boot)])
    yidx = np.random.randint(0, ny, size=ny*n_boot)
    datay = np.vstack([y[yidx[i*ny:(i+1)*ny]] for i in range(n_boot)])
    esty = np.array([estimate_quantile(yii, quantile) for yii in datay])
    del datay

    diff = np.sort(estx - esty)
    lower = int((alpha * n_boot) // 2)
    upper = n_boot - lower
    ci_lower = diff[lower+1]
    ci_upper = diff[upper]
    pp = (np.sum(diff<0) + 0.5 * np.sum(diff==0)) / n_boot
    pvalue = 2 * min(pp, 1 - pp)
    se = np.std(diff)

    return ci_lower, ci_upper, pvalue, se

def bootstrap_est_vec(x, y, n_boot, quantile, alpha=0.05, seed=30):
    '''Compare quantiles from 2 groups using bootstrap resampling, vectorized version

    Args:
        x (ndarray): 1d array, data from group 1.
        y (ndarray): 1d array, data from group 2.
        n_boot (int): number of random bootstrap resampling.
        quantile (float): quantile to compare.
    Keyward Args:
        alpha (float): confidence level.
        seed (int or None): seed to use for random sampling.
    Returns:
        estx (float): estimated quantile for x.
        esty (float): estimated quantile for y.
        ci_lower (float): lower bound of confidence interval.
        ci_upper (float): upper bound of confidence interval.
        pvalue (float): p-value of H1.
        se (float): standard error of difference estimate.
    '''

    # Get weights for quantile estimates
    nx = len(x)
    ny = len(y)
    weights_x = get_hd_weights(nx, quantile)[None, :]
    if ny == nx:
        weights_y = weights_x
    else:
        weights_y = get_hd_weights(ny, quantile)[None, :]

    if seed is not None:
        np.random.seed(seed)

    # compute estimates for x
    # random samples with replacement
    # method 1: random.choice
    #datax = np.vstack([np.random.choice(x, size=len(x), replace=True) for _ in range(n_boot)])

    # method 2: random.randint
    #datax = np.vstack([x[np.random.randint(0, nx, size=nx)] for _ in range(n_boot)])

    # method 3: get all random.randint and slice, seem to be fastest
    xidx = np.random.randint(0, nx, size=nx*n_boot)
    datax = np.vstack([x[xidx[i*nx:(i+1)*nx]] for i in range(n_boot)])

    # append the original x
    datax = np.vstack([[x], datax])
    # compute quantile estimates
    datax = np.sort(datax, axis=1)
    estx_all = (datax * weights_x).sum(axis=1)
    estx = estx_all[0]
    del datax

    # compute estimates for y
    #datay = np.vstack([np.random.choice(y, size=len(y), replace=True) for _ in range(n_boot)])
    #datay = np.vstack([y[np.random.randint(0, ny, size=ny)] for _ in range(n_boot)])
    yidx = np.random.randint(0, ny, size=ny*n_boot)
    datay = np.vstack([y[yidx[i*ny:(i+1)*ny]] for i in range(n_boot)])
    datay = np.vstack([[y], datay])
    datay = np.sort(datay, axis=1)
    esty_all = (datay * weights_y).sum(axis=1)
    esty = esty_all[0]
    del datay

    # compute quantile difference significance
    diff = np.sort(estx_all[1:] - esty_all[1:])
    lower = int((alpha * n_boot) // 2)
    upper = n_boot - lower
    ci_lower = diff[lower+1]
    ci_upper = diff[upper]
    pp = (np.sum(diff<0) + 0.5 * np.sum(diff==0)) / n_boot
    pvalue = 2 * min(pp, 1 - pp)
    se = np.std(diff)

    return estx, esty, ci_lower, ci_upper, pvalue, se

def estimate_quantile(x, quantile):
    '''Estimate quantile of array using Harrell-Davis method

    Args:
        x (1darray): 1d input array, without missing values.
        quantile (float): quantile in [0, 1].
    Returns:
        res (float): estimated quantile score of <x> at <quantile>.
    '''
    wi = get_hd_weights(len(x), quantile)
    res = np.sort(x).dot(wi)
    return res


def compare_quantiles(x, y, quantiles=None, n_boot=2000, alpha=0.05, adj_ci=True, seed=30):
    '''Compare quantiles of 2 samples using Wilcox method

    Args:
        x (ndarray): 1d array, data from group 1.
        y (ndarray): 1d array, data from group 2.
    Keyward Args:
        quantiles (1d array or None): quantiles to compare. If None, use the default
            quantiles [0.1, 0.25, 0.5, 0.75, 0.9].
        n_boot (int): number of random bootstrap resampling.
        alpha (float): confidence level.
        adj_ci (bool): whether to adjust confidence intervals using a 2nd iteration.
        seed (int or None): seed to use for random sampling.
    Returns:
        output (dict): computed statistics, with these keys:
            'quantiles': quantiles comparison are made,
            'nx': number of valid samples in <x>,
            'ny': number of valid samples in <y>,
            'estx': 1darray, estimated quantile scores of <x> corresponding to <quantiles>,
            'esty': 1darray, estimated quantile scores of <y> corresponding to <quantiles>,
            'estx-esty': estx - esty differences,
            'ci_lower': 1darray, lower bound of confidence interval at <alpha> level,
            'ci_upper': 1darray, upper bound of confidence interval at <alpha> level,
            'p_crtic': 1darray, critical p value at <alpha> level,
            'pvalue': 1darray, computed p values,
            'is_sig': 1darray, boolean array of whether pvalue > p_critic,
    '''

    #--------------------Preprocess--------------------
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    else:
        quantiles = np.asarray(quantiles).flatten()
    nq = len(quantiles)
    output = np.zeros([nq, 9]) # store results

    xval, nx = preprocess(x)
    yval, ny = preprocess(y)

    #--------------Loop through quantiles--------------
    for ii, qii in enumerate(quantiles):
        output[ii, 0] = qii
        output[ii, 1] = estimate_quantile(xval, qii)
        output[ii, 2] = estimate_quantile(yval, qii)
        ci_lower, ci_upper, pvalue, se = bootstrap_est(xval, yval, n_boot, qii, alpha, seed)
        output[ii, 4] = ci_lower
        output[ii, 5] = ci_upper
        output[ii, 7] = pvalue

    output[:, 3] = output[:, 1] - output[:, 2]

    tmp = np.argsort(output[:, 7])[::-1]
    zvec = alpha / np.arange(1, nq+1)
    output[tmp, 6] = zvec

    #-------------Adjust ci by re-compute-------------
    if adj_ci:
        for ii, qii in enumerate(quantiles):
            ci_lower, ci_upper, pvalue, se = bootstrap_est(
                    xval, yval, n_boot, qii, output[ii, 6], seed)
            output[ii, 4] = ci_lower
            output[ii, 5] = ci_upper
            output[ii, 7] = pvalue

    output[:, 8] = output[:, 7] <= output[:, 6]

    #------------------Prepare output------------------
    output = {
            'quantiles': quantiles,
            'nx': nx,
            'ny': ny,
            'estx': output[:,1],
            'esty': output[:,2],
            'estx-esty': output[:,3],
            'ci_lower': output[:,4],
            'ci_upper': output[:,5],
            'p_crtic': output[:,6],
            'pvalue': output[:,7],
            'is_sig': output[:,8],
            }

    return output


def compare_quantiles_vec(x, y, quantiles=None, n_boot=2000, alpha=0.05, adj_ci=True, seed=30):
    '''Compare quantiles of 2 samples using Wilcox method, vectorized version

    Args:
        x (ndarray): 1d array, data from group 1.
        y (ndarray): 1d array, data from group 2.
    Keyward Args:
        quantiles (1d array or None): quantiles to compare. If None, use the default
            quantiles [0.1, 0.25, 0.5, 0.75, 0.9].
        n_boot (int): number of random bootstrap resampling.
        alpha (float): confidence level.
        adj_ci (bool): whether to adjust confidence intervals using a 2nd iteration.
        seed (int or None): seed to use for random sampling.
    Returns:
        output (dict): computed statistics, with these keys:
            'quantiles': quantiles comparison are made,
            'nx': number of valid samples in <x>,
            'ny': number of valid samples in <y>,
            'estx': 1darray, estimated quantile scores of <x> corresponding to <quantiles>,
            'esty': 1darray, estimated quantile scores of <y> corresponding to <quantiles>,
            'estx-esty': estx - esty differences,
            'ci_lower': 1darray, lower bound of confidence interval at <alpha> level,
            'ci_upper': 1darray, upper bound of confidence interval at <alpha> level,
            'p_crtic': 1darray, critical p value at <alpha> level,
            'pvalue': 1darray, computed p values,
            'is_sig': 1darray, boolean array of whether pvalue > p_critic,
    '''

    #--------------------Preprocess--------------------
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    else:
        quantiles = np.asarray(quantiles).flatten()
    nq = len(quantiles)
    output = np.zeros([nq, 9]) # store results

    xval, nx = preprocess(x)
    yval, ny = preprocess(y)

    #--------------Loop through quantiles--------------
    for ii, qii in enumerate(quantiles):
        output[ii, 0] = qii
        estxii, estyii, ci_lower, ci_upper, pvalue, se = bootstrap_est_vec(
                xval, yval, n_boot, qii, alpha, seed)
        output[ii, 1] = estxii
        output[ii, 2] = estyii
        output[ii, 4] = ci_lower
        output[ii, 5] = ci_upper
        output[ii, 7] = pvalue

    output[:, 3] = output[:, 1] - output[:, 2]

    tmp = np.argsort(output[:, 7])[::-1]
    zvec = alpha / np.arange(1, nq+1)
    output[tmp, 6] = zvec

    #-------------Adjust ci by re-compute-------------
    if adj_ci:

        for ii, qii in enumerate(quantiles):
            estxii, estyii, ci_lower, ci_upper, pvalue, se = bootstrap_est_vec(
                    xval, yval, n_boot, qii, output[ii, 6], seed)
            output[ii, 4] = ci_lower
            output[ii, 5] = ci_upper
            output[ii, 7] = pvalue

    output[:, 8] = output[:, 7] <= output[:, 6]
    output = {
            'quantiles': quantiles,
            'nx': nx,
            'ny': ny,
            'estx': output[:,1],
            'esty': output[:,2],
            'estx-esty': output[:,3],
            'ci_lower': output[:,4],
            'ci_upper': output[:,5],
            'p_crtic': output[:,6],
            'pvalue': output[:,7],
            'is_sig': output[:,8],
            }

    return output

