import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
import numpy as np
import genm


def exp_term(eps, score, sen):
    return math.exp(eps * (- score) / (2 * sen))


def calc_sample_pr(bins, x, eps):
    """
    Calculate the selection distribution
    """
    j = -1
    for bid in range(len(bins)-1):
        if bins[bid] <= x < bins[bid+1]:
            j = bid
    
    larr = []
    if j == 0:
        larr = [1]
    else:
        for i in range(0, j+1):
            larr.append(exp_term(eps, bins[j]-bins[i], bins[j]-bins[0]))
            
    rarr = []
    if j + 1 == len(bins) - 1:
        rarr = [1]
    else:
        for i in range(j+1, len(bins)):
            rarr.append(exp_term(eps, bins[i]-bins[j+1], bins[-1]-bins[j+1]))

    lsum = sum(larr)
    rsum = sum(rarr)
    arr = larr + rarr

    pr = []
    for k in range(0, j+1):
        pr.append(arr[k] / lsum)
    for k in range(j+1, len(bins)):
        pr.append(arr[k] / rsum)

    return pr


def pri_loss(bins, c, erm_g):
    """
    Calculate the standard differential privacy loss given parameters
    """
    m = len(bins)
    j_min = -1
    j_max = -1
    for bid in range(len(bins)-1): 
        if bins[bid] <= -c < bins[bid+1]:
            j_min = bid
        if bins[bid] <= c < bins[bid+1]:
            j_max = bid

    assert j_min != -1 and j_max != -1, f"Wrong j"

    bins_loss = []
    for bid in range(m): 
        pr_arr = []

        for bid2 in range(j_min+1, j_max+1): 
            sample_pr = calc_sample_pr(bins, bins[bid2], erm_g)
            bins_pr = genm.calc_bins_pr(bins, sample_pr, bins[bid2])
            pr_arr.append(bins_pr[bid])
        
        sample_pr = calc_sample_pr(bins, -c, erm_g)
        bins_pr = genm.calc_bins_pr(bins, sample_pr, -c)
        pr_arr.append(bins_pr[bid])

        sample_pr = calc_sample_pr(bins, c, erm_g)
        bins_pr = genm.calc_bins_pr(bins, sample_pr, c)
        pr_arr.append(bins_pr[bid])

        for k in range(max(j_min+1, bid+1), j_max+1):  
            sample_pr = calc_sample_pr(bins, bins[k-1], erm_g)  
            pr = 0
            for r in range(k+1, len(bins)):  
                pr += sample_pr[r] * ((bins[r] - bins[k]) / (bins[r] - bins[bid]))  
            pr = pr * sample_pr[bid]
            pr_arr.append(pr)
        
        for k in range(j_min+1, min(j_max, bid)+1):
            sample_pr = calc_sample_pr(bins, bins[k-1], erm_g)  
            pr = 0
            for r in range(0, k):  
                pr += sample_pr[r] * ((bins[k] - bins[r]) / (bins[bid] - bins[r]))  
            pr = pr * sample_pr[bid]
            pr_arr.append(pr)

        bins_loss.append(max(pr_arr) / min(pr_arr))

    return max(bins_loss)


def opt_par(c, m, eps):
    """
    Using grid search to find the optimal bin values over uniform distribution
    """
    dir = "parameters/ERM/c{}_m{}_eps{:.1f}/".format(c, m, eps)
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    else:
        bins = np.load(dir + "opt_bins.npy")
        gamma = np.load(dir + "opt_g.npy")
        return bins, gamma

    min_err = 100
    min_delta = -1
    min_bins = []
    min_g = -1

    for r in np.arange(1, 5.1, 0.1):
        delta = c * r
        for dis in np.arange(0.1, 1.1, 0.1):
            bins = np.array([-c-delta, -dis, dis, c+delta])
            g = opt_g(c, bins, eps)
            if g != None:
                erm_err_arr = []
                x_iter = np.arange(-c, c+0.01, 0.04)
                for x in x_iter:
                    erm_bins = bins
                    erm_sample_pr = calc_sample_pr(erm_bins, x, g)
                    erm_bins_pr = genm.calc_bins_pr(erm_bins, erm_sample_pr, x)
                    erm_err = genm.calc_expected_err(erm_bins, erm_bins_pr, x)
                    erm_err_arr.append(erm_err)

                err = np.mean(erm_err_arr)
                if err < min_err:
                    min_delta = delta
                    min_bins = bins
                    min_err = err
                    min_g = g

    assert min_delta != -1 and min_g != -1

    np.save(dir + "opt_delta.npy", min_delta)
    np.save(dir + "opt_bins.npy", np.array(min_bins))
    np.save(dir + "opt_g.npy", min_g)

    return min_bins, min_g


def quant_res(bins, erm_eps, x):
    erm_sample_pr = calc_sample_pr(bins, x, erm_eps)
    erm_bins_pr = genm.calc_bins_pr(bins, erm_sample_pr, x)
    return np.random.choice(bins, p=erm_bins_pr)


def opt_g(c, bins, eps):
    """
    Finding the optimal gamma given the value of bins, privacy parameter eps, and c
    """
    for g in np.arange(0.001, eps, 0.001):
    # for g in np.arange(0.001, 2, 0.001):
        if pri_loss(bins, c, g) <= math.exp(eps) < pri_loss(bins, c, g+0.001):
            return g


def asy_opt_par(c, m, eps, mu, sigma):
    """
    Using grid search to find the optimal bin values over truncated Gaussian distribution
    """
    dir = "parameters/ASY_ERM/c{}_m{}_eps{:.1f}_mu{:1f}_sigma{:1f}/".format(c, m, eps, mu, sigma)
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    else:
        bins = np.load(dir + "opt_bins.npy")
        gamma = np.load(dir + "opt_g.npy")
        return bins, gamma

    min_err = 100
    min_bins = []
    min_g = -1

    bin_loc = np.linspace(-c, c, 17).tolist()[1:-1]

    assert m == 4, f"m != 4"

    for r in np.arange(4, 6.1, 0.1):
        for id1 in range(0, len(bin_loc)-1):
            for id2 in range(id1+1, len(bin_loc)): 
                delta = c * r
                bins = [-c-delta, bin_loc[id1], bin_loc[id2], c+delta]
                g = opt_g(c, bins, eps)
                if g != None:
                    erm_err_arr = []
                    x_iter = np.random.normal(mu, sigma, 1000)
                    for x in x_iter:
                        if x < -c:
                            x = -c
                        elif x > c:
                            x = c
                        
                        erm_sample_pr = calc_sample_pr(bins, x, g)
                        erm_bins_pr = genm.calc_bins_pr(bins, erm_sample_pr, x)
                        erm_err = genm.calc_expected_err(bins, erm_bins_pr, x)
                        erm_err_arr.append(erm_err)

                    err = np.mean(erm_err_arr)
                    if err < min_err:
                        min_bins = bins
                        min_err = err
                        min_g = g
            
    assert min_g != -1

    np.save(dir + "opt_bins.npy", np.array(min_bins))
    np.save(dir + "opt_g.npy", min_g)

    return min_bins, min_g


# if __name__ == '__main__':

    # print(asy_opt_par(1, 4, 1, 0.5, 0.1))
