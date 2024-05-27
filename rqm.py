import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
import numpy as np
import genm


def calc_sample_pr(bins, x, q):
    """
    Calculate the selection distribution
    """
    j = -1
    for bid in range(len(bins)-1):
        if bins[bid] <= x <= bins[bid+1]:
            j = bid

    pr = []
    pr.append((1 - q) ** j)
    for i in range(1, j+1):
        pr.append(q * ((1 - q) ** (j - i)))
    for i in range(j+1, len(bins)-1):
        pr.append(q * ((1 - q) ** (i - j - 1)))
    pr.append((1 - q) ** (len(bins) - j - 2))
    return pr


def pri_loss(bins, c, q):
    """
    Calculate the standard differential privacy loss given parameters
    """
    m = len(bins)
    j_min = -1
    j_max = -1
    for bid in range(len(bins)-1): # check
        if bins[bid] <= -c < bins[bid+1]:
            j_min = bid
        if bins[bid] <= c < bins[bid+1]:
            j_max = bid

    assert j_min != -1 and j_max != -1, f"Wrong j"

    bins_loss = []
    for bid in range(j_max+1): 
        max_pr_arr = []
        min_pr_arr = []

        # find the max probability
        if bins[bid] <= -c: 
            sample_pr = calc_sample_pr(bins, -c, q)
            bins_pr = genm.calc_bins_pr(bins, sample_pr, -c)
            max_pr_arr.append(bins_pr[bid])
            for bid2 in range(j_min+1, j_max+1): 
                sample_pr = calc_sample_pr(bins, bins[bid2], q)
                bins_pr = genm.calc_bins_pr(bins, sample_pr, bins[bid2])
                max_pr_arr.append(bins_pr[bid])
        else:
            for bid2 in range(bid, j_max+1): 
                sample_pr = calc_sample_pr(bins, bins[bid2], q)
                bins_pr = genm.calc_bins_pr(bins, sample_pr, bins[bid2])
                max_pr_arr.append(bins_pr[bid])
            
            sample_pr = calc_sample_pr(bins, bins[m-bid-1], q)
            bins_pr = genm.calc_bins_pr(bins, sample_pr, bins[m-bid-1])
            max_pr_arr.append(bins_pr[m-bid-1])

        # find the minimal probability
        for k in range(max(j_min+1, bid+1), j_max+1):  
            sample_pr = calc_sample_pr(bins, bins[k-1], q)  
            pr = 0
            for r in range(k+1, len(bins)):
                pr += sample_pr[r] * ((bins[r] - bins[k]) / (bins[r] - bins[bid]))
            pr = pr * sample_pr[bid]
            min_pr_arr.append(pr)
        
        sample_pr = calc_sample_pr(bins, c, q)
        pr = 0
        for r in range(j_max+1, len(bins)):
            pr += sample_pr[r] * ((bins[r] - c) / (bins[r] - bins[bid]))
        pr = pr * sample_pr[bid]
        min_pr_arr.append(pr)

        bins_loss.append(max(max_pr_arr) / min(min_pr_arr))

    return max(bins_loss)


def opt_par(c, m, eps):
    """
    Find the value of delta and q of inducing the optimal performance
    """

    dir = "parameters/RQM/c{}_m{}_eps{:.1f}/".format(c, m, eps)
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    else:
        delta = np.load(dir + "opt_delta.npy")
        q = np.load(dir + "opt_q.npy")
        return delta, q

    min_delta = -1
    min_q = -1
    min_err = 100

    for r in np.arange(1, 2, 0.1):
        delta = c * r
        bins = np.linspace(-c-delta, c+delta, m)
        pri = math.exp(eps)

        for q in np.arange(0.001, 1, 0.001):
            if pri_loss(bins, c, q) < pri:
                err_arr = []
                x_iter = np.arange(-c, c+0.01, 0.04)
                for x in x_iter:
                    sample_pr = calc_sample_pr(bins, x, q)
                    bins_pr = genm.calc_bins_pr(bins, sample_pr, x)
                    err = genm.calc_expected_err(bins, bins_pr, x)
                    err_arr.append(err)
                err = np.mean(err_arr)

                if err < min_err:
                    min_delta = delta
                    min_err = err
                    min_q = q

    assert min_delta != -1 and min_q != -1

    np.save(dir + "opt_delta.npy", min_delta)
    np.save(dir + "opt_q.npy", min_q)

    return min_delta, min_q


# if __name__ == '__main__':
#     print(opt_par(1, 4, 1))
