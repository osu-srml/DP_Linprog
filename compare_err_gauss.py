import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import genm, rqm, erm, asy_optm
# from dp_compression.mechanisms import MVUMechanism


def quant_err(c, m, eps, mechanism, mu, sigma):
    """
    Output the estimated error of the mechanism
    """
    err_arr = []
    xs = np.random.normal(mu, sigma, 1000)
    for xid in range(len(xs)):
        if xs[xid] < -c:
            xs[xid] = -c
        elif xs[xid] > c:
            xs[xid] = c

    if mechanism == "RQM":
        delta, q = rqm.opt_par(c, m, eps)
        assert delta != None and q != None
        bins = np.linspace(-c-delta, c+delta, 4)
        for x in xs:
            sample_pr = rqm.calc_sample_pr(bins, x, q)
            bins_pr = genm.calc_bins_pr(bins, sample_pr, x)
            err = genm.calc_expected_err(bins, bins_pr, x)
            err_arr.append(err)

    # elif mechanism == "MVU":
    #     mechanism = MVUMechanism(budget=2, epsilon=eps, input_bits=2, method="trust-region")
    #     ys = (mechanism.decode(mechanism.privatize((xs + 1) / 2)))
    #     for yid in range(len(ys)):
    #         err_arr.append(abs(xs[yid]-ys[yid]) * 2)
    
    elif mechanism == "OPTM":
        bins, q_arr = asy_optm.opt_par(c, m, eps, mu, sigma)
        for x in xs:
            j = -1
            for id in range(m-1):
                if bins[id] <= x < bins[id+1]:
                    j = id
            assert j != -1, f"Wrong j for NUOPTM"
            
            bins_pr = genm.calc_bins_pr(bins, q_arr[j], x)
            err = genm.calc_expected_err(bins, bins_pr, x)
            err_arr.append(err)
    
    return np.mean(err_arr)


if __name__ == '__main__':

    c = 1
    m = 4
    eps = 1
    mu = 0.5
    for mechanism in ["OPTM", "RQM"]:
        for sigma in [0.1, 0.2, 0.3]:
            err = quant_err(c, m, eps, mechanism, mu, sigma)
            print("{}: sigma={:.1f}, err={:.3f}".format(mechanism, sigma, err))

    
        