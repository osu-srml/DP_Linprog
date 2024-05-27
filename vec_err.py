import numpy as np
# from dp_compression.mechanisms import MVUMechanism
import erm, optm, rqm, genm
import matplotlib.pyplot as plt
from random import random
import argparse
import matplotlib


font = {'size': 15}
matplotlib.rc('font', **font)


def l2_rand_vec(low, high, dim):
    u = np.random.normal(0, 1, dim)
    norm = np.sum(u**2) ** 0.5
    r = random() ** (1.0/dim)
    res = r * u / norm
    return res * high


def l1_rand_vec(low, high, dim):
    return np.random.uniform(low, high, dim)


def calc_vec_err(c, m, dim, mechanism, eps, num_trials, type):
    """
    Calculate the mean and variance of the error of a specific mechanism instance with vector inputs
    """
    err_arr = []

    if mechanism == 'RQM':
        rqm_delta, rqm_q = rqm.opt_par(c, m, eps)
        rqm_bins = np.linspace(-c-rqm_delta, c+rqm_delta, 4)
        for _ in range(num_trials):
            if type == "l1":
                vec = l1_rand_vec(-c, c, dim)
            if type == "l2":
                vec = l2_rand_vec(-c, c, dim)
            pvec = []  # record privatized vectors
            for id in range(len(vec)):
                rqm_sample_pr = rqm.calc_sample_pr(rqm_bins, vec[id], rqm_q)
                rqm_bins_pr = genm.calc_bins_pr(rqm_bins, rqm_sample_pr, vec[id])
                pvec.append(np.random.choice(rqm_bins, p=rqm_bins_pr))
            err_arr.append(np.linalg.norm(vec - pvec))

    elif mechanism == 'ERM':
        erm_bins, erm_g = erm.opt_par(c, m, eps)
        for _ in range(num_trials):
            if type == "l1":
                vec = l1_rand_vec(-c, c, dim)
            if type == "l2":
                vec = l2_rand_vec(-c, c, dim)
            pvec = []
            for id in range(len(vec)):
                erm_sample_pr = erm.calc_sample_pr(erm_bins, vec[id], erm_g)
                erm_bins_pr = genm.calc_bins_pr(erm_bins, erm_sample_pr, vec[id])
                pvec.append(np.random.choice(erm_bins, p=erm_bins_pr))
            err_arr.append(np.linalg.norm(vec - pvec))

    elif mechanism == 'OPTM':
        # optm_bins, optm_q_arr = optm.opt_par(c, m, eps)

        # use customized bin values
        optm_bins = [-3, -0.5, 0.5, 3]
        _, optm_q_arr = optm.opt_par(c, m, eps, bins=optm_bins)

        for _ in range(num_trials):
            if type == "l1":
                vec = l1_rand_vec(-c, c, dim)
            if type == "l2":
                vec = l2_rand_vec(-c, c, dim)
            
            pvec = []
            for id in range(len(vec)):
                nuj = -1
                for bid in range(len(optm_bins)):
                    if optm_bins[bid] <= vec[id] < optm_bins[bid+1]:
                        nuj = bid
                assert nuj != -1, f"Wrong j for OPTM"
            
                optm_bins_pr = genm.calc_bins_pr(optm_bins, optm_q_arr[nuj], vec[id])
                pvec.append(np.random.choice(optm_bins, p=optm_bins_pr))
            err_arr.append(np.linalg.norm(vec - pvec))
    
    return np.mean(err_arr), np.var(err_arr)


# def test_mvu(dim, eps, num_trials, type):

#     if type == "l1":
#         x = l1_rand_vec(-1, 1, num_trials*dim)
#     if type == "l2":
#         x = l2_rand_vec(-1, 1, num_trials*dim)
#     mechanism = MVUMechanism(budget=2, epsilon=eps, input_bits=2, method="trust-region")
#     y = (mechanism.decode(mechanism.privatize((x + 1) / 2)))
#     mvu_arr = []
#     for id in range(num_trials):
#         vec = x[id*dim:(id+1)*dim]
#         pvec = y[id*dim:(id+1)*dim]
#         mvu_arr.append(np.linalg.norm(vec - pvec) * 2)
#     return np.mean(mvu_arr), np.var(mvu_arr)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=int, default=1, help='clipping range of input')
    parser.add_argument('--type', type=str, default="l1", help='l1 or l2')
    parser.add_argument('--dim', type=int, default=10, help='dimension of random vectors')
    args = parser.parse_args()

    c = args.c
    m = 4
    type = args.type
    dim = args.dim
    num_trials = 10000
    eps_arr = np.array([1, 1.5, 2, 2.5, 3])

    fig = plt.figure(figsize=(6,6))

    for mechanism in ["OPTM", "RQM"]:
        mean_arr = []
        var_arr = []
        for eps in eps_arr:
            mean, var = calc_vec_err(c, m, dim, mechanism, eps, num_trials, type)
            mean_arr.append(mean)
            var_arr.append(var)
        plt.errorbar(eps_arr, mean_arr, yerr=var, label=mechanism, linewidth=2, markersize=15)

    plt.xlabel(rf"$\epsilon$")
    plt.ylabel("Average Error")
    plt.legend()
    plt.grid(ls = "--")
    plt.show()