import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rqm
import genm
import erm
import optm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=int, default=1, help='clipping range of input')
parser.add_argument('--eps', type=float, default=1, help='privacy budget parameter epsilon')

args = parser.parse_args()

font = {'size': 15}
matplotlib.rc('font', **font)


c = args.c
m = 4
pri_eps = args.eps
pri = math.exp(pri_eps)

x_iter = np.arange(-c, c+0.01, 0.04)


erm_err_arr = []
rqm_err_arr = []
optm_err_arr = []

rqm_delta, rqm_q = rqm.opt_par(c, m, pri_eps)
rqm_bins = np.linspace(-c-rqm_delta, c+rqm_delta, 4)

erm_bins, erm_g = erm.opt_par(c, m, pri_eps)

optm_bins, optm_q_arr = optm.opt_par(c, m, pri_eps)

for x in x_iter:
    
    erm_sample_pr = erm.calc_sample_pr(erm_bins, x, erm_g)
    erm_bins_pr = genm.calc_bins_pr(erm_bins, erm_sample_pr, x)
    erm_err = genm.calc_expected_err(erm_bins, erm_bins_pr, x)
    erm_err_arr.append(erm_err)

    rqm_sample_pr = rqm.calc_sample_pr(rqm_bins, x, rqm_q)
    rqm_bins_pr = genm.calc_bins_pr(rqm_bins, rqm_sample_pr, x)
    rqm_err = genm.calc_expected_err(rqm_bins, rqm_bins_pr, x)
    rqm_err_arr.append(rqm_err)

    optm_j = -1
    for id in range(m-1):
        if optm_bins[id] <= x < optm_bins[id+1]:
            optm_j = id
    assert optm_j != -1, f"Wrong j for NUOPTM"
    
    optm_bins_pr = genm.calc_bins_pr(optm_bins, optm_q_arr[optm_j], x)
    optm_err = genm.calc_expected_err(optm_bins, optm_bins_pr, x)
    optm_err_arr.append(optm_err)


plt.plot(np.array(x_iter), np.array(optm_err_arr), label=rf"OPTM", marker="*", color="red")
plt.plot(np.array(x_iter), np.array(erm_err_arr), label=rf"ERM", marker="s", color="green")
plt.plot(np.array(x_iter), np.array(rqm_err_arr), label=rf"RQM", marker="^", color="orange")
# plt.plot(np.array(mvu_x), np.array(mvu_err), label=rf"MVU", marker="o", color="blue")

plt.title(rf"$m$ = %d, $c$ = %.2f, $\epsilon$ = %.2f" % (m, c, pri_eps))
plt.ylim(0, 3)
plt.xlabel("Input")
plt.ylabel("Mean Absolute Error")
plt.grid(ls = "--")
plt.legend(loc="lower center", ncol=2)
plt.show()