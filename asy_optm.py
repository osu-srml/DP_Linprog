import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
import numpy as np
from pulp import *


def calc_dens(c, bins, mu, sigma):
    """
    Estimate the density function of the input distribution
    """
    dens = []
    num = 0

    for _ in range(len(bins)-1):
        dens.append(0)

    x_iter = np.random.normal(mu, sigma, 1000)
    for x in x_iter:
        if x < -c:
            x = -c
        elif x > c:
            x = c
        num += 1
        for id in range(len(bins)-1):
            if bins[id] <= x < bins[id+1]:
                dens[id] += 1
    
    for id in range(len(bins)-1):
        dens[id] = dens[id] / num    

    return dens


def opt_par(c, m, eps, mu, sigma):
    """
    Using grid search to find the optimal bin values given samples of inputs
    """

    dir = "parameters/ASY_OPTM/c{}_m{}_eps{:.1f}_mu{:.1f}_sigma{:.1f}/".format(c, m, eps, mu, sigma)
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    else:
        bins = np.load(dir + "opt_bins.npy")
        q_arr = np.load(dir + "opt_q_arr.npy")
        return bins, q_arr

    bin_loc = np.linspace(-c, c, 6).tolist()[1:-1]
    min_q_arr = []
    min_bins = []
    min = 100
    pri = math.exp(eps)

    assert m == 4, f"m != 4"

    for r in np.arange(1, 3.1, 1):
        delta = c * r
        for id1 in range(0, len(bin_loc)-1):
            for id2 in range(id1+1, len(bin_loc)):

                bins = [-c-delta, bin_loc[id1], bin_loc[id2], c+delta]
                dens = calc_dens(c, bins, mu, sigma)
                
                a1 = bins[0]
                a2 = bins[1]
                a3 = bins[2]
                a4 = bins[3]

                for ql21lb in np.arange(0.05, 1, 0.05):
                    ql21ub = ql21lb + 0.01
                    for qr21lb in np.arange(0.05, 1, 0.05):
                        qr21ub = qr21lb + 0.01

                        model = pulp.LpProblem('linear_programming', LpMinimize)

                        solver = pulp.PULP_CBC_CMD(msg=0)

                        ql21 = LpVariable('ql21', lowBound = ql21lb, upBound = ql21ub, cat = 'continuous')
                        ql22 = LpVariable('ql22', lowBound = 1e-5, cat = 'continuous')

                        ql31 = LpVariable('ql31', lowBound = 1e-5, cat = 'continuous')
                        ql32 = LpVariable('ql32', lowBound = 1e-5, cat = 'continuous')
                        ql33 = LpVariable('ql33', lowBound = 1e-5, cat = 'continuous')

                        qr21 = LpVariable('qr21', lowBound = qr21lb, upBound = qr21ub, cat = 'continuous')
                        qr22 = LpVariable('qr22', lowBound = 1e-5, cat = 'continuous')

                        qr31 = LpVariable('qr31', lowBound = 1e-5, cat = 'continuous')
                        qr32 = LpVariable('qr32', lowBound = 1e-5, cat = 'continuous')
                        qr33 = LpVariable('qr33', lowBound = 1e-5, cat = 'continuous')


                        model += dens[0] * ((a2 - a1) + qr32 * (a3 - a2) + qr31 * (a4 - a2)) \
                            + dens[1] * (ql21 * (a2 - a1) + (a3 - a2) + qr21 * (a4 - a3)) \
                            + dens[2] * (ql31 * (a3 - a1) + ql32 * (a3 - a2) + (a4 - a3))
                        
                        lpl11 = qr33 * (a2 + c) / (a2 - a1) + qr32 * (a3 + c) / (a3 - a1) + qr31 * (a4 + c) / (a4 - a1)
                        lpl21 = 1 * (qr22 * (a3 - a2) / (a3 - a1) + qr21 * (a4 - a2) / (a4 - a1))
                        lpl31 = ql31 * (a4 - a3) / (a4 - a1)
                        lpl22 = ql22
                        lpl32 = ql32 * (a4 - a3) / (a4 - a2)
                        lpl33 = ql33

                        lpr44 = ql33 * (c - a3) / (a4 - a3) + ql32 * (c - a2) / (a4 - a2) + ql31 * (c - a1) / (a4 - a1)
                        lpr34 = 1 * (ql21 * (a3 - a1) / (a4 - a1) + ql22 * (a3 - a2) / (a4 - a2))
                        lpr24 = qr31 * (a4 - a2) / (a4 - a1)
                        lpr33 = qr22
                        lpr23 = qr32 * (a2 - a1) / (a3 - a1)
                        lpr22 = qr33

                        spl21 = qr32 * (a3 - a2) / (a3 - a1) + qr31 * (a4 - a2) / (a4 - a1)
                        spl31 = ql21 * qr21lb * (a4 - a3) / (a4 - a1)
                        spl41 = ql31 * (a4 - c) / (a4 - a1)
                        spl32 = ql22 * qr21lb * (a4 - a3) / (a4 - a2)
                        spl42 = ql32 * (a4 - c) / (a4 - a2)
                        spl43 = ql33 * (a4 - c) / (a4 - a3)
                        spr34 = ql31 * (a3 - a1) / (a4 - a1) + ql32 * (a3 - a2) / (a4 - a2)
                        spr24 = qr21lb * ql21 * (a2 - a1) / (a4 - a1)
                        spr14 = qr31 * (- c - a1) / (a4 - a1)
                        spr23 = qr22 * ql21lb * (a2 -a1) / (a3 - a1)
                        spr13 = qr32 * (- c - a1) / (a3 - a1)
                        spr12 = qr33 * (- c - a1) / (a2 - a1)

                        max_set = [lpl11, lpl21, lpl31, lpl22, lpl32, lpl33, lpr44, lpr34, lpr24, lpr33, lpr23, lpr22]
                        min_set = [spl21, spl31, spl41, spl32, spl42, spl43, spr34, spr24, spr14, spr23, spr13, spr12]

                        for max_pr in max_set:
                            for min_pr in min_set:
                                model += max_pr <= pri * min_pr
                        
                        model += ql21 + ql22 == 1
                        model += ql31 + ql32 + ql33 == 1
                        model += qr21 + qr22 == 1
                        model += qr31 + qr32 + qr33 == 1

                        results = model.solve(solver=solver)

                        if LpStatus[results] == 'Optimal' and value(model.objective) < min:
                            min_q_arr = np.array([[1, value(qr33), value(qr32), value(qr31)],
                                    [value(ql21), value(ql22), value(qr22), value(qr21)],
                                    [value(ql31), value(ql32), value(ql33), 1]])
                            min_bins = bins
                            min = value(model.objective)

    assert min != 100

    np.save(dir + "opt_bins.npy", min_bins)
    np.save(dir + "opt_q_arr.npy", min_q_arr)

    return min_bins, min_q_arr


if __name__ == '__main__':
    
    c = 1
    m = 4
    eps = 1
    pri = math.exp(eps)

    print(opt_par(c, m, eps, 0.5, 0.1))
