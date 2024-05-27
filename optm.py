import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import genm
from pulp import *
import math
import numpy as np


def optimize(c, delta, m, eps, **kw):
    """
    Optimize for the parameters with given bin values
    """
    bins = []
    q_arr = []
    pri = math.exp(eps)
    dis = kw['dis']
    assert dis <= c, f"dis > c" 
    assert m == 4

    a1 = - c - delta
    a2 = - dis
    a3 = dis
    a4 = c + delta
    bins = [a1, a2, a3, a4]
    
    min = 100

    for l21 in np.arange(0.01, 1, 0.01):
        u21 = l21 + 0.01

        model = pulp.LpProblem('linear_programming', LpMinimize)

        # get solver
        solver = pulp.PULP_CBC_CMD(msg=0)

        # declare decision variables
        q21 = LpVariable('q21', lowBound = l21, upBound = u21, cat = 'continuous')
        q22 = LpVariable('q22', lowBound = 1e-5, cat = 'continuous')

        q31 = LpVariable('q31', lowBound = 1e-5, cat = 'continuous')
        q32 = LpVariable('q32', lowBound = 1e-5, cat = 'continuous')
        q33 = LpVariable('q33', lowBound = 1e-5, cat = 'continuous')

        # declare objective
        model += (a2 + c) * (q32 * (a3 - a2) + q31 * (a4 - a2)) + (-a2) * 2 * (q21 * (a2 - a1))

        # declare constraints
        model += q21 + q22 == 1
        model += q31 + q32 + q33 == 1

        p11 = q33 * (a2 + c) / (a2 - a1) + q32 * (a3 + c) / (a3 - a1) + q31 * (a4 + c) / (a4 - a1)

        # larger probability
        lp21 = u21 * (q22 * (a3 - a2) / (a3 - a1) + q21 * (a4 - a2) / (a4 - a1))
        lp31 = q31 * ((a4 - a3) / (a4 - a1))
        
        # smaller probability
        sp21 = q32 * (a3 - a2) / (a3 - a1) + q31 * (a4 - a2) / (a4 - a1)
        sp31 = l21 * q21 * (a4 - a3) / (a4 - a1)
        sp41 = q31 * (a4 - c) / (a4 - a1)

        p32 = q22 * l21 * (a4 - a3) / (a4 - a2)
        p42 = q32 * (a4 - c) / (a4 - a2)

        p43 = q33 * (a4 - c) / (a4 - a3)

        model += p11 <= pri * sp21
        model += p11 <= pri * sp31
        model += p11 <= pri * sp41

        model += lp21 <= pri * sp21
        model += lp21 <= pri * sp31
        model += lp21 <= pri * sp41

        model += lp31 <= pri * sp21
        model += lp31 <= pri * sp31
        model += lp31 <= pri * sp41

        model += q22 <= pri * p32
        model += q22 <= pri * p42

        model += q33 <= pri * p32
        model += q33 <= pri * p42

        model += q33 <= pri * p43

        model += q22 <= pri * p43

        # solve
        results = model.solve(solver=solver)

        if LpStatus[results] == 'Optimal' and value(model.objective) < min:
            q_arr = [value(q21), value(q22), value(q31), value(q32), value(q33)]
            min = value(model.objective) 

    return bins, q_arr


def opt_par(c, m, eps, **kw):
    """
    Using grid search to find the optimal bin values
    """
    dir = "parameters/OPTM/c{}_m{}_eps{:.1f}/".format(c, m, eps)
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    elif len(kw) == 0:
        bins = np.load(dir + "opt_bins.npy")
        delta = bins[-1] - c
        dis = bins[2]  # The value of delta and dis can decide the bin values when m = 4
        q_arr = np.load(dir + "delta{:.2f}_dis{:.2f}_q_arr.npy".format(delta, dis))
        return bins, q_arr
    
    if len(kw) != 0:
        bins = kw['bins']
        delta = bins[-1] - c
        dis = bins[2]
        name = dir + "delta{:.2f}_dis{:.2f}_q_arr.npy".format(delta, dis)
        if os.path.exists(name) == True:
            q_arr = np.load(name)
        else:
            bins, q_arr = optimize(c, delta, m, eps, dis=dis)
            q21 = q_arr[0]
            q22 = 1 - q21
            q31 = q_arr[2]
            q32 = q_arr[3]
            q33 = 1 - q31 - q32
            q_arr = [[1, q33, q32, q31],
                    [q21, q22, q22, q21],
                    [q31, q32, q33, 1]]
            np.save(dir + "delta{:.2f}_dis{:.2f}_q_arr.npy".format(delta, dis), np.array(q_arr))
        return bins, q_arr

    min_err = 100
    min_delta = -1
    min_dis = -1
    min_q_arr = []
    min_bins = []

    for r in np.arange(1, 2.1, 0.1):
        delta = c * r
        for dis in np.arange(0.1, 1.1, 0.1):
            bins, q_arr = optimize(c, delta, m, eps, dis=dis) 
            if q_arr != []:
                q21 = q_arr[0]
                q22 = 1 - q21
                q31 = q_arr[2]
                q32 = q_arr[3]
                q33 = 1 - q31 - q32

                sample_pr_arr = [[1, q33, q32, q31],
                                    [q21, q22, q22, q21],
                                    [q31, q32, q33, 1]]
                
                x_iter = np.arange(-c, c+0.01, 0.04)
                optm_err_arr = []
                for x in x_iter:
                    nuj = -1
                    for id in range(len(bins)-1):
                        if bins[id] <= x < bins[id+1]:
                            nuj = id
                    assert nuj != -1, f"Wrong j for NUOPTM"
                    
                    optm_bins_pr = genm.calc_bins_pr(bins, sample_pr_arr[nuj], x)
                    optm_err = genm.calc_expected_err(bins, optm_bins_pr, x)
                    optm_err_arr.append(optm_err)
                
                err = np.mean(optm_err_arr)
                if err < min_err:
                    min_delta = delta
                    min_dis = dis
                    min_err = err
                    min_bins = bins
                    min_q_arr = sample_pr_arr

    assert min_err != 100
    np.save(dir + "opt_delta.npy", np.array(min_delta))
    np.save(dir + "opt_bins.npy", np.array(min_bins))
    np.save(dir + "delta{:.2f}_dis{:.2f}_q_arr.npy".format(min_delta, min_dis), np.array(min_q_arr))

    return min_bins, min_q_arr


# if __name__ == '__main__':

#     c = 1
#     m = 4
#     eps = 1
#     print(opt_par(c, m, eps))
