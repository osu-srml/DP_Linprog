def calc_bins_pr(bins, bins_sample_pr, x):
    """
    Calculate the probability of different output given input x and selection distribution
    """
    j = -1
    for bid in range(len(bins)-1):
        if bins[bid] <= x < bins[bid+1]:
            j = bid
    assert j != -1, f"Wrong j"

    bins_pr = []
    for l in range(0, j+1):
        pr = 0
        for r in range(j+1, len(bins)):
            pr = pr + bins_sample_pr[r] * ((bins[r] - x) / (bins[r] - bins[l]))
        pr = pr * bins_sample_pr[l]
        bins_pr.append(pr)
    for r in range(j+1, len(bins)):
        pr = 0
        for l in range(0, j+1):
            pr = pr + bins_sample_pr[l] * ((x - bins[l]) / (bins[r] - bins[l]))
        pr = pr * bins_sample_pr[r]
        bins_pr.append(pr)

    return bins_pr


def calc_expected_err(bins, bins_pr, x):
    """
    Calculate the expected absolute error given input x and output distribution
    """
    err = 0
    for i in range(len(bins)):
        err += bins_pr[i] * abs(x - bins[i])
    return err