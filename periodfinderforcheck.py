import numpy as np


def repLineVal(vals,dp):
    round = np.around(vals, dp)
    unique, counts = np.unique(round, return_counts=True)
    val = unique[np.where(counts == np.max(counts))]
    return val[0]


def periodID(t,vals,dp=5):
    val = repLineVal(vals,dp)
    vals = np.around(vals,dp)
    valTs = t[np.where(vals == val)]
    Tarr = []
    for i in range(len(valTs)-1):
        Tarr.append(valTs[i+1]-valTs[i])
    T = np.min(Tarr)
    return val,T


def xvalyval(t,xvals,yvals,dp = 5):
    x_period = periodID(t, xvals, dp=dp)
    xval = x_period[0]
    T = x_period[1]
    xval_t_i = np.where(np.around(xvals, dp) == xval)[0][0]
    xval_t = t[xval_t_i]
    yval = yvals[np.where(t == xval_t)[0]][0]
    return xval,yval,T


