import numpy as np

"""
Functions to manually find the periodic orbit of a function by looking for repeated values in its x and y values.
"""


def repeated_line_value(vals):
    """
    Takes a list of values and returns the value which repeats the most.
    :param vals: List of values
    :return: Returns value from the list of values which repeat the most
    """

    unique, counts = np.unique(vals, return_counts=True)
    val = unique[np.where(counts == np.max(counts))]
    return val[0]


def periodID(t, vals, dp=5):
    """
    Takes a list of times and corresponding values and returns the value which repeats the most and the time period of
    its repeats
    :param t: List of times
    :param vals: List of values
    :param dp: Decimal places to round values to for comparison
    :return: Returns val, T where val is the most repeated value and T is the time period of the repeats
    """

    vals = np.around(vals, dp)  # rounds all values to specified dp
    val = repeated_line_value(vals)  # most repeated value
    val_ts = t[np.where(vals == val)]  # values of t corresponding with val
    T_array = []

    """
    Subtracts pairwise t values to get time period of repeated values
    """
    for i in range(len(val_ts) - 1):
        T_array.append(val_ts[i + 1] - val_ts[i])

    """
    Smallest T is best. All Ts will be multiples of each other, but the smallest value is the true time period.
    """
    T = np.min(T_array)
    return val, T


def manual_period_finder(t, xvals, yvals, dp=5):
    """
    Manually finds period orbit given a list of time, x, and y values.
    :param t: List of time values
    :param xvals: List of x values
    :param yvals: List of y values
    :param dp: Decimal places to round values to for comparison
    :return: Returns the x,y,T where (x,y) is the start coordinate of the periodic orbit and T is its time period
    """
    x_period = periodID(t, xvals, dp=dp)  # Gets most repeated value and its time period for the x values
    x_val = x_period[0]
    T = x_period[1]

    # Gets index of first occurrence of  most repeated x value
    x_val_t_i = np.where(np.around(xvals, dp) == x_val)[0][0]
    # Gets corresponding time value of first occurrence of  most repeated x value
    x_val_t = t[x_val_t_i]
    # Gets corresponding y value of first occurrence of  most repeated x value
    y_val = yvals[np.where(t == x_val_t)[0]][0]

    """
    Returns the x value, y value, and time period.
    This is the start (x,y) coordinates of the period orbit and its time period T.
    """
    return x_val, y_val, T
