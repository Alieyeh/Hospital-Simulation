"""
xxxxxx
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import pandas as pd
import warnings
import random
import seaborn as sns


def warmup_detection(results, warm_up=None):
    """
    Time series inspection method (graphical) for warmup period detection

    Parameters:
    ----------
    results: dict
        The dict of results taken from warmup_analysis
    """
    # create the 4 chart areas to plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 9))

    # take the mean of the columns for each metric and plot
    ax[0].plot(results['percentage'].mean(axis=1))
    ax[1].plot(results['beds_util'].mean(axis=1))

    # set the label of each chart
    ax[0].set_ylabel('beds_wait_within4')
    ax[1].set_ylabel('beds_util')

    if warm_up is not None:
        # add warmup cut-off vertical line if one is specified
        ax[0].axvline(x=warm_up, color='red', ls='--')
        ax[1].axvline(x=warm_up, color='red', ls='--')

    return fig, ax


def randomization_test(results):
    """
    Randomization test method (statistical) for warmup period detection

    Parameters:
    ----------
    results: pd.Dataframe
        The dataframe of results taken from warmup_analysis
    """
    warmup_period = []
    for col in results:
        rep_mean = list(results[col].mean(axis=1))
        temp = [rep_mean]
        for n_rand in range(10000):
            rand = random.sample(rep_mean, len(rep_mean))
            temp.append(rand)

        for batch in range(1, 51):
            arr1 = np.mean(np.array(temp)[:, :batch], axis=1)
            arr2 = np.mean(np.array(temp)[:, batch:], axis=1)
            diff = arr2 - arr1
            observed_diff = diff[0]
            count = 0
            for rand_diff in diff[1:]:
                if rand_diff >= observed_diff:
                    count += 1
            p = count / 10000

            if p >= 0.05:
                break
        warmup_period.append(batch)
        print(f"The warmup period for {col} is {batch}")

    return warmup_period


def confidence_interval_method(replications, alpha=0.05, desired_precision=0.05,
                               min_rep=5, decimal_place=2):
    """
    The confidence interval method for selecting the number of replications
    to run in a simulation.

    Finds the smallest number of replications where the width of the confidence
    interval is less than the desired_precision.

    Returns both the number of replications and the full results dataframe.

    Parameters:
    ----------
    replications: array-like
        Array (e.g. np.ndarray or list) of replications of a performance metric

    alpha: float, optional (default=0.05)
        procedure constructs a 100(1-alpha) confidence interval for the
        cumulative mean.

    desired_precision: float, optional (default=0.05)
        Desired mean deviation from confidence interval.

    min_rep: int, optional (default=5)
        set to an integer > 0 and ignore all the replications prior to it
        when selecting the number of replications to run to achieve the desired
        precision.  Useful when the number of replications returned does not
        provide a stable precision below target.

    decimal_places: int, optional (default=2)
        sets the number of decimal places of the returned dataframe containing
        the results

    Returns:
    --------
        tuple: int, pd.DataFrame

    """
    n = len(replications)
    cumulative_mean = [replications[0]]
    running_var = [0.0]
    for i in range(1, n):
        cumulative_mean.append(cumulative_mean[i - 1] +
                               (replications[i] - cumulative_mean[i - 1]) / (i + 1))

        # running biased variance
        running_var.append(running_var[i - 1] + (replications[i]
                                                 - cumulative_mean[i - 1])
                           * (replications[i] - cumulative_mean[i]))

    # unbiased std dev = running_var / (n - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        running_std = np.sqrt(running_var / np.arange(n))

    # half width of interval
    dof = len(replications) - 1
    t_value = t.ppf(1 - (alpha / 2), dof)
    with np.errstate(divide='ignore', invalid='ignore'):
        std_error = running_std / np.sqrt(np.arange(1, n + 1))

    half_width = t_value * std_error

    # upper and lower confidence interval
    upper = cumulative_mean + half_width
    lower = cumulative_mean - half_width

    # Mean deviation
    with np.errstate(divide='ignore', invalid='ignore'):
        deviation = (half_width / cumulative_mean) * 100

    # combine results into a single dataframe
    results = pd.DataFrame([replications, cumulative_mean,
                            running_std, lower, upper, deviation]).T
    results.columns = ['Mean', 'Cumulative Mean', 'Standard Deviation',
                       'Lower Interval', 'Upper Interval', '% deviation']
    results.index = np.arange(1, n + 1)
    results.index.name = 'replications'

    # get the smallest no. of reps where deviation is less than precision target
    try:
        n_reps = results.iloc[min_rep:].loc[results['% deviation']
                                            <= desired_precision * 100].iloc[0].name
    except:
        # no replications with desired precision
        message = 'WARNING: the replications do not reach desired precision'
        warnings.warn(message)
        n_reps = -1

    return n_reps, results.round(decimal_place)


def plot_confidence_interval_method(n_reps, conf_ints, metric_name):
    """
    Plot the confidence intervals and cumulative mean

    Parameters:
    ----------
    n_reps: int
        minimum number of reps selected

    conf_ints: pandas.DataFrame
       results of the `confidence_interval_method` function

    metric_name: str
        Name of the performance measure

    figsize: tuple, optional (default=(12,4))
        The size of the plot

    Returns:
    -------
        matplotlib.pyplot.axis
    """
    # plot cumulative mean + lower/upper intervals
    ax = sns.lineplot(x=conf_ints.index, y='Cumulative Mean', data=conf_ints)
    ax.fill_between(conf_ints.index, conf_ints['Lower Interval'], conf_ints['Upper Interval'], alpha=0.2)
    # add the
    ax.axvline(x=n_reps, ls='--', color='red')

    ax.set_ylabel(f'cumulative mean: {metric_name} with 95% CI')

    return ax
