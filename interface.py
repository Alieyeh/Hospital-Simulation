import pandas as pd
import matplotlib.pyplot as plt
import Hospital.acutestrokeunit as acutestrokeunit
from Hospital.acutestrokeunit import AcuteStrokeUnit, multiple_replications, Scenario, \
    warmup_analysis, single_run, run_scenario_analysis, get_scenarios, scenario_summary_frame, \
    sensitivity_scenarios, sensitivity_summary_frame, plot_tornado
from Hospital.paramdetection import confidence_interval_method, plot_confidence_interval_method, \
    warmup_detection, randomization_test

# set default constants
acutestrokeunit.DEFAULT_RESULTS_COLLECTION_PERIOD = 365 * 5
acutestrokeunit.DEFAULT_N_REPS = 10
acutestrokeunit.DEFAULT_WARMUP = 0
acutestrokeunit.N_BEDS = 9
acutestrokeunit.TRACE = False

# base case scenario with default parameters
default_args = Scenario()


def single():
    """
    Preforms a single run of the model with no input and default constants.
    Prints results and summary.
    """
    # run model
    results = single_run(default_args)

    # print results
    print(results)


def get_warmup():
    """
    This function calculates and plots the results of the randomization method
    of obtaining the optimal warm-up period.

    Output:
    --------
        Plots results of the randomization method. Returns nothing.
    """
    # set constants
    acutestrokeunit.DEFAULT_RESULTS_COLLECTION_PERIOD = 365 * 5
    acutestrokeunit.DEFAULT_N_REPS = 10
    acutestrokeunit.DEFAULT_WARMUP = 0

    # get warmup period
    results = warmup_analysis(default_args)
    warmup_detection(results, warm_up=50)
    plt.show()
    randomization_test(results)


def rep_num():
    """
    This function calculates and plots the results of the confidence interval
    method for obtaining the optimal number of replications.

    Output:
    --------
        Plots results of the confidence interval method. Returns nothing.
    """
    # 50*5 as warmup, now calculate the replication numbers
    acutestrokeunit.DEFAULT_WARMUP = 250
    acutestrokeunit.DEFAULT_N_REPS = 100

    replications = multiple_replications(default_args)
    # print(replications)
    n_reps, conf_ints = confidence_interval_method(replications['percentage'].to_numpy() * 100,
                                                   desired_precision=0.05, min_rep=50)

    print('Analysis of replications for operator utilisation...')

    # print out the min number of replications to achieve precision
    print(f'\nminimum number of reps for 5% precision: {n_reps}\n')

    # plot the confidence intervals
    ax = plot_confidence_interval_method(n_reps, conf_ints,
                                         metric_name='beds_util')
    plt.show()


# different scenario
def get_scen():
    """
    This function sets the constant parameters and run various scenarios
    with different bed numbers and arrival rates and saves the summary of
    these scenarios into a csv.

    Returns:
    --------
        pd.DataFrame
            The dataframe containing the summary of all produced scenarios.
    """
    acutestrokeunit.DEFAULT_RESULTS_COLLECTION_PERIOD = 365
    acutestrokeunit.DEFAULT_WARMUP = 250
    acutestrokeunit.DEFAULT_N_REPS = 50

    scenarios = get_scenarios(n_beds=range(9, 15), admission_increase=[0, 0.05, 0.1, 0.15, 0.2])
    scenario_results = run_scenario_analysis(scenarios)
    summary_frame = scenario_summary_frame(scenario_results)
    summary_frame.round(2).to_csv("summary.csv")
    summary_frame = pd.read_csv("summary.csv")
    return summary_frame


# The result summary
def plot_summary(df):
    """
    This function plots two subplots, one showing the preformance given
    an increase in number of beds (9-14) and an increase in admission rate
    (0-20). The other shows bed utilization for the same data.

    Parameters:
    ----------
    df: pd.DataFrame
       Pandas data frame containing our model summary data.

    Output:
    --------
        Plots model summary. Does not return anything.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    beds_0 = df[df.columns[1::5]]
    beds_5 = df[df.columns[2::5]]
    beds_10 = df[df.columns[3::5]]
    beds_15 = df[df.columns[4::5]]
    beds_20 = df[df.columns[5::5]]

    ax[0].plot([9, 10, 11, 12, 13, 14], beds_0.loc['mean_percentage'].values, marker='o',
               markersize=8, linestyle='-', label='0% increased admission')
    ax[0].fill_between([9, 10, 11, 12, 13, 14], beds_0.loc['lower_percentage'],
                       beds_0.loc['upper_percentage'], alpha=0.2)
    ax[0].plot([9, 10, 11, 12, 13, 14], beds_5.loc['mean_percentage'].values, marker='o',
               markersize=8, linestyle='-', label='5% increased admission')
    ax[0].fill_between([9, 10, 11, 12, 13, 14], beds_5.loc['lower_percentage'],
                       beds_5.loc['upper_percentage'], alpha=0.2)
    ax[0].plot([9, 10, 11, 12, 13, 14], beds_10.loc['mean_percentage'].values, marker='o',
               markersize=8, linestyle='-', label='10% increased admission')
    ax[0].fill_between([9, 10, 11, 12, 13, 14], beds_10.loc['lower_percentage'],
                       beds_10.loc['upper_percentage'], alpha=0.2)
    ax[0].plot([9, 10, 11, 12, 13, 14], beds_15.loc['mean_percentage'].values, marker='o',
               markersize=8, linestyle='-', label='15% increased admission')
    ax[0].fill_between([9, 10, 11, 12, 13, 14], beds_15.loc['lower_percentage'],
                       beds_15.loc['upper_percentage'], alpha=0.2)
    ax[0].plot([9, 10, 11, 12, 13, 14], beds_20.loc['mean_percentage'].values, marker='o',
               markersize=8, linestyle='-', label='20% increased admission')
    ax[0].fill_between([9, 10, 11, 12, 13, 14], beds_20.loc['lower_percentage'],
                       beds_20.loc['upper_percentage'], alpha=0.2)

    y1 = beds_0.loc['mean_percentage'].values[0]
    y2 = beds_0.loc['mean_percentage'].values[2]
    ax[0].annotate(f"Baseline: (+ 9, + {y1})", xy=(9, y1 + 0.01), xytext=(9 - 0.1, y1 + 0.2),
                   fontsize=9, arrowprops=dict(facecolor='darkgreen', shrink=0.1))
    ax[0].annotate(f"(+ 11, + {y2})", xy=(11, y2 + 0.01), xytext=(11 + 0.1, y2 + 0.08),
                   fontsize=9, arrowprops=dict(facecolor='darkgreen', shrink=0.2))
    ax[0].set_ylabel('precentage of patients within 4 hours (%)')
    ax[0].set_xlabel('number of beds')
    ax[0].plot(type='scatter')
    ax[0].axvline(x=11, ls='--', color='green')
    ax[0].legend()

    ax[1].plot([9, 10, 11, 12, 13, 14], beds_0.loc['mean_beds_util'].values, marker='o',
               markersize=8, linestyle='-', label='0% increased admission')
    ax[1].fill_between([9, 10, 11, 12, 13, 14], beds_0.loc['lower_beds_util'],
                       beds_0.loc['upper_beds_util'], alpha=0.2)
    ax[1].plot([9, 10, 11, 12, 13, 14], beds_5.loc['mean_beds_util'].values, marker='o',
               markersize=8, linestyle='-', label='5% increased admission')
    ax[1].fill_between([9, 10, 11, 12, 13, 14], beds_5.loc['lower_beds_util'],
                       beds_5.loc['upper_beds_util'], alpha=0.2)
    ax[1].plot([9, 10, 11, 12, 13, 14], beds_10.loc['mean_beds_util'].values, marker='o',
               markersize=8, linestyle='-', label='10% increased admission')
    ax[1].fill_between([9, 10, 11, 12, 13, 14], beds_10.loc['lower_beds_util'],
                       beds_10.loc['upper_beds_util'], alpha=0.2)
    ax[1].plot([9, 10, 11, 12, 13, 14], beds_15.loc['mean_beds_util'].values, marker='o',
               markersize=8, linestyle='-', label='15% increased admission')
    ax[1].fill_between([9, 10, 11, 12, 13, 14], beds_15.loc['lower_beds_util'],
                       beds_15.loc['upper_beds_util'], alpha=0.2)
    ax[1].plot([9, 10, 11, 12, 13, 14], beds_20.loc['mean_beds_util'].values, marker='o',
               markersize=8, linestyle='-', label='20% increased admission')
    ax[1].fill_between([9, 10, 11, 12, 13, 14], beds_20.loc['lower_beds_util'],
                       beds_20.loc['upper_beds_util'], alpha=0.2)

    y1 = beds_0.loc['mean_beds_util'].values[0]
    y2 = beds_0.loc['mean_beds_util'].values[2]
    ax[1].set_ylabel('mean bed utilization (%)')
    ax[1].set_xlabel('number of beds')
    ax[1].plot(type='scatter')
    ax[1].axvline(x=11, ls='--', color='green')
    ax[1].annotate(f"Baseline: (+ 9, + {y1})", xy=(9, y1 - 0.01), xytext=(9 - 0.1, y1 - 0.1),
                   fontsize=9, arrowprops=dict(facecolor='darkgreen', shrink=0.1))
    ax[1].annotate(f"(+ 11, + {y2})", xy=(11, y2 - 0.01), xytext=(11 - 0.1, y2 - 0.1),
                   fontsize=9, arrowprops=dict(facecolor='darkgreen', shrink=0.1))
    ax[1].legend()


def sensitivity():
    """
    Preforms a sensitivity analysis on number of beds, length of stay,
    admission rate and priority (for queuing discipline). It then creates
    a tornado plot of the results.

    Output:
    --------
        Plots a Tornado plot of sensitivity. Does not return anything.
    """
    # set constant parameters
    acutestrokeunit.DEFAULT_RESULTS_COLLECTION_PERIOD = 365
    acutestrokeunit.DEFAULT_WARMUP = 250
    acutestrokeunit.DEFAULT_N_REPS = 50

    sens = sensitivity_scenarios(n_beds=range(5, 15))
    sensitivity_bed = sensitivity_summary_frame(run_scenario_analysis(sens))
    increase_list = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]

    # one way sensitivity for admission
    sens = sensitivity_scenarios(admission_increase=increase_list)
    sensitivity_admission = sensitivity_summary_frame(run_scenario_analysis(sens))

    # one way sensitivity for admission
    sens = sensitivity_scenarios(stay_length_increase=increase_list)
    sensitivity_stay = sensitivity_summary_frame(run_scenario_analysis(sens))

    # one way sensitivity for prior
    sens = sensitivity_scenarios(priority=1)
    sensitivity_prior = sensitivity_summary_frame(run_scenario_analysis(sens))

    results = {"Beds number \n(5 to 14 beds)": sensitivity_bed,
               "Arrival of admissions \n(base case x0.8 to x 1.2)": sensitivity_admission,
               "Stay length in hospital \n(base case x0.8 to x 1.2)": sensitivity_stay,
               "Admitted per priority": sensitivity_prior}

    # print(np.array(list(results.values()))[:, 0])
    plot_tornado(results)
    plt.show()
