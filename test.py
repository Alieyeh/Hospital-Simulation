import random

import pandas as pd
import acutestrokeunit
from acutestrokeunit import AcuteStrokeUnit, multiple_replications, Scenario, warmup_analysis, single_run, \
    run_scenario_analysis, get_scenarios, scenario_summary_frame, sensitivity_scenarios,\
    sensitivity_summary_frame, plot_tornado
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from paramdetection import confidence_interval_method, plot_confidence_interval_method, warmup_detection, \
    randomization_test

# Turn on tracing
acutestrokeunit.DEFAULT_RESULTS_COLLECTION_PERIOD = 365
acutestrokeunit.TRACE = False
acutestrokeunit.DEFAULT_WARMUP = 250
acutestrokeunit.DEFAULT_N_REPS = 50
acutestrokeunit.N_BEDS = 9

# base case scenario with default parameters
default_args = Scenario()

# create the model
# model = AcuteStrokeUnit(default_args)

# set up the process
# model.run()
# results = model.run_summary_frame()
# results = single_run(default_args)
# results.to_csv("test.csv")
# print(results)
# print(re)
# results = multiple_replications(default_args)
# print(np.mean(results, axis=0))
# print(results)
# time_series_inspection(results, warm_up=120)

# print(f'end of run. simulation clock time = {env.now}')
#
# mt = [pt.time_to_bed for pt in model.patients]
# print(np.mean(mt))
# at = [pt.four_hour_target for pt in model.patients]
# print(np.mean(at))
# run for 40 days
# RUN_LENGTH = 365*5

# run at least 5 replications, but more might be needed for noisy data
# N_REPS = 10

# get warmup period
# results = warmup_analysis(default_args)
# warmup_detection(results, warm_up=100)
# plt.show()
# randomization_test(results)


# 50*5 as warmup, now calculate the replication numbers
# replications = multiple_replications(default_args)
# print(replications)
# n_reps, conf_ints = confidence_interval_method(replications['percentage'].to_numpy() * 100,
#                                                desired_precision=0.05, min_rep=50)
#
# print('Analysis of replications for operator utilisation...')
#
# # print out the min number of replications to achieve precision
# print(f'\nminimum number of reps for 5% precision: {n_reps}\n')
#
# # plot the confidence intervals
# ax = plot_confidence_interval_method(n_reps, conf_ints,
#                                      metric_name='beds_util')
# plt.show()


# different scenario
# scenarios = get_scenarios(n_beds=range(10, 15), admission_increase=[0.05, 0.1, 0.15, 0.2])
# scenario_results = run_scenario_analysis(scenarios)
# summary_frame = scenario_summary_frame(scenario_results)
# summary_frame.round(2).to_csv("test.csv")

# one-way sensitivity analysis
# one-way sensitivity for beds
sensitivity = sensitivity_scenarios(n_beds=range(5, 15))
sensitivity_bed = sensitivity_summary_frame(run_scenario_analysis(sensitivity))

# one way sensitivity for admission
sensitivity = sensitivity_scenarios(admission_increase=[-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])
sensitivity_admission = sensitivity_summary_frame(run_scenario_analysis(sensitivity))

# one way sensitivity for admission
sensitivity = sensitivity_scenarios(stay_length_increase=[-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])
sensitivity_stay = sensitivity_summary_frame(run_scenario_analysis(sensitivity))

# one way sensitivity for prior
sensitivity = sensitivity_scenarios(priority=1)
sensitivity_prior = sensitivity_summary_frame(run_scenario_analysis(sensitivity))

results = {"Beds number \n(5 to 14 beds)": sensitivity_bed,
           "Arrival of admissions \n(base case x0.8 to x 1.2)": sensitivity_admission,
           "Stay length in hospital \n(base case x0.8 to x 1.2)": sensitivity_stay,
           "Admitted per priority": sensitivity_prior}

# print(np.array(list(results.values()))[:, 0])
plot_tornado(results)
plt.show()




