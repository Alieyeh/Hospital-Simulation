import pandas as pd
import acutestrokeunit
# from acutestrokeunit import AcuteStrokeUnit, single_run, multiple_replications, Scenario
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Turn on tracing
acutestrokeunit.TRACE = False
acutestrokeunit.DEFAULT_RESULTS_COLLECTION_PERIOD = 400
acutestrokeunit.DEFAULT_WARMUP = 0

# base case scenario with default parameters
default_args = acutestrokeunit.Scenario()

# create the model
# model = acutestrokeunit.AcuteStrokeUnit(default_args)

# set up the process
# model.run()
# results = model.run_summary_frame()
# results = acutestrokeunit.single_run(default_args)
results = acutestrokeunit.multiple_replications(default_args)
print(results['time_to_beds'].mean(axis=1))
# acutestrokeunit.time_series_inspection(results, warm_up=120)

# print(f'end of run. simulation clock time = {env.now}')
#
# mt = [pt.time_to_bed for pt in model.patients]
# print(np.mean(mt))
# at = [pt.four_hour_target for pt in model.patients]
# print(np.mean(at))

