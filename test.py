import pandas as pd

from acutestrokeunit import AcuteStrokeUnit, single_run, Scenario
import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Turn on tracing
TRACE = True
DEFAULT_RESULTS_COLLECTION_PERIOD = 365*2

# base case scenario with default parameters
default_args = Scenario()

# create the model
model = AcuteStrokeUnit(default_args)

# set up the process
# model.run()
# results = model.run_summary_frame()
results = single_run(default_args)
print(results)
# print(f'end of run. simulation clock time = {env.now}')
#
# mt = [pt.time_to_bed for pt in model.patients]
# print(np.mean(mt))
# at = [pt.four_hour_target for pt in model.patients]
# print(np.mean(at))

