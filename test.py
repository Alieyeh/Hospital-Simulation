from acutestrokeunit import AcuteStrokeUnit, Patient, Scenario
import simpy
import numpy as np

# Turn on tracing
TRACE = True
DEFAULT_RESULTS_COLLECTION_PERIOD = 365*2

# create simpy environment
env = simpy.Environment()

# base case scenario with default parameters
default_args = Scenario()

# create the model
model = AcuteStrokeUnit(env, default_args)

# set up the process
# model.arrivals_generator()
# env.run(until=365)
model.run()
print(f'end of run. simulation clock time = {env.now}')

mt = [pt.time_to_bed for pt in model.patients]
print(mt)
at = [pt.four_hour_target for pt in model.patients]
print(np.mean(at))
