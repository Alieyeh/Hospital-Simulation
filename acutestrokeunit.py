"""
An acute stroke unit model .

Main model class: AcuteStrokeUnit
Patient process class: Patient

Process overview:
1. Patients arrive at acute stroke unit (ACU) and
   waits for a bed for further treatment
2. Patients stay in the hospital for a few days and discharge
"""

import numpy as np
import pandas as pd
import itertools
import simpy
from joblib import Parallel, delayed
from Hospital.distributions import (Exponential, Lognormal)
import matplotlib.pyplot as plt

# declare constants for module
# default bed resources
N_BEDS = 9

# default parameters for inter-arrival times distributions (days)
MEAN_IAT1 = 1.2
MEAN_IAT2 = 9.5
MEAN_IAT3 = 3.5

# default parameters for stay length distributions
MEAN_STAY1 = 7.4
STD_STAY1 = 8.5
MEAN_STAY2 = 1.8
STD_STAY2 = 2.3
MEAN_STAY3 = 2.0
STD_STAY3 = 2.5

# Should we show a trace of simulated events?
TRACE = False

# default random number SET
DEFAULT_RNG_SET = 1234
N_STREAMS = 6

# default results collection period
DEFAULT_RESULTS_COLLECTION_PERIOD = 365*5

# default number of replications
DEFAULT_N_REPS = 10

# warmup
DEFAULT_WARMUP = 0

# warmup auditing
DEFAULT_WARMUP_AUDIT_INTERVAL = 5

# accepting patient by priority?
PRIORITY = False


def trace(msg):
    """
    Utility function for printing simulation
    set the TRACE constant FALSE to
    turn tracing off.

    Params:
    -------
    msg: str
        string to print to screen.
    """
    if TRACE:
        print(msg)


class Scenario:
    """
    Parameter class for ACU simulation model
    """

    def __init__(self, random_number_set=DEFAULT_RNG_SET):
        """
        The init method sets up our defaults.

        Parameters:
        -----------
        random_number_set: int, optional (default=DEFAULT_RNG_SET)
            Set to control the initial seeds of each stream of pseudo
            random numbers used in the model.

        """
        # resource counts
        self.n_beds = N_BEDS
        self.prior = PRIORITY

        # inter-arrival rate
        self.mean_iat1 = MEAN_IAT1
        self.mean_iat2 = MEAN_IAT2
        self.mean_iat3 = MEAN_IAT3

        # stay in hospital
        self.mean_stay1 = MEAN_STAY1
        self.std_stay1 = STD_STAY1
        self.mean_stay2 = MEAN_STAY2
        self.std_stay2 = STD_STAY2
        self.mean_stay3 = MEAN_STAY3
        self.std_stay3 = STD_STAY3

        # warm-up
        self.warm_up = DEFAULT_WARMUP

        # run length
        self.run_length = DEFAULT_RESULTS_COLLECTION_PERIOD

        # treat patients base on priority?
        self.prior = PRIORITY

        # sampling
        self.random_number_set = random_number_set
        self.init_sampling()

    def set_random_no_set(self, random_number_set):
        """
        Controls the random sampling

        Parameters:
        ----------
        random_number_set: int
            Used to control the set of pseudo random numbers
            used by the distributions in the simulation.
        """
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        """
        Create the distributions used by the model and initialise
        the random seeds of each.
        """
        # create random number streams
        rng_streams = np.random.default_rng(self.random_number_set)
        self.seeds = rng_streams.integers(0, 999999999, size=N_STREAMS)

        # create inter-arrival distributions
        self.arrival_dist1 = Exponential(self.mean_iat1, random_seed=self.seeds[0])
        self.arrival_dist2 = Exponential(self.mean_iat2, random_seed=self.seeds[1])
        self.arrival_dist3 = Exponential(self.mean_iat3, random_seed=self.seeds[2])

        # create study length distributions
        self.stay_dist1 = Lognormal(MEAN_STAY1, STD_STAY1, random_seed=self.seeds[3])
        self.stay_dist2 = Lognormal(MEAN_STAY2, STD_STAY2, random_seed=self.seeds[4])
        self.stay_dist3 = Lognormal(MEAN_STAY3, STD_STAY3, random_seed=self.seeds[5])


class Patient:
    """
    Patient in the ACU process
    """

    def __init__(self, identifier, stroke_type, env, args):
        """
        Constructor method

        Params:
        -----
        identifier: int
            a numeric identifier for the patient.

        stroke_type: int
            stroke type (1-Acute strokes, 2-TIA, 3-Complex Neurological)

        env: simpy.Environment
            the simulation environment

        args: Scenario
            The input data for the scenario
        """
        # patient id, type, and, environment
        self.identifier = identifier
        self.type = stroke_type
        self.env = env

        # triage parameters
        self.args = args
        self.beds = args.beds

        # stay length distributions
        self.stay_dist1 = args.stay_dist1
        self.stay_dist2 = args.stay_dist2
        self.stay_dist3 = args.stay_dist3

        # individual patient metrics
        self.arrival_time = 0.000
        self.stay_duration = 0.000
        self.time_to_bed = 0.000
        self.four_hour_target = 0

    def assessment(self):
        """
        simulates the process for ACU

        1. request and wait for a bed
        2. treatment
        3. exit system
        """
        # record the time that patient entered the system
        self.arrival_time = self.env.now

        # allocate beds based on priority type or not
        pri = None
        if self.args.prior:
            pri = self.type

        # request a bed
        with self.beds.request(priority=pri) as req:
            yield req

            # calculate waiting time to bed
            self.time_to_bed = self.env.now - self.arrival_time
            # calculate if meets the 4-hour target
            if 0 <= self.time_to_bed <= 1/6:
                self.four_hour_target = 1
            self.waiting_complete()

            # sampling stay in hospital duration.
            if self.type == 1:
                self.stay_duration = self.stay_dist1.sample()
            elif self.type == 2:
                self.stay_duration = self.stay_dist2.sample()
            elif self.type == 3:
                self.stay_duration = self.stay_dist3.sample()

            yield self.env.timeout(self.stay_duration)

            self.treatment_complete()

    def waiting_complete(self):
        """
        Printing information for time to bed
        """
        trace(f'2. patient {self.identifier}, type {self.type} waiting for bed ended {self.env.now:.3f}; '
              + f'waiting time was {self.time_to_bed:.3f}')

    def treatment_complete(self):
        """
        Printing information when discharging the hospital
        """
        trace(f'3. patient {self.identifier}, type {self.type} staying in hospital ended {self.env.now:.3f}; '
              + f'stay length was {self.stay_duration:.3f}')


class AcuteStrokeUnit:
    """
    Main class to simulate the working of an acute stroke unit.
    """

    def __init__(self, args):
        """
        Params:
        -------
        args: Scenario
            container class for simulation model inputs.
        """
        self.env = simpy.Environment()
        self.args = args
        self.init_model_resources(args)

        self.patients = []
        self.patient_count = 0

    def init_model_resources(self, args):
        """
        Set up the simpy resource objects

        Params:
        ------
        args - Scenario
            Simulation Parameter Container
        """
        args.beds = simpy.PriorityResource(self.env, capacity=args.n_beds)

    def run(self):
        """
        Conduct a single run of the model in its current
        configuration.
        """
        # set up the arrival process
        self.env.process(self.type1())
        self.env.process(self.type2())
        self.env.process(self.type3())

        # run
        self.env.run(until=self.args.run_length + self.args.warm_up)

    def generate_new_arrival(self, patient_type):
        """
        Create new patients.

        Params:
        --------
        patient_type: int
            stroke type (1-Acute strokes, 2-TIA, 3-Complex Neurological)
        """
        self.patient_count += 1
        trace(f"1. patient {self.patient_count}, type {patient_type} arrive at{self.env.now: 3f}")
        # create a new patient and pass in env and args
        new_patient = Patient(self.patient_count, patient_type, self.env, self.args)
        # keep a record of the patient for results calculation
        self.patients.append(new_patient)
        # init the ACU process for this patient
        self.env.process(new_patient.assessment())

    def type1(self):
        """
        Patient with type1 stroke (acute stroke),
        IAT is exponentially distributed
        """
        while True:
            # sampling inter-arrival time
            inter_arrival_time = self.args.arrival_dist1.sample()
            patient_type = 1
            yield self.env.timeout(inter_arrival_time)
            self.generate_new_arrival(patient_type)

    def type2(self):
        """
        Patient with type2 stroke (Transient Ischaemic Attack (TIA)),
        IAT is exponentially distributed
        """
        while True:
            # sampling inter-arrival time
            inter_arrival_time = self.args.arrival_dist2.sample()
            patient_type = 2
            yield self.env.timeout(inter_arrival_time)
            self.generate_new_arrival(patient_type)

    def type3(self):
        """
        Patient with type3 stroke (Complex Neurological),
        IAT is exponentially distributed
        """
        while True:
            # sampling inter-arrival time
            inter_arrival_time = self.args.arrival_dist3.sample()
            patient_type = 3
            yield self.env.timeout(inter_arrival_time)
            self.generate_new_arrival(patient_type)

    def raw_summary(self):
        """
        Store raw summary data per patient
        including patient_id, patient_type,
        arrival_time, time_to_beds,
        stay_in_hospital, and four_hour_target.

        Returns:
        --------
          raw_df: pandas.DataFrame
        """
        # append to results df
        pc = [i for i in range(1, self.patient_count+1)]
        ty = [pt.type for pt in self.patients]
        at = [pt.arrival_time for pt in self.patients]
        wt = [pt.time_to_bed for pt in self.patients]
        st = [pt.stay_duration for pt in self.patients]
        ft = [pt.four_hour_target for pt in self.patients]

        raw_df = pd.DataFrame({'patient_id': pc,
                               'patient_type': ty,
                               'arrival_time': at,
                               'time_to_beds': wt,
                               'stay_in_hospital': st,
                               'four_hour_target': ft})
        raw_df = raw_df[raw_df['arrival_time'] >= self.args.warm_up]
        return raw_df

    def run_summary_frame(self):
        """
        Store summary performance data
        including beds_util and percentage
        (pts being admitted to hospital
        within 4hrs).

        Returns:
        --------
          sum_df: pandas.DataFrame
        """
        raw_df = self.raw_summary()
        # adjust util calculations for warmup period
        rc_period = self.env.now - self.args.warm_up
        util = np.sum(raw_df['stay_in_hospital']) / (rc_period * self.args.n_beds)
        ratio = np.mean(raw_df['four_hour_target'])

        df = pd.DataFrame({'1': {'beds_util': util,
                                 'percentage': ratio}})
        sum_df = df.T
        sum_df.index.name = 'rep'
        return sum_df


def single_run(scenario,
               random_no_set=DEFAULT_RNG_SET):
    """
    Perform a single run of the model and return the results

    Parameters:
    -----------

    scenario: Scenario object
        The scenario/parameters to run

    random_no_set: int or None, optional (default=1)
        Controls the set of random seeds used by the stochastic parts of the
        model.  Set to different ints to get different results.  Set to None
        for a random set of seeds.

    Returns:
    --------
        pandas.DataFrame:
          results from single run.
    """

    # set random number set - this controls sampling for the run.
    scenario.set_random_no_set(random_no_set)

    # create an instance of the model
    model = AcuteStrokeUnit(scenario)

    model.run()

    # run the model
    results_summary = model.run_summary_frame()

    return results_summary


def multiple_replications(scenario,
                          n_jobs=-1):
    """
    Perform multiple replications of the model.

    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configure  the model

    n_jobs, int, optional (default=-1)
        No. replications to run in parallel.

    Returns:
    --------
    df_results: pandas.DataFrame
      performance metrics per replication
    """
    res = Parallel(n_jobs=n_jobs)(delayed(single_run)(scenario,
                                                      random_no_set=rep)
                                  for rep in range(DEFAULT_N_REPS))

    # format and return results in a dataframe
    df_results = pd.concat(res)
    df_results.index = np.arange(1, len(df_results) + 1)
    df_results.index.name = 'rep'
    return df_results


class WarmupAuditor:
    """
    Warmup Auditor for the model.

    Stores the cumulative means for:
    1. percentage of patients being admitted
       to hospital within 4-hour
    2. bed utilisation
    """

    def __init__(self, model):
        self.env = model.env
        self.model = model
        self.interval = DEFAULT_WARMUP_AUDIT_INTERVAL
        self.wait_for_beds = []
        self.beds_util = []
        self.percentage = []

    def run(self):
        """
        Run the audited model
        """
        # set up data collection for warmup variables.
        self.env.process(self.audit_model())
        self.model.run()

    def audit_model(self):
        """
        Audit the model at the specified intervals
        """
        for i in itertools.count():
            yield self.env.timeout(self.interval)

            # Performance metrics
            # calculate the utilisation metrics
            util = sum([pt.stay_duration for pt in self.model.patients]) / \
                (self.env.now * self.model.args.n_beds)
            ratio = sum([pt.four_hour_target for pt in self.model.patients]) / \
                len([pt.four_hour_target for pt in self.model.patients])

            # store the metrics
            self.beds_util.append(util)
            self.percentage.append(ratio)

    def summary_frame(self):
        """
        Return the audit observations in a summary dataframe

        Returns:
        -------
        df: pd.DataFrame
        """

        df = pd.DataFrame([self.beds_util,
                           self.percentage]).T
        df.columns = ['beds_util', 'percentage']

        return df


def warmup_single_run(scenario,
                      random_no_set=DEFAULT_RNG_SET):
    """
    Perform a single run of the model as part of the warm-up
    analysis.

    Parameters:
    -----------

    scenario: Scenario object
        The scenario/parameters to run

    Returns:
    --------
        Tuple:
        (bed_util, percentage)
    """
    # set random number set - this controls sampling for the run.
    scenario.set_random_no_set(random_no_set)

    # create an instance of the model
    model = AcuteStrokeUnit(scenario)

    # create warm-up model auditor and run
    audit_model = WarmupAuditor(model)
    audit_model.run()

    return audit_model.summary_frame()


# example solution
def warmup_analysis(scenario,
                    n_jobs=-1):
    """
    Conduct a warm-up analysis of key performance measures in the model.

    The analysis runs multiple replications of the model.
    In each replication a WarmupAuditor periodically takes observations
    of the following metrics:

    metrics included:
    1. percentage of patients being admitted
       to hospital within 4-hour
    2. Bed utilisation

    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configure the model
    n_jobs: int, optional (default=-1)
        Number of processors for parallel running of replications

    Returns:
    --------
    dict of pd.DataFrames where each dataframe related to a metric.
    Each column of a dataframe represents a replication and each row
    represents an observation.
    """
    res = Parallel(n_jobs=n_jobs)(delayed(warmup_single_run)(scenario,
                                                             random_no_set=rep)
                                  for rep in range(DEFAULT_N_REPS))

    # format and return results
    metrics = {'beds_util': [],
               'percentage': []}

    # preprocess results of each replication
    for rep in res:
        metrics['beds_util'].append(rep.beds_util)
        metrics['percentage'].append(rep.percentage)

    # cast to dataframe
    metrics['beds_util'] = pd.DataFrame(metrics['beds_util']).T
    metrics['percentage'] = pd.DataFrame(metrics['percentage']).T

    # index as obs number
    metrics['beds_util'].index = np.arange(1, len(metrics['beds_util']) + 1)
    metrics['percentage'].index = np.arange(1, len(metrics['percentage']) + 1)

    # obs label
    metrics['beds_util'].index.name = "audit"
    metrics['percentage'].index.name = "audit"

    # columns as rep number
    cols = [f'rep_{i}' for i in range(1, DEFAULT_N_REPS + 1)]
    metrics['beds_util'].columns = cols
    metrics['percentage'].columns = cols

    return metrics


# example answer
def get_scenarios(n_beds, admission_increase):
    """
    Creates a dictionary object containing
    objects of type `Scenario` to run.

    Params:
    --------
    n_beds: range
      number of beds in the ACU
    admission_increase: range
      x% increase in patients requiring an admission

    Returns:
    --------
    dict:
        Contains the scenarios for the model
    """
    scenarios = {'base': Scenario()}

    for n in n_beds:
        for rate in admission_increase:
            scenarios[f'bed_{n} with {rate*100}% increase admission'] = Scenario()
            scenarios[f'bed_{n} with {rate*100}% increase admission'].n_beds = n
            scenarios[f'bed_{n} with {rate*100}% increase admission'].mean_iat1 /= (1 + rate)
            scenarios[f'bed_{n} with {rate*100}% increase admission'].mean_iat2 /= (1 + rate)
            scenarios[f'bed_{n} with {rate*100}% increase admission'].mean_iat3 /= (1 + rate)

    return scenarios


def run_scenario_analysis(scenarios):
    """
    Run each of the scenarios for a specified results
    collection period, warmup and replications.
    """

    scenario_results = {}
    for sc_name, scenario in scenarios.items():
        replications = multiple_replications(scenario)

        # save the results
        scenario_results[sc_name] = replications

    print('Scenario analysis complete.')
    return scenario_results


def scenario_summary_frame(scenario_results):
    """
    Mean, 95%CI results for each performance measure by scenario

    Parameters:
    ----------
    scenario_results: dict
        dictionary of replications.
        Key identifies the performance measure

    Returns:
    -------
    summary: pd.DataFrame
    """
    columns = []
    summary = pd.DataFrame()
    for sc_name, replications in scenario_results.items():
        # calculate mean
        mean = replications.mean()

        # calculate se
        se = replications.sem()

        # calculate 95%CI
        upper = mean + 1.96*se
        upper = upper.add_prefix('upper_')
        lower = mean - 1.96*se
        lower = lower.add_prefix('lower_')
        mean = mean.add_prefix("mean_")

        # combine them into one dataframe
        stats = pd.concat([mean, upper, lower], axis=0)
        summary = pd.concat([summary, stats], axis=1)
        columns.append(sc_name)

    summary.columns = columns
    return summary


def sensitivity_scenarios(n_beds=None, admission_increase=None,
                          stay_length_increase=None, priority=None):
    """
        Creates a dictionary object containing
        objects of type `Scenario` for
        sensitivity analysis to run.

        Params:
        --------
        n_beds: range
          number of beds in the ACU
        admission_increase: range
          x% increase in patients requiring an admission
        stay_length_increase: range
          x% increase in patients staying in the hospital
        priority: bool
          treat patients based on type of disease (priority)

        Returns:
        --------
        dict:
            Contains the scenarios for the model
        """
    scenarios = {}

    if n_beds is not None:
        for n in n_beds:
            scenarios[f'bed_{n}'] = Scenario()
            scenarios[f'bed_{n}'].n_beds = n

    if admission_increase is not None:
        for rate in admission_increase:
            scenarios[f'{rate * 100}% change in admission'] = Scenario()
            scenarios[f'{rate * 100}% change in admission'].mean_iat1 /= (1 + rate)
            scenarios[f'{rate * 100}% change in admission'].mean_iat2 /= (1 + rate)
            scenarios[f'{rate * 100}% change in admission'].mean_iat3 /= (1 + rate)

    if stay_length_increase is not None:
        for p in stay_length_increase:
            scenarios[f'{p * 100}% in stay in hospital length'] = Scenario()
            scenarios[f'{p * 100}% in stay in hospital length'].mean_stay1 /= (1 + p)
            scenarios[f'{p * 100}% in stay in hospital length'].std_stay1 /= (1 + p)
            scenarios[f'{p * 100}% in stay in hospital length'].mean_stay2 /= (1 + p)
            scenarios[f'{p * 100}% in stay in hospital length'].std_stay2 /= (1 + p)
            scenarios[f'{p * 100}% in stay in hospital length'].mean_stay3 /= (1 + p)
            scenarios[f'{p * 100}% in stay in hospital length'].std_stay3 /= (1 + p)

    if priority is not None:
        if priority:
            scenarios[f'turn on priority'] = Scenario()
            scenarios[f'turn on priority'].prior = True

    return scenarios


def sensitivity_summary_frame(sensitivity_results):
    """
    Min & Max results for each performance measure by scenario

    Parameters:
    ----------
    sensitivity_results: dict
        dictionary of replications.
        Key identifies the performance measure

    Returns:
    -------
    summary: pd.DataFrame
    """
    summary = pd.DataFrame()
    for sc_name, replications in sensitivity_results.items():
        # calculate max & min
        maxi = replications.max()
        mini = replications.min()

        maxi = maxi.add_prefix("max_")
        mini = mini.add_prefix("min_")

        # combine them into one dataframe
        stats = pd.concat([maxi, mini], axis=0)
        summary = pd.concat([summary, stats], axis=1)

    df = pd.concat([summary.min(axis=1).iloc[2:4].reset_index(drop=True),
                    summary.max(axis=1).iloc[0:2].reset_index(drop=True)],
                   axis=1)
    df.index = ["beds_util", "percentage"]
    df.columns = ["min", "max"]
    df["width"] = df["max"] - df["min"]

    return df


def plot_tornado(results):
    """
    Plotting tornado graph for sensitivity analysis

    Parameters
    ----------
    results : dict
        summary table from sensitivity_summary_frame()

    Returns
    ----------
    ax, fig
    """
    # get base scenario
    args = Scenario()
    base = multiple_replications(args)
    base_value = np.mean(base, axis=0)

    labels = list(results.keys())
    x_text = ["bed utilisation", "percentage of patients being admitted within 4 hrs"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    for i in [0, 1]:
        data = np.array(list(results.values()))[:, i]
        # Plot Bars
        ax[i].barh(labels, width=data[:, 2], left=data[:, 0], height=0.5)
        # Add Zero Reference Line
        ax[i].axvline(base_value[i], linestyle='--', color='black', label="base")
        # X Axis
        ax[i].set_xlim(0, 1)
        ax[i].set_xlabel(x_text[i])
    fig.tight_layout()

    return fig, ax
