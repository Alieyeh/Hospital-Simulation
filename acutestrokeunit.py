import numpy as np
import pandas as pd
import itertools
import simpy
import matplotlib.pyplot as plt
import math
from scipy.stats import t
from joblib import Parallel, delayed
from distributions import (Exponential, Lognormal, Bernoulli, Poisson)

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

# scheduled audit intervals in minutes.
# AUDIT_FIRST_OBS = 10
# AUDIT_OBS_INTERVAL = 5

# default results collection period
DEFAULT_RESULTS_COLLECTION_PERIOD = 450

# default number of replications
DEFAULT_N_REPS = 5

# warmup
DEFAULT_WARMUP = 0


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

        # warm-up
        self.warm_up = DEFAULT_WARMUP

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
        self.arrival_dist1 = Exponential(MEAN_IAT1, random_seed=self.seeds[0])
        self.arrival_dist2 = Exponential(MEAN_IAT2, random_seed=self.seeds[1])
        self.arrival_dist3 = Exponential(MEAN_IAT3, random_seed=self.seeds[2])

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

        type: int
            stroke type

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
        self.beds = args.beds
        # self.triage_dist = args.triage_dist

        # inter-arrival distributions
        self.arrival_dist1 = args.arrival_dist1
        self.arrival_dist2 = args.arrival_dist2
        self.arrival_dist3 = args.arrival_dist3

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

        # request a bed
        with self.beds.request() as req:
            yield req

            # time to bed
            self.time_to_bed = self.env.now - self.arrival_time
            self.waiting_complete()

            # sample stay duration.
            if self.type == 1:
                self.stay_duration = self.stay_dist1.sample()
            elif self.type == 2:
                self.stay_duration = self.stay_dist2.sample()
            elif self.type == 3:
                self.stay_duration = self.stay_dist3.sample()

            yield self.env.timeout(self.stay_duration)

            self.treatment_complete()

            if self.time_to_bed <= 1/6:
                self.four_hour_target = 1

    def waiting_complete(self):
        trace(f'2. patient {self.identifier}, type {self.type} waiting for bed ended {self.env.now:.3f}; '
              + f'waiting time was {self.time_to_bed:.3f}')

    def treatment_complete(self):
        trace(f'3. patient {self.identifier}, type {self.type} staying in hospital ended {self.env.now:.3f}; '
              + f'stay length was {self.stay_duration:.3f}')


class AcuteStrokeUnit:
    """
    Model of ACU
    """

    def __init__(self, args):
        """

        Params:
        -------
        env: simpy.Environment

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
        args.beds = simpy.Resource(self.env, capacity=args.n_beds)

    def run(self, results_collection_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
            warm_up=DEFAULT_WARMUP):
        """
        Conduct a single run of the model in its current
        configuration

        run length = results_collection_period + warm_up

        Parameters:
        ----------
        results_collection_period, float, optional
            default = DEFAULT_RESULTS_COLLECTION_PERIOD

        warm_up, float, optional (default=0)
            length of initial transient period to truncate
            from results.

        Returns:
        --------
            None

        """
        # set up the arrival process
        self.env.process(self.type1())
        self.env.process(self.type2())
        self.env.process(self.type3())

        # run
        self.env.run(until=results_collection_period + warm_up)

    def type1(self):
        while True:
            inter_arrival_time = self.args.arrival_dist1.sample()
            patient_type = 1
            yield self.env.timeout(inter_arrival_time)
            self.patient_count += 1
            trace(f"1. patient {self.patient_count}, type {patient_type} arrive at{self.env.now: 3f}")
            # create a new minor patient and pass in env and args
            new_patient = Patient(self.patient_count, patient_type, self.env, self.args)
            # keep a record of the patient for results calculation
            self.patients.append(new_patient)
            # init the minor injury process for this patient
            self.env.process(new_patient.assessment())

    def type2(self):
        while True:
            inter_arrival_time = self.args.arrival_dist2.sample()
            patient_type = 2
            yield self.env.timeout(inter_arrival_time)
            self.patient_count += 1
            trace(f"1. patient {self.patient_count}, type {patient_type} arrive at{self.env.now: 3f}")
            # create a new minor patient and pass in env and args
            new_patient = Patient(self.patient_count, patient_type, self.env, self.args)
            # keep a record of the patient for results calculation
            self.patients.append(new_patient)
            # init the minor injury process for this patient
            self.env.process(new_patient.assessment())

    def type3(self):
        while True:
            inter_arrival_time = self.args.arrival_dist3.sample()
            patient_type = 3
            yield self.env.timeout(inter_arrival_time)
            self.patient_count += 1
            trace(f"1. patient {self.patient_count}, type {patient_type} arrive at{self.env.now: 3f}")
            # create a new minor patient and pass in env and args
            new_patient = Patient(self.patient_count, patient_type, self.env, self.args)
            # keep a record of the patient for results calculation
            self.patients.append(new_patient)
            # init the minor injury process for this patient
            self.env.process(new_patient.assessment())

    def arrivals_generator(self):
        """
        IAT is exponentially distributed

        Parameters:
        ------
        env: simpy.Environment

        args: Scenario
            Container class for model data inputs
        """
        # type1_dist = Poisson(1 / MEAN_IAT1, 1234).sample(DEFAULT_RESULTS_COLLECTION_PERIOD)
        # type2_dist = Poisson(1 / MEAN_IAT2, 1234).sample(DEFAULT_RESULTS_COLLECTION_PERIOD)
        # type3_dist = Poisson(1 / MEAN_IAT3, 1234).sample(DEFAULT_RESULTS_COLLECTION_PERIOD)
        # dist = []
        # for i, j, k in zip(type1_dist, type2_dist, type3_dist):
        #     type_1 = np.ones(i)
        #     type_2 = np.ones(j) * 2
        #     type_3 = np.ones(k) * 3
        #     day = list(np.concatenate((type_1, type_2, type_3), axis=0))
        #     if len(day) == 0:
        #         day = [0]
        #     dist.append(random.sample(day, len(day)))
        # patient_count = 0
        # for patient_count in itertools.count(start=1):
        # for day in dist:
        #     for patient in day:
        #     inter_arrival_time = np.nan
        #     patient_type = 1

            # if patient != 0:
            #     patient_count = patient_count + 1
            # if patient_type == 1:
            #     inter_arrival_time = self.args.arrival_dist1.sample()
            #     patient_type = 1
            # elif patient_type == 2:
            #     inter_arrival_time = self.args.arrival_dist2.sample()
            #     patient_type = 2
            # elif patient_type == 3:
            #     inter_arrival_time = self.args.arrival_dist3.sample()
            #     patient_type = 3
            # print(inter_arrival_time)
            # yield self.env.timeout(inter_arrival_time)
            #
            # trace(f'1. patient {patient_count}, type {patient_type} arrives at: {self.env.now:.3f}')

            # create a new minor patient and pass in env and args
            # new_patient = Patient(patient_count, patient_type, self.env, self.args)

            # keep a record of the patient for results calculation
            # self.patients.append(new_patient)

            # init the minor injury process for this patient
            # self.env.process(new_patient.assessment())
        self.env.process(self.type1())
        self.env.process(self.type2())
        self.env.process(self.type3())

    def run_summary_frame(self):

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
        raw_df = raw_df[raw_df['arrival_time'] > self.args.warm_up]

        # adjust util calculations for warmup period
        rc_period = self.env.now - self.args.warm_up
        util = np.sum(raw_df['stay_in_hospital']) / (rc_period * self.args.n_beds)
        mean_waiting = np.mean(raw_df['time_to_beds'])
        ratio = np.mean(raw_df['four_hour_target'])

        df = pd.DataFrame({'1': {'time_to_beds': mean_waiting,
                                 # 'beds_queue': self.operator_queue,
                                 'beds_util': util,
                                 'percentage': ratio}})
        df = df.T
        df.index.name = 'rep'
        return df


def single_run(scenario,
               rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
               warm_up=DEFAULT_WARMUP,
               random_no_set=DEFAULT_RNG_SET):
    """
    Perform a single run of the model and return the results

    Parameters:
    -----------

    scenario: Scenario object
        The scenario/parameters to run

    rc_period: int
        The length of the simulation run that collects results

    warm_up: int, optional (default=0)
        warm-up period in the model.  The model will not collect any results
        before the warm-up period is reached.

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

    model.run(results_collection_period=rc_period, warm_up=warm_up)

    # run the model
    results_summary = model.run_summary_frame()

    return results_summary


def multiple_replications(scenario,
                          rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
                          warm_up=DEFAULT_WARMUP,
                          n_reps=DEFAULT_N_REPS,
                          n_jobs=-1):
    """
    Perform multiple replications of the model.

    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configure  the model

    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.
        the number of minutes to run the model beyond warm up
        to collect results

    warm_up: float, optional (default=0)
        initial transient period.  no results are collected in this period

    n_reps: int, optional (default=DEFAULT_N_REPS)
        Number of independent replications to run.

    n_jobs, int, optional (default=-1)
        No. replications to run in parallel.

    Returns:
    --------
    List
    """
    res = Parallel(n_jobs=n_jobs)(delayed(single_run)(scenario,
                                                      rc_period,
                                                      warm_up,
                                                      random_no_set=rep)
                                  for rep in range(n_reps))

    # format and return results in a dataframe
    df_results = pd.concat(res)
    df_results.index = np.arange(1, len(df_results) + 1)
    df_results.index.name = 'rep'
    return df_results


class WarmupAuditor:
    """
    Warmup Auditor for the model.

    Stores the cumulative means for:
    1. operator waiting time
    2. nurse waiting time
    3. operator utilisation
    4. nurse utilitsation.
    """

    def __init__(self, model, interval=DEFAULT_WARMUP):
        self.env = model.env
        self.model = model
        self.interval = interval
        self.wait_for_beds = []
        self.beds_util = []

    def run(self, rc_period):
        """
        Run the audited model

        Parameters:
        ----------
        rc_period: float
            Results collection period.  Typically this should be many times
            longer than the expected results collection period.

        Returns:
        -------
        None.
        """
        # set up data collection for warmup variables.
        self.env.process(self.audit_model())
        self.model.run(rc_period, 0)

    def audit_model(self):
        """
        Audit the model at the specified intervals
        """
        for i in itertools.count():
            yield self.env.timeout(self.interval)

            # Performance metrics
            # calculate the utilisation metrics
            wait_for_beds = np.sum([pt.time_to_bed for pt in self.model.patients]) / \
                         self.model.args.patient_count
            util = np.sum([pt.stay_duration for pt in self.model.patients]) / \
                (self.env.now * self.model.args.n_beds)

            # store the metrics
            self.wait_for_beds.append(wait_for_beds)
            self.beds_util.append(util)

    def summary_frame(self):
        '''
        Return the audit observations in a summary dataframe

        Returns:
        -------
        pd.DataFrame
        '''

        df = pd.DataFrame([self.wait_for_operator,
                           self.wait_for_nurse,
                           self.operator_util,
                           self.nurse_util]).T
        df.columns = ['operator_wait', 'nurse_wait', 'operator_util',
                      'nurse_util']

        return df


def warmup_single_run(scenario, rc_period,
                      interval=DEFAULT_WARMUP_AUDIT_INTERVAL,
                      random_no_set=DEFAULT_RNG_SET):
    '''
    Perform a single run of the model as part of the warm-up
    analysis.

    Parameters:
    -----------

    scenario: Scenario object
        The scenario/paramaters to run

    results_collection_period: int
        The length of the simulation run that collects results

    audit_interval: int, optional (default=60)
        during between audits as the model runs.

    Returns:
    --------
        Tuple:
        (mean_time_in_system, mean_time_to_nurse, mean_time_to_triage,
         four_hours)
    '''
    # set random number set - this controls sampling for the run.
    scenario.set_random_no_set(random_no_set)

    # create an instance of the model
    model = UrgentCareCallCentre(scenario)

    # create warm-up model auditor and run
    audit_model = WarmupAuditor(model, interval)
    audit_model.run(rc_period)

    return audit_model.summary_frame()


# example solution
def warmup_analysis(scenario, rc_period, n_reps=DEFAULT_N_REPS,
                    interval=DEFAULT_WARMUP_AUDIT_INTERVAL,
                    n_jobs=-1):
    '''
    Conduct a warm-up analysis of key performance measures in the model.

    The analysis runs multiple replications of the model.
    In each replication a WarmupAuditor periodically takes observations
    of the following metrics:

    metrics included:
    1. Operator waiting time
    2. Nurse callback waiting time
    3. Operator utilisation
    4. Nurse utilisation

    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configurethe model

    rc_period: int
        number of minutes to run the model in simulated time

    n_reps: int, optional (default=5)
        Number of independent replications to run.

    n_jobs: int, optional (default=-1)
        Number of processors for parallel running of replications

    Returns:
    --------
    dict of pd.DataFrames where each dataframe related to a metric.
    Each column of a dataframe represents a replication and each row
    represents an observation.
    '''
    res = Parallel(n_jobs=n_jobs)(delayed(warmup_single_run)(scenario,
                                                             rc_period,
                                                             random_no_set=rep,
                                                             interval=interval)
                                  for rep in range(n_reps))

    # format and return results
    metrics = {'operator_wait': [],
               'nurse_wait': [],
               'operator_util': [],
               'nurse_util': []}

    # preprocess results of each replication
    for rep in res:
        metrics['operator_wait'].append(rep.operator_wait)
        metrics['nurse_wait'].append(rep.nurse_wait)
        metrics['operator_util'].append(rep.operator_util)
        metrics['nurse_util'].append(rep.nurse_util)

    # cast to dataframe
    metrics['operator_wait'] = pd.DataFrame(metrics['operator_wait']).T
    metrics['nurse_wait'] = pd.DataFrame(metrics['nurse_wait']).T
    metrics['operator_util'] = pd.DataFrame(metrics['operator_util']).T
    metrics['nurse_util'] = pd.DataFrame(metrics['nurse_util']).T

    # index as obs number
    metrics['operator_wait'].index = np.arange(1,
                                               len(metrics['operator_wait']) + 1)
    metrics['nurse_wait'].index = np.arange(1, len(metrics['nurse_wait']) + 1)
    metrics['operator_util'].index = np.arange(1,
                                               len(metrics['operator_util']) + 1)
    metrics['nurse_util'].index = np.arange(1, len(metrics['nurse_util']) + 1)

    # obs label
    metrics['operator_wait'].index.name = "audit"
    metrics['nurse_wait'].index.name = "audit"
    metrics['operator_util'].index.name = "audit"
    metrics['nurse_util'].index.name = "audit"

    # columns as rep number
    cols = [f'rep_{i}' for i in range(1, n_reps + 1)]
    metrics['operator_wait'].columns = cols
    metrics['nurse_wait'].columns = cols
    metrics['operator_util'].columns = cols
    metrics['nurse_util'].columns = cols

    return metrics


def time_series_inspection(results, warm_up=None):
    """
    Time series inspection method

    Parameters:
    ----------
    results: dict
        The dict of results taken from warmup_analysis
    """

    # create the 4 chart areas to plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 9))

    # take the mean of the columns for each metric and plot
    ax[0][0].plot(results['time_to_beds'].mean(axis=1))
    ax[0][1].plot(results['beds_util'].mean(axis=1))

    # set the label of each chart
    ax[0][0].set_ylabel('time_to_beds')
    ax[0][1].set_ylabel('beds_util')

    if warm_up is not None:
        # add warmup cut-off vertical line if one is specified
        ax[0][0].axvline(x=warm_up, color='red', ls='--')
        ax[0][1].axvline(x=warm_up, color='red', ls='--')

    return fig, ax
