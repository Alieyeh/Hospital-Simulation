import random

import numpy as np
import pandas as pd
import itertools
import simpy
import math
from scipy.stats import t
from joblib import Parallel, delayed
from simulation.distributions import (Exponential, Lognormal, Bernoulli, Poisson)

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
TRACE = True

# default random number SET
DEFAULT_RNG_SET = 1234
N_STREAMS = 6

# scheduled audit intervals in minutes.
AUDIT_FIRST_OBS = 10
AUDIT_OBS_INTERVAL = 5

# default results collection period
DEFAULT_RESULTS_COLLECTION_PERIOD = 365*5

# default number of replications
DEFAULT_N_REPS = 5

# warmup auditing
DEFAULT_WARMUP_AUDIT_INTERVAL = 120


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
        self.beds = N_BEDS

        # warm-up
        self.warm_up = 0.0

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
        self.stay_duration = 0.000
        self.time_to_bed = 0.000
        self.time_to_stay = 0.000
        self.four_hour_target = 0

    def assessment(self):
        """
        simulates the process for ACU

        1. request and wait for a bed
        2. treatment
        3. exit system

        """
        # record the time that patient entered the system
        arrival_time = self.env.now

        # request a bed
        with self.beds.request() as req:
            yield req

            trace(f'bed {self.identifier} at {self.env.now:.3f}')

            # time to bed
            self.time_to_bed = self.env.now - arrival_time
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

    def __init__(self, env, args):
        """

        Params:
        -------
        env: simpy.Environment

        args: Scenario
            container class for simulation model inputs.
        """
        self.env = env
        self.args = args
        self.init_model_resources(args)
        self.patients = []

    def init_model_resources(self, args):
        """
        Set up the simpy resource objects

        Params:
        ------
        args - Scenario
            Simulation Parameter Container
        """
        args.beds = simpy.Resource(self.env,
                                   capacity=args.beds)

    def run(self, results_collection_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
            warm_up=0):
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
        self.env.process(self.arrivals_generator())

        # run
        self.env.run(until=results_collection_period + warm_up)

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
        for patient_count in itertools.count(start=1):
        # for day in dist:
        #     for patient in day:
            inter_arrival_time = np.nan
            patient_type = 1

            # if patient != 0:
            #     patient_count = patient_count + 1
            if patient_type == 1:
                inter_arrival_time = self.args.arrival_dist1.sample()
                patient_type = 1
            elif patient_type == 2:
                inter_arrival_time = self.args.arrival_dist2.sample()
                patient_type = 2
            elif patient_type == 3:
                inter_arrival_time = self.args.arrival_dist3.sample()
                patient_type = 3
            print(inter_arrival_time)
            yield self.env.timeout(inter_arrival_time)

            trace(f'1. patient {patient_count}, type {patient_type} arrives at: {self.env.now:.3f}')

            # create a new minor patient and pass in env and args
            new_patient = Patient(patient_count, patient_type, self.env, self.args)

            # keep a record of the patient for results calculation
            self.patients.append(new_patient)

            # init the minor injury process for this patient
            self.env.process(new_patient.assessment())

