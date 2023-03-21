"""
Module: distributions

Provides some distribution objects encapsulate a
numpy random number generator

"""
import numpy as np
from abc import ABC, abstractmethod
import math


def generate_seed_vector(one_seed_to_rule_them_all=42, size=20):
    """
    Return a controllable numpy array
    of integer seeds to use in simulation model.
    
    Values are between 1000 and 10^10
    
    Params:
    ------
    one_seed_to_rule_them_all: int, optional (default=42)
        seed to produce the seed vector
        
    size: int, optional (default=20)
        length of seed vector
    """
    rng = np.random.default_rng(seed=one_seed_to_rule_them_all)
    return rng.integers(low=1000, high=10**10, size=size)


class Distribution(ABC):
    """
    Distribution interface
    """
    @abstractmethod
    def sample(self, size=None):
        pass
        

class Bernoulli(Distribution):
    """
    Convenience class for the Bernoulli distribution.
    packages up distribution parameters, seed and random generator.
    """
    def __init__(self, p, random_seed=None):
        """
        Constructor
        
        Params:
        ------
        p: float
            probability of drawing a 1
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        self.rand = np.random.default_rng(seed=random_seed)
        self.p = p
        
    def sample(self, size=None):
        """
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        return self.rand.binomial(n=1, p=self.p, size=size)
    

class Exponential(Distribution):
    """
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    """
    def __init__(self, mean, random_seed=None):
        """
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the exponential distribution
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        self.rand = np.random.default_rng(seed=random_seed)
        self.mean = mean
        
    def sample(self, size=None):
        """
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        return self.rand.exponential(self.mean, size=size)
    

class Poisson(Distribution):
    """
    Convenience class for the poisson distribution.
    packages up distribution parameters, seed and random generator.
    """
    def __init__(self, mean, random_seed=None):
        """
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the poisson distribution
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        self.rand = np.random.default_rng(seed=random_seed)
        self.mean = mean
        
    def sample(self, size=None):
        """
        Generate a sample from the poisson distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        return self.rand.poisson(self.mean, size=size)


class Lognormal:
    """
    Encapsulates a log normal distribution
    """

    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean = mean of the log normal distribution
        stdev = standard dev of the log normal distribution
        """
        self.rand = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev ** 2)
        self.mu = mu
        self.sigma = sigma

    def normal_moments_from_lognormal(self, m, v):
        """
        Returns mu and sigma of normal distribution
        underlying a log normal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html

        Params:
        -------
        m = mean of log normal distribution
        v = variance of log normal distribution

        Returns:
        -------
        (float, float)
        """
        phi = math.sqrt(v + m ** 2)
        mu = math.log(m ** 2 / phi)
        sigma = math.sqrt(math.log(phi ** 2 / m ** 2))
        return mu, sigma

    def sample(self, size=None):
        """
        Sample from the normal distribution
        """
        return self.rand.lognormal(self.mu, self.sigma, size=size)

