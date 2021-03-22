import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm


def KLDivergence(u, p):
    """Compute the Kullback Leibler Divergence

    Parameters
    ----------
    u: float

    p: float
        float between 0 and 1, probability of success the Bernoulli distribution

    Returns
    -------

    hp: float
        KL Divergence

    """

    hp = (1 - u) * math.log((1 - u) / (1 - p)) + u * math.log(u / p)

    return hp


def CramerChernoffUB(n, t, p):
    """Compute the Cramer Chernoff upper bound of the probability that the sum
    of n Bernoulli random variables (empirical estimator of the VaR) might deviate 
    from its expectation (the real VaR) from more than t 

    Paramters
    ---------

    n: int
        number of data

    p: float
        float between 0 and 1, probability of success of the Bernoulli distribution 

    t: float
        threshold, maximum accepted deviation

    Returns
    -------

    UB: float
        Cramer-Chernoff Upper Bound 
        Upper Bound of the probability of deviation from the empirical estimator of the VaR 
        and its expectation (the theoretical VaR)
    """
    hp = KLDivergence(t + p, p)
    UB = math.exp(-n * hp)

    return UB


def HoeffdingUB(n, t):
    """Compute the Hoeffding upper bound of the probability that the sum
    of the n Bernoulli random variables (empirical estimator of the VaR) 
    might deviate from its expectation (the real VaR) from more than t 

    Paramters
    ---------

    n: int 
        number of data

    t: float
        threshold, maximum accepted deviation

    Returns
    -------

    UB: float
        Hoeffding Upper Bound 
        Upper Bound of the probability of deviation from the empirical estimator of the VaR 
        and its expectation (the theoretical VaR)
    """

    UB = math.exp(-2 * n * t ** 2)

    return UB


def CramerChernoffEstimationNbData(t, p, epsilon):
    """Estimate the number of data n required so that the probability that the sum 
    of the n Bernoulli random variables (empirical estimator of the VaR) might deviate 
    from its expectation from more than t might be lower than a given epsilon

    Parameters
    ---------

    t: float 
        threshold, maximum accepted deviation

    p: float
        float between 0 and 1, probability of the success of the Bernoulli distribution

    epsilon: float
        upper bound of the probability of deviation

    Returns
    -------

    n: float
        minimum number of data required with the Cramer Chernoff method 
        so that the probability that the empirical estimator of the VaR 
        might deviate from its expectation (the real VaR) from more than t 
        might be lower than a given epsilon
    """

    hp = KLDivergence(t + p, p)

    n = -math.log(epsilon) / hp

    return n


def HoeffdingEstimationNbData(t, epsilon):
    """Estimate the number of data n required so that the probability that the sum 
    of the n Bernoulli random variables (empirical estimator of the VaR) might deviate 
    from its expectation (the real VaR) from more that t might be lower than epsilon

    Paramters
    ---------

    t: float
        threshold, maximum accepted deviation

    epsilon: float
        upper bound of the probability of deviation

    Returns
    -------

    n: float
        minimum number of data required with the Hoeffding method 
        so that the probability that the empirical estimator of the VaR 
        might deviate from its expectation (real VaR) from more than t 
        might be lower than a given epsilon 

    """

    n = -math.log(epsilon) / (2 * t ** 2)

    return n


def TCLUpperBound(n, p, t):
    """Compute an equivalent, when n becomes large, but in the non-asymptotic case,
    of the probability that the sum of the n Bernoulli random variables (empirical estimator of the VaR)
    might deviate from its expectation (the real VaR) from more than t. 
    This equivalent is an approximation of the probability of deviation in the non-asymptotic case. 

    Paramters
    ---------

    n: float
        minimum number of data required with the TCL method 
        so that the equivalent of the probability of deviation 
        might be lower than a given epsilon 

    t: float
        threshold, maximum accepted deviation


    Returns
    --------

    epsilon: float
        upper bound of the probability of deviation

    """

    epsilon = stats.norm.cdf(-t * math.sqrt(n / (p * (1 - p))))

    return epsilon


def TCLEstimationNbData(t, p, epsilon):
    """Estimate the number of data n required so that the probability that the sum 
    of the n Bernouli random variables (empirical estimator of the VaR) might deviate 
    from its expectation (the real VaR) from more than t might be equal to epsilon 

    Paramters
    ---------
    t: float 
        threshold, maximum accepted deviation

    p: float
        float between 0 and 1, probability of the success of the Bernoulli distribution

    epsilon: float
        almost sure probability of deviation for a given t 


    Returns
    -------

    n: float 
        number of data required so that the probability of the sum
        of the n Bernoulli random variables (empirical estimator of the VaR) might deviate 
        from its expectation (the real VaR) from more than t might be lower than epsilon
    """
    t_epsilon = stats.norm.ppf(epsilon, loc=0, scale=1)
    n = p * (1 - p) * (t_epsilon ** 2) / (t ** 2)

    return n


def EquivalentQuantile(epsilon):
    """Compute the equivalent as epsilon goes to 0 of the quantile of the Binomial distribution

    Parameters
    ----------

    epsilon: float
        probability of the deviation between the empirical estimator of the VaR 
        (the sum of the n Bernoulli random variables) and its expectation (the real VaR)

    Returns
    --------

    q_epsilon: float
        equivalent of the quantile as epsilon goes to 0
    """

    q_epsilon = math.sqrt(-2 * np.log(epsilon))

    return q_epsilon


def TCLNBDataEstimationWithEquivalentQuantile(t, p, epsilon):
    """Compute, for the TCL method, the number of data required so that the probability that 
    the sum of the n Bernoulli random variables (empirical estimator of the VaR)
    might deviate from its expectation (the real VaR) from more than t might be lower than a given epsilon
    using an equivalent of the quantile of the Binomial distribution as epsilon goes to 0

    Parameters
    -----------
    t: float
        maximal accepted deviation

    p: float
        between 0 and 1, probability of success of the Bernoulli distribution

    epsilon: float
        probability that the sum of the n Bernoulli variables (empirical estimator of the VaR) 
        might deviate from its expecatation (the real VaR) from more than t

    Returns
    --------
    """

    q_epsilon = EquivalentQuantile(epsilon)
    n = p * (1 - p) * (q_epsilon ** 2) / (t ** 2)

    return n


def TCLMaximalDeviationEstimation(n, p, epsilon):
    """Compute the maximal accepted deviation between the empirical estimator of the VaR 
    and its expectation (the real VaR) given a number n of data, a probability p of success 
    of the Bernoulli distribution and a probability of deviation epsilon
    
    Paramters
    --------

    n: int
        number of data

    p: float 
        between 0 and 1, probability of sucess of the Bernoulli distribution

    epsilon: float
        probability of deviation of the empirical estimator of the VaR 
        (the sum of the n Bernoulli random variables) from its expectation (the real VaR)

    Returns
    ------

    t: float
        value of the maximal accepted deviation between the empirical estimator of the VaR 
        (the sum of the n Bernoulli random variables) and its expectation (the real VaR)
    """
    q_epsilon = stats.norm.ppf(1 - epsilon, loc=0, scale=1)
    t = math.sqrt(p * (1 - p) / n) * q_epsilon

    return t


def BinomialSurvivalFunction(n, p, t):
    """Compute the survival function of the binomial distribution

    Paramters
    ---------

    n: int
        number of data

    p: float 
        between 0 and 1, probability of success of the Bernoulli distribution

    t: float 
        maximal deviation

    Returns
    -------

    sf: float
        survival function of the Bernoulli distribution

    """
    print("n * (p + t) = ", n * (p + t))
    sf = 1 - stats.binom.cdf(n * (p + t), n, p)

    return sf


def PlotUB(t, p, epsilon):
    """Display the evolution of the different UB in function of the number of data

    Paramters
    ---------

    t: float
        threshold, maximum accepted deviation

    p: float
        float between 0 and 1, probability of the success of the Bernoulli distribution

    epsilon: float
        upper bound of the probability of deviation
    """

    n_range = np.linspace(start=0, stop=10000, num=100, endpoint=True, retstep=False)

    Cramer_Chernoff_UB_list = list()
    Hoeffding_UB_list = list()
    TCL_UB_list = list()
    Binomial_UB_list = list()

    for n in enumerate(n_range):
        Hoeffding_ub = HoeffdingUB(n=n[1], t=t)
        CramerChernoff_ub = CramerChernoffUB(n=n[1], t=t, p=p)
        TCL_ub = TCLUpperBound(n=n[1], p=p, t=t)
        Cramer_Chernoff_UB_list.append(CramerChernoff_ub)
        Hoeffding_UB_list.append(Hoeffding_ub)
        TCL_UB_list.append(TCL_ub)
        Binomial_ub = BinomialSurvivalFunction(n[1], p, t)
        Binomial_UB_list.append(Binomial_ub)

    plt.semilogy(n_range, Hoeffding_UB_list, label="Hoeffding UB")
    plt.semilogy(n_range, Cramer_Chernoff_UB_list, label="Chernoff UB")
    plt.semilogy(n_range, TCL_UB_list, label="TCL UB")
    plt.semilogy(n_range, Binomial_UB_list, label="Binomial UB")
    plt.xlabel("n (number of data)")
    plt.ylabel("Upper Bound")
    plt.title(
        "Evolution of the Upper Bound of the probability of deviation in function of the number of data (n)"
    )
    plt.legend()
    plt.show()


def PlotNbDataFunctionOfLogEpsilon(t, p):
    """Display the evolution of the number of data in function of the log(epsilon)
    For each method, the evolution of the number of data n is linear in function of the log(epsilon)
    with a slope specific to each method.

    Paramters
    ---------

    t: float
        maximal accepted deviation between the empirical estimator of the VaR (the sum of the n Bernoulli random variables)
        and its expectation (the real VaR)

    p: float
        between 0 and 1, probability of success of the Bernoulli distribution
    """

    epsilon = [0.1, 0.01, 0.001]
    CramerChernoff_list = list()
    Hoeffding_list = list()
    TCL_list = list()

    for idx, eps in enumerate(epsilon):
        n_CramerChernoff = CramerChernoffEstimationNbData(t, p, eps)
        CramerChernoff_list.append(n_CramerChernoff)
        n_Hoeffding = HoeffdingEstimationNbData(t, eps)
        Hoeffding_list.append(n_Hoeffding)
        n_TCL = TCLNBDataEstimationWithEquivalentQuantile(t, p, eps)
        TCL_list.append(n_TCL)

    plt.semilogx(epsilon, CramerChernoff_list, label="n Cramer Chernoff")
    plt.semilogx(epsilon, Hoeffding_list, label="n Hoeffding")
    plt.semilogx(epsilon, TCL_list, label="n TCL")
    plt.xlabel("log(epsilon)")
    plt.ylabel("n")
    plt.title(
        "Plot of the Number of data (n) in function of the log of the probability of deviation (epsilon)"
    )
    plt.legend()
    plt.show()


def PlotNbDataFunctionOfMaximalDeviation(p, epsilon):
    """Display the evolution of the number of data n in function of the maximal deviation t 

    Parameters
    ----------

    p: float
        probability of success of the Bernoulli distribution

    epsilon: float
        probability of deviation between the empirical estimator of the VaR (the sum of the n Bernoulli random variables) 
        and its expectation (the real VaR)
    """

    range_t = [0.1, 0.01, 0.001]
    CramerChernoff_list = list()
    Hoeffding_list = list()
    TCL_list = list()

    for idx, t in enumerate(range_t):
        n_CramerChernoff = CramerChernoffEstimationNbData(t, p, epsilon)
        CramerChernoff_list.append(n_CramerChernoff)
        n_Hoeffding = HoeffdingEstimationNbData(t, epsilon)
        Hoeffding_list.append(n_Hoeffding)
        n_TCL = TCLEstimationNbData(t, p, epsilon)
        TCL_list.append(n_TCL)

    plt.loglog(range_t, CramerChernoff_list, label="n Cramer Chernoff")
    plt.loglog(range_t, Hoeffding_list, label="n Hoeffding")
    plt.loglog(range_t, TCL_list, label="n TCL")
    plt.xlabel("t (maximal deviation)")
    plt.ylabel("n")
    plt.title(
        "Log-Log Plot of the Number of data in function of the maximal deviation t"
    )
    plt.legend()
    plt.show()


def PlotMaximalDeviationFunctionOfProbaOfDeviation(p, n):
    """Display the evolution of the maximal deviation t in function of the probability of deviation epsilon

    Parameters
    ----------

    p: float
        beetween 0 and 1, probability of success of the Bernoulli distribution

    n: int
        number of data
    """

    range_epsilon = [0.1, 0.01, 0.001]
    t_TCL_list = list()

    for idx, eps in enumerate(range_epsilon):
        t_TCL = TCLMaximalDeviationEstimation(n, p, eps)
        t_TCL_list.append(t_TCL)

    plt.plot(range_epsilon, t_TCL_list, label="TCL")
    plt.xlabel("epsilon (maximal probability of deviation)")
    plt.ylabel("t")
    plt.title("Maximal deviation in function of the maximal probability of deviation")
    plt.legend()
    plt.show()


def PlotMaximalDeviationFunctionOfNbData(p, epsilon):
    """Display the evolution of the maximal deviation t in function of the number of data n 

    Paramters
    ---------

    p: float
        between 0 and 1, probability of success of the Bernoulli distribution

    epsilon: float
        between 0 and 1, probability that the empirical estimator of the VaR (the sum of the n Bernoulli random variables) 
        might deviate from its expectation (the real VaR) from more than t
    """

    range_n = np.linspace(start=0, stop=1000, num=100, endpoint=True, retstep=False)

    t_TCL_list = list()

    for n in range_n:
        t_TCL = TCLMaximalDeviationEstimation(n, p, epsilon)
        t_TCL_list.append(t_TCL)

    plt.loglog(range_n, t_TCL_list, label="TCL")
    plt.xlabel("n (number of data)")
    plt.ylabel("t")
    plt.title("Log-Log plot of the maximal deviation in function of the number of data")
    plt.legend()
    plt.show()


def main():
    u = 0.1
    p = 0.001
    n = 10000
    t = 0.01
    hp = KLDivergence(t + p, p)
    epsilon = 0.01
    CramerChernoff_UB = CramerChernoffUB(n=n, t=t, p=p)
    Hoeffding_UB = HoeffdingUB(n=n, t=t)
    TCL_UB = TCLUpperBound(n=n, p=p, t=t)

    n_CramerChernoff = CramerChernoffEstimationNbData(t=t, p=p, epsilon=epsilon)
    n_Hoeffding = HoeffdingEstimationNbData(t=t, epsilon=epsilon)
    n_TCL = TCLEstimationNbData(t=t, p=p, epsilon=epsilon)

    t_TCL = TCLMaximalDeviationEstimation(n, p, epsilon)

    print("Cramer Chernoff UB = ", CramerChernoff_UB)
    print("Hoeffding UB = ", Hoeffding_UB)
    print("TCL UB = ", TCL_UB)
    print("hp = ", hp)
    print("n Cramer Chernoff = ", n_CramerChernoff)
    print("n Hoeffding = ", n_Hoeffding)
    print("n TCL = ", n_TCL)
    print("t TCL = ", t_TCL)

    PlotUB(t=t, p=p, epsilon=epsilon)
    PlotNbDataFunctionOfLogEpsilon(t, p)
    PlotNbDataFunctionOfMaximalDeviation(p, epsilon)
    PlotMaximalDeviationFunctionOfProbaOfDeviation(p, n)
    PlotMaximalDeviationFunctionOfNbData(p, epsilon)


if __name__ == "__main__":
    # execute only if run as a script
    main()
