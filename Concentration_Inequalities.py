import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns

sns.set()


def KLDivergence(u, p):
    """Compute the Kullback Leibler Divergence

    Parameters
    ----------
    u: float

    p: floap
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
    # stats.binom.sf

    return sf


def PlotUB(t, p, start, stop, nb_points, epsilon, ax):
    """Display the evolution of the different UB in function of the number of data

    Paramters
    ---------
    t: float
        threshold, maximum accepted deviation

    p: float
        float between 0 and 1, probability of the success of the Bernoulli distribution

    epsilon: float
        upper bound of the probability of deviation

    start: int
        starting point of the range of n

    stop: int
        ending point of the range of n

    ax: numpy array
        vector containing the characteristics of the graphs

    Returns
    -------
    None
    """

    n_range = np.linspace(
        start=start, stop=stop, num=nb_points, endpoint=True, retstep=False
    )

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

    # mask_Hoeffding_ub = ma.masked_less_equal(Hoeffding_UB_list, 1e-16)
    # mask_CramerChernoff_ub = ma.masked_less_equal(Cramer_Chernoff_UB_list, 1e-16)
    # mask_TCL_ub = ma.masked_less_equal(TCL_UB_list, 1e-16)
    # mask_Binomial_ub = ma.masked_less_equal(Binomial_UB_list, 1e-16)

    ax.plot(
        n_range,
        np.log(Hoeffding_UB_list),
        ".--",
        color="g",
        markersize=8,
        label="Hoeffding UB : y = -2nt^2",
    )
    ax.plot(
        n_range,
        np.log(Cramer_Chernoff_UB_list),
        "+--",
        color="b",
        markersize=8,
        label="Cramer Chernoff UB : y = nKL(t + p)",
    )
    ax.plot(
        n_range,
        np.log(TCL_UB_list),
        "^--",
        color="r",
        markersize=8,
        label="TCL Approximation",
    )
    ax.plot(
        n_range,
        np.log(Binomial_UB_list),
        "v--",
        color="c",
        markersize=8,
        label="Exact Binomial Probability",
    )
    ax.set_xlabel("n")
    ax.set_ylabel("Log Probability Of Deviation")
    ax.set_title(
        "Parameters : p = "
        + " "
        + str(p)
        + " ,"
        + " t = "
        + str(t)
        + " ,"
        + "n_max = "
        + str(stop)
    )
    ax.axhline(y=np.log(epsilon), color="red")


def PlotUB_as_function_detection_threshold(u, p, start, stop, nb_points, epsilon, ax):
    """Display the evolution of the different UB in function of the number of data

    Paramters
    ---------
    u: float
        risk detection threshold

    p: float
        float between 0 and 1, probability of the success of the Bernoulli distribution

    epsilon: float
        upper bound of the probability of deviation

    start: int
        starting point of the range of n

    stop: int
        ending point of the range of n

    ax: numpy array
        vector containing the characteristics of the graphs

    Returns
    -------
    None
    """
    n_range = np.linspace(
        start=start, stop=stop, num=nb_points, endpoint=True, retstep=False
    )

    Cramer_Chernoff_UB_list = list()
    Hoeffding_UB_list = list()
    TCL_UB_list = list()
    Binomial_UB_list = list()

    for n in enumerate(n_range):
        t = u / np.sqrt(n[1])
        Hoeffding_ub = HoeffdingUB(n=n[1], t=t)
        CramerChernoff_ub = CramerChernoffUB(n=n[1], t=t, p=p)
        TCL_ub = TCLUpperBound(n=n[1], p=p, t=t)
        Cramer_Chernoff_UB_list.append(CramerChernoff_ub)
        Hoeffding_UB_list.append(Hoeffding_ub)
        TCL_UB_list.append(TCL_ub)
        Binomial_ub = BinomialSurvivalFunction(n[1], p, t)
        Binomial_UB_list.append(Binomial_ub)

    # mask_Hoeffding_ub = ma.masked_less_equal(Hoeffding_UB_list, 1e-16)
    # mask_CramerChernoff_ub = ma.masked_less_equal(Cramer_Chernoff_UB_list, 1e-16)
    # mask_TCL_ub = ma.masked_less_equal(TCL_UB_list, 1e-16)
    # mask_Binomial_ub = ma.masked_less_equal(Binomial_UB_list, 1e-16)

    ax.plot(
        n_range,
        np.log(Hoeffding_UB_list),
        ".--",
        color="g",
        markersize=8,
        label="Hoeffding UB : y = -2nt^2",
    )
    ax.plot(
        n_range,
        np.log(Cramer_Chernoff_UB_list),
        "+--",
        color="b",
        markersize=8,
        label="Cramer Chernoff UB : y = nKL(t + p)",
    )
    ax.plot(
        n_range,
        np.log(TCL_UB_list),
        "^--",
        color="r",
        markersize=8,
        label="TCL Approximation",
    )
    ax.plot(
        n_range,
        np.log(Binomial_UB_list),
        "v--",
        color="c",
        markersize=8,
        label="Exact Binomial Probability",
    )
    ax.set_xlabel("n")
    ax.set_ylabel("Log Probability Of Deviation")
    ax.set_title(
        "Parameters : p = "
        + " "
        + str(p)
        + " ,"
        + " u = "
        + str(u)
        + " ,"
        + "n_max = "
        + str(stop)
    )
    ax.axhline(y=np.log(epsilon), color="red")


def PlotNbDataFunctionOfLogEpsilon(t, p, start, stop, nb_points, ax):
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

    start: float
        starting point of the range of epsilon

    stop: float
        ending point of the range of epsilon

    nb_points: int
        nubmer of points of discretization in the interval of epsilon

    ax: numpy array
        vector containing the characteristics of the graphs

    Returns
    -------
    None
    """
    CramerChernoff_list = list()
    Hoeffding_list = list()
    TCL_list = list()

    epsilon = np.linspace(
        start=start, stop=stop, num=nb_points, endpoint=True, retstep=False
    )

    for idx, eps in enumerate(epsilon):
        n_CramerChernoff = CramerChernoffEstimationNbData(t, p, eps)
        CramerChernoff_list.append(n_CramerChernoff)
        n_Hoeffding = HoeffdingEstimationNbData(t, eps)
        Hoeffding_list.append(n_Hoeffding)
        n_TCL = TCLNBDataEstimationWithEquivalentQuantile(t, p, eps)
        TCL_list.append(n_TCL)

    ax.plot(
        np.log(epsilon),
        CramerChernoff_list,
        "+--",
        color="b",
        markersize=8,
        label="n Cramer Chernoff : y = - log(epsilon) / KL(t + p)",
    )
    ax.plot(
        np.log(epsilon),
        Hoeffding_list,
        ".--",
        color="g",
        markersize=8,
        label="n Hoeffding : y = - log(epsilon) / (2t^2)",
    )
    ax.plot(
        np.log(epsilon),
        TCL_list,
        "^--",
        color="r",
        markersize=8,
        label="n TCL : y = (2p(1 - p) / t^2) log(epsilon)",
    )
    ax.set_xlabel("log(epsilon)")
    ax.set_ylabel("n")
    ax.set_title("Parameters : p = " + str(p) + " , t = " + str(t))


def main():
    u = 0.1
    p = 0.01
    p1 = 0.5
    p2 = 0.1
    p3 = 0.01
    p4 = 0.001
    n = 10
    t = 0.01
    u1 = 0.9
    u2 = 0.1
    u3 = 0.01
    u4 = 0.001
    t1 = 0.5
    t2 = 0.1
    t3 = 0.01
    t4 = 0.001
    start = 100
    stop1 = 100
    stop2 = 100
    stop3 = 5000
    stop4 = 100000
    stop5 = 20000
    stop6 = 40000
    stop7 = 2000
    stop8 = 400
    stop = 100000
    stop_tmp = 1000
    stop9 = 2000
    stop10 = 60000

    start_epsilon = 0.001
    stop_epsilon = 0.9
    epsilon = 0.01

    # First part UB = f(n), fixed p, varying t
    fig, axlist = plt.subplots(2, 2)
    fig.suptitle(
        "Evolution of the Log Probability of Deviation in function of the number of observations for different deviations t"
    )
    axlist = axlist.flatten()
    PlotUB(
        t=t1,
        p=p,
        start=start,
        stop=stop_tmp,
        nb_points=n,
        epsilon=epsilon,
        ax=axlist[0],
    )

    PlotUB(
        t=t2,
        p=p,
        start=start,
        stop=stop_tmp,
        nb_points=n,
        epsilon=epsilon,
        ax=axlist[1],
    )

    PlotUB(
        t=t3, p=p, start=start, stop=stop, nb_points=n, epsilon=epsilon, ax=axlist[2]
    )

    PlotUB(
        t=t4, p=p, start=start, stop=stop, nb_points=n, epsilon=epsilon, ax=axlist[3]
    )
    axlist.flatten()[2].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=4)
    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3
    )
    plt.show()

    # Second part UB = f(n), fixed t, varying p

    fig, axlist = plt.subplots(2, 2)
    fig.suptitle(
        "Evolution of the Log Probability of Deviation in function of the number of observations for different level of risk p"
    )
    axlist = axlist.flatten()

    PlotUB(
        t=t, p=p1, start=start, stop=stop, nb_points=n, epsilon=epsilon, ax=axlist[0]
    )

    PlotUB(
        t=t, p=p2, start=start, stop=stop, nb_points=n, epsilon=epsilon, ax=axlist[1]
    )

    PlotUB(
        t=t, p=p3, start=start, stop=stop, nb_points=n, epsilon=epsilon, ax=axlist[2]
    )

    PlotUB(
        t=t, p=p4, start=start, stop=stop, nb_points=n, epsilon=epsilon, ax=axlist[3]
    )

    axlist.flatten()[2].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=4)
    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3
    )
    plt.show()

    fig, axlist = plt.subplots(2, 2)
    fig.suptitle(
        "Evolution of the log probability of deviation in function of the number of data n and the risk detection threshold"
    )
    axlist = axlist.flatten()

    PlotUB_as_function_detection_threshold(
        u=u1, p=p, start=start, stop=stop10, nb_points=n, epsilon=epsilon, ax=axlist[0]
    )

    PlotUB_as_function_detection_threshold(
        u=u2, p=p, start=start, stop=stop9, nb_points=n, epsilon=epsilon, ax=axlist[1]
    )

    PlotUB_as_function_detection_threshold(
        u=u3, p=p, start=start, stop=stop9, nb_points=n, epsilon=epsilon, ax=axlist[2]
    )

    PlotUB_as_function_detection_threshold(
        u=u4, p=p, start=start, stop=stop9, nb_points=n, epsilon=epsilon, ax=axlist[3]
    )

    axlist.flatten()[2].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=4)
    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3
    )
    plt.show()

    # fig, axlist = plt.subplots(2, 2)
    # fig.suptitle(
    #     "Evolution of the number of data n in function of the log probability of deviation epsilon for different deviations t"
    # )
    # axlist = axlist.flatten()

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t1, p=p, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[0]
    # )

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t2, p=p, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[1]
    # )

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t3, p=p, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[2]
    # )

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t4, p=p, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[3]
    # )

    # axlist.flatten()[2].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=4)
    # plt.subplots_adjust(
    #     left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3
    # )
    # plt.show()

    # fig, axlist = plt.subplots(2, 2)
    # fig.suptitle(
    #     "Evolution of the number of data n in function of the log probability of deviation epsilon for different level of risk p"
    # )
    # axlist = axlist.flatten()

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t, p=p1, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[0]
    # )

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t, p=p2, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[1]
    # )

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t, p=p3, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[2]
    # )

    # PlotNbDataFunctionOfLogEpsilon(
    #     t=t, p=p4, start=start_epsilon, stop=stop_epsilon, nb_points=n, ax=axlist[3]
    # )

    # axlist.flatten()[2].legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=4)
    # plt.subplots_adjust(
    #     left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3
    # )
    # plt.show()

    # PlotNbDataFunctionOfLogEpsilon(t, p)
    # PlotNbDataFunctionOfMaximalDeviation(p, epsilon)
    # PlotMaximalDeviationFunctionOfProbaOfDeviation(p, n)
    # PlotMaximalDeviationFunctionOfNbData(p, epsilon)


if __name__ == "__main__":
    # execute only if run as a script
    main()
