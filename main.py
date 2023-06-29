import numpy as np
import time

import likelihood as lh
import mc3.proposal as proposal
from mc3.sampler import PTSampler

import multiprocessing as mp
import warnings

import argparse


def logprior(*args, **kwargs):
    return 0


def main(parser):
    parser.add_argument('--use_gpu', type=int, default=1, help='Flag whether to use gpu; 0 - not use, 1 - use '
                                                               '(default = 1)')
    parser.add_argument('--resume', type=int, default=0, help='Flag whether to resume current run; 0 - start new, '
                                                              '1 - resume (default = 0)')
    parser.add_argument('--niter', type=int, default=10000, help='Number of iterations (default = 10000)')
    parser.add_argument('--n_chains', type=int, default=5, help='Number of chains (default = 5)')
    parser.add_argument('--maxT', type=int, default=50, help='Maximum temperature (default = 50)')

    args = parser.parse_args()

    use_gpu = True if args.use_gpu else False  # True if you want to use gpu for computing likelihood
    resume = True if args.resume else False

    # Initialize true parameters of the signal
    T = 0.733  # observation time (years)
    dt = 5.0  # time step (seconds)
    M = 507236.8057121273 * 1.2  # large mass (solar)
    a = 0.6  # spin / will be ignored in Schwarzschild waveform
    mu = 18.063091389346287  # small mass (solar)
    p0 = 10.752251937834985  # initial separation
    e0 = 0.3857270761433499  # initial eccentricity
    x0 = 0.7  # initial cosine of the inclination / will be ignored in Schwarzschild waveform
    qK = np.pi / 2 - 0.6275167236796371  # polar spin angle
    phiK = 1.7262549907689677  # azimuthal viewing angle
    qS = 2.17583780178878  # polar sky angle
    phiS = 0.7101021513597163  # azimuthal viewing angle
    dist = 1.4357198958825074 / 2  # distance
    Phi_phi0 = 0  # initial phase in phi
    Phi_theta0 = 1.7262549907689677  # initial phase in theta
    Phi_r0 = 0  # initial phase in r

    # Parallel tempering parameters
    Tmax = args.maxT
    Nchains = args.n_chains

    priors = [[M * 0.6, M * 1.4], [mu * 0.6, mu * 1.4], [0.1, 0.9], [10, 12], [0.05, 0.5], [-0.9, 0.9],
              [dist * 0.5, dist * 1.5], [-1.0, 1.0],
              [0.0, 2 * np.pi], [-1.0, 1.0], [0.0, 2 * np.pi]]

    calculator = lh.LikelihoodCalculator(Phi_phi0, Phi_theta0, Phi_r0, T, dt, priors, use_gpu=use_gpu)
    calculator.setup_with_sangria([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK])

    names = ["M", "mu", "spin", "p0", "e0", "x0", "dist", "cos(qS)", "phiS", "cos(qK)", "phiK"]

    S = PTSampler(Nchains, priors, calculator.loglikelihood, logprior, names, Tmax=Tmax, profiling=True)

    if not resume:
        # Set a starting point
        x0 = [[M, mu, a, p0, e0, x0, dist, np.cos(qS), phiS, np.cos(qK), phiK]
              for _ in range(Nchains)]  # start from truth
        S.set_starting_point(x0)

    SL = proposal.Slice(names).slice
    SC = proposal.SCAM(names).SCAM
    p_dict = [{SL: 40, SC: 70}] * Nchains
    S.set_proposals(p_dict)
    print("Sampler setup finished!")

    print("Available cores:", mp.cpu_count())

    if resume:
        path = "."
        filenames = [path + "/chain_" + str(i) + ".npy" for i in range(Nchains)]
        S.resume(filenames)

    start = time.time()
    niter = args.niter
    c = S.run_mcmc(niter, pSwap=0.95, printN=100, multiproc=True, n0_swap=500)
    end = time.time()
    print("Time to complete", niter, "iterations:", round((end - start) / 60, 3), "minutes")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    warnings.filterwarnings('ignore')
    mp.freeze_support()

    arg_parser = argparse.ArgumentParser(description='Main file to start a run')
    main(arg_parser)
