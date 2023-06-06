# Imports
import copy

import cupy as cp
import numpy as np

import time

# Package required for EMRI waveforms
from few.waveform import Pn5AAKWaveform

# Package for LISA response with gpu
from fastlisaresponse import ResponseWrapper

# Imports from ldc package
from ldc.common.series import TDI
from ldc.lisa.noise import get_noise_model
from ldc.common.tools import window

# Hard-coding: cannot be stored inside of class as self.xp, because it breaks multiprocessing
# Need to be = cp if computing on gpu, = np if computing on cpu
XP = cp


# Helping functions

def inner_product(lhsA, lhsE, rhsA, rhsE, SA, df):
    return 4.0 * df * XP.sum(XP.real(lhsA * XP.conj(rhsA) + lhsE * XP.conj(rhsE)) / SA)


def fourier(data, dt, n=0):
    """
    params: data - list like with elements - arrays-like of equal size
    return: list - fourier transforms, frequencies
    """

    if n == 0:
        n = data[0].size

    for i in range(len(data)):
        data[i] = XP.fft.rfft(data[i], n)[1:]

    freq = XP.fft.rfftfreq(n, d=dt)[1:]  # cause we want freq[0] != 0
    return data, freq


def crop_data(data, freq, fmin, fmax):
    """
    params: data - list like with elements - arrays-like of equal size; freq - array-like of original frequencies;
    return: list - cropped data, cropped frequencies
    """
    if fmin == 0 and fmax == XP.inf:
        return data, freq

    n = freq.size
    imin, imax = 0, n - 1
    for i in range(n):
        if freq[i] > fmin:
            imin = i
            break
    for i in range(n - 1, -1, -1):
        if freq[i] < fmax:
            imax = i
            break
    for i in range(len(data)):
        data[i] = data[i][imin:imax]
    freq = freq[imin:imax]
    return data, freq


class LikelihoodCalculator:

    def __init__(self, Phi_phi0, Phi_theta0, Phi_r0, T, dt, priors, use_gpu=False):
        self.use_gpu = use_gpu

        self.Phi_phi0 = Phi_phi0
        self.Phi_theta0 = Phi_theta0
        self.Phi_r0 = Phi_r0
        self.T = T  # in years
        self.dt = dt  # in seconds

        self.priors = priors

        self.dA = None  # in freq. domain
        self.dE = None  # in freq. domain
        self.freq = None

        self.SA = None
        self.n = None  # size of original signal in time
        self.df = None

    def _response_wrapper_setup(self):

        # keyword arguments for inspiral generator (RunKerrGeneriXPn5Inspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(1e4),  # all the trajectories will be well under len = 1000
        }

        # keyword arguments for summation generator (AAKSummation)
        sum_kwargs = {
            "use_gpu": self.use_gpu,  # GPU is available for this type of summation
            "pad_output": True,
        }

        AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs,
                                            use_gpu=self.use_gpu)

        t0 = 20000.0  # time at which signal starts (chops off data at start of waveform where information is incorrect)

        # order of the lagrangian interpolation
        order = 25

        orbit_file_esa = "esa-trailing-orbits.h5"

        orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

        # 1st or 2nd or custom (see docs for custom)
        tdi_gen = "1st generation"

        index_lambda = 8
        index_beta = 7

        tdi_kwargs_esa = dict(
            orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
        )

        return ResponseWrapper(
            AAK_waveform_model,
            self.T,
            self.dt,
            index_lambda,
            index_beta,
            t0=t0,
            flip_hx=True,  # set to True if waveform is h+ - ihx
            use_gpu=self.use_gpu,
            remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
            is_ecliptic_latitude=False,  # False if using polar angle (theta)
            remove_garbage="zero",  # removes the beginning of the signal that has bad information
            **tdi_kwargs_esa
        )

    def _get_tdi(self, point):

        responseWrapper = self._response_wrapper_setup()

        tdi = responseWrapper(*point, Phi_phi0=self.Phi_phi0, Phi_theta0=self.Phi_theta0, Phi_r0=self.Phi_r0)

        # Windowing
        t = XP.arange(0, len(tdi[0]) * self.dt, self.dt)
        window_arr = window(t, xl=30000, kap=0.0005)

        A, E = [window_arr * tdi[i] for i in range(2)]

        return A, E

    def setup(self, point):
        """
        Use this setup function in order to have a pure EMRI source as a template, without "signal noise" (sangria)
        """

        A, E = self._get_tdi(point)
        self.n = A.size

        [self.dA, self.dE], self.freq = fourier([A, E], self.dt, self.n)
        [self.dA, self.dE], self.freq = crop_data([self.dA, self.dE], self.freq, 6e-4, 2e-2)  # For now hardcode

        self.df = self.freq[1] - self.freq[0]

        # Setup noise model
        noise = get_noise_model("SciRDv1", self.freq, wd=1)
        self.SA = noise.psd(self.freq)

        SN2 = inner_product(self.dA, self.dE, self.dA, self.dE, self.SA, self.df)
        print("Setup successful!")
        print("SNR of the original signal =", round(XP.sqrt(SN2), 3))

    def setup_with_sangria(self, point):
        """
        Use this setup function in order to have an EMRI source in a soup of different signals as a template,
        aka added to the sangria dataset
        """

        A, E = self._get_tdi(point)

        sangria_fn = "LDC2_sangria_training_v2.h5"
        tdi_ts = TDI.load(sangria_fn, name="obs/tdi")
        tdi_mbhb_ts = TDI.load(sangria_fn, name="sky/mbhb/tdi")

        tdi_ts -= tdi_mbhb_ts
        tdi_ts.XYZ2AET()

        self.n = tdi_ts['A'].values.size
        A = XP.concatenate([A, XP.zeros(self.n - A.size)])
        E = XP.concatenate([E, XP.zeros(self.n - E.size)])
        tdi_ts['A'].values += A
        tdi_ts['E'].values += E

        [self.dA, self.dE], self.freq = fourier([tdi_ts['A'].values, tdi_ts['E'].values], self.dt, self.n)
        [self.dA, self.dE], self.freq = crop_data([self.dA, self.dE], self.freq, 6e-4, 2e-2)

        self.df = self.freq[1] - self.freq[0]

        # Setup noise model
        noise = get_noise_model("sangria", self.freq, wd=1)
        self.SA = noise.psd(self.freq)

        print("Setup successful!")

    def loglikelihood(self, point, i=0, T=1):
        start = time.time()

        for n in range(len(self.priors)):
            if point[n] < self.priors[n][0] or point[n] > self.priors[n][1]:
                return -100000

        angle_point = copy.copy(point)

        angle_point[7] = XP.arccos(angle_point[7])
        angle_point[9] = XP.arccos(angle_point[9])

        try:
            A, E = self._get_tdi(angle_point)
        except ValueError:
            return -100000

        [hA, hE], freq = fourier([A, E], self.dt, self.n)
        [hA, hE], freq = crop_data([hA, hE], freq, 6e-4, 2e-2)

        result = inner_product(self.dA, self.dE, hA, hE, self.SA, self.df) - 0.5 * inner_product(hA, hE, hA, hE,
                                                                                                 self.SA, self.df)

        end = time.time()
        print("Time for computing likelihood =", round(end - start, 2), "seconds, T = " + str(T) + ", i =", i)

        return result
