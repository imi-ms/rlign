"""
    MIT License
    Copyright (c) 2024 Lucas Bickmann, Lucas Plagwitz
"""

import logging
import numpy as np

from typing import Tuple
from scipy.signal import resample
from scipy.stats import theilslopes
from sklearn.utils import check_array
from neurokit2 import ecg_clean, ecg_peaks


class Template(object):

    def __init__(self,
                 seconds_len: float = 10,
                 sampling_rate: int = 500,
                 template_bpm: int = 60,
                 offset: float = 0.5):
        """
            Creates a binary template pattern for an ECG, specifically designed for positioning the QRST interval.
            This function is instrumental in establishing a standardized template that dictates the spatial
            arrangement of R-peaks within a cardiac cycle, tailored to a specific beats per minute (BPM) rate.

            Parameters:
                seconds_len:
                     Determines the duration of the ECG template in seconds.

                sampling_rate: The sampling rate for the template. This value must match the sampling rate
                    used for the input ECGs to ensure consistency and accuracy in the template creation.
                    The sampling rate determines the temporal resolution of the ECG data.

                template_bpm: The desired normalized BPM value for the template. This parameter sets the
                    heart rate around which the QRST pattern is structured, thereby standardizing the R-peak
                    positions according to a specific BPM.

                offset: The offset specifies the starting point for the first normalized QRS complex in the
                    template. In percentage of sampling_rate.


            Note:
                This class is integral to the ECG analysis process, providing a foundational template for
                subsequent signal processing and interpretation.
        """
        if not (0 <= offset < 1):
            raise ValueError("Parameter offset must be between 0 and 1.")
        offset = int(offset * sampling_rate)
        total_len = seconds_len * sampling_rate
        template_bpm_len = (template_bpm / 60) * seconds_len
        self.bpm = template_bpm

        # compute spacing between rpeaks
        template_intervals = [int(total_len / template_bpm_len)] * int(template_bpm_len)

        # get equally spaced r-peak positions
        template_rpeaks = np.cumsum([offset, *template_intervals])

        # remove overhanging intervals
        self.rpeaks = template_rpeaks[template_rpeaks < total_len]
        self.intervals = template_intervals[:len(template_rpeaks)]


def find_rpeaks(
            ecg_lead: np.ndarray,
            sampling_rate: int,
            neurokit_method: str = "neurokit",
            correct_artifacts: bool = True) -> Tuple[np.ndarray, dict]:
    """
        Internal function which calls neurokit for rpeak and qrs-complex computation.

        Parameters:
            ecg_lead: An array representing a single-lead ECG signal. The input is
                expected to be a one-dimensional array of voltage values over time, representing
                the electrical activity of the heart as captured by a specific ECG lead.

            sampling_rate: Defines the sampling rate for all ECG recordings.

            neurokit_method: Chooses the algorithm for R-peak detection from the
                NeuroKit package. Different algorithms may offer varying performance
                based on the ECG signal characteristics.

            correct_artifacts: If set to True, artifact correction is applied
                exclusively for R-peak detections, enhancing the accuracy of peak
                identification in noisy signals.

        Returns:
            (rpeaks, qrs_epochs): A pair of elements (rpeaks, qrs_epochs) representing
                the outcomes of the R-peak detection and QRS complex computation processes,
                respectively. If an error occurs during the processing, the function
                returns None, indicating a failure in signal analysis.
    """
    try:
        # clean the ecg as recommended by neurokit
        data_ = ecg_clean(
            ecg_lead,
            sampling_rate=sampling_rate
        )

        # caluclate rpeaks
        _, r_peaks = ecg_peaks(
            data_,
            sampling_rate=sampling_rate,
            method=neurokit_method,
            correct_artifacts=correct_artifacts
        )
        rpeaks = r_peaks['ECG_R_Peaks'].astype(np.int32)

    except Exception as e:
        logging.warning(f'Failure in neurokit: {e}\n')
        return None

    if not len(rpeaks):
        return None
    return rpeaks


def _resample_multichannel(
    xs: np.ndarray,
    num_target: int,
    axis: int = -1,
    window: str = "hamming"
) -> np.ndarray:
    """
    Vectorized multichannel resampling using SciPy FFT.

    Parameters:
        xs: ndarray
            The signal array.
        num_target : int
            The target number of samples to which the signal is resampled along
            the given axis.

        axis : int
            The axis along which the signal is resampled, i.e. the time axis
            (defaults to -1, the last dimension).

        window: str
            The Fourier-domain window applied by the FFT resampling to reduce
            spectral leakage (defaults to "hamming"). See
            scipy.signal.resample for accepted values.

    Returns:
        Array of the resampled signal values, with the specified axis resized
        to num_target and all other dimensions unchanged.

    Note:
        The method have been modified from wfdbs resample_multichan and resample_sig.

    """
    orig_len = xs.shape[axis]
    if orig_len == num_target:
        return xs
    if orig_len <= 0 or num_target <= 0:
        shape = list(xs.shape)
        shape[axis] = max(0, num_target)
        return np.empty(shape, dtype=xs.dtype)

    # Handle internal NaNs using fast 1D NumPy interpolation (bypassing pandas)
    if np.isnan(xs).any():
        xs = _nan_interpolate_nd(xs, axis=axis)

    return resample(xs, num=num_target, axis=axis, window=window)


def _nan_interpolate_nd(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Fast pure-NumPy 1D linear interpolation along an axis to fill NaNs."""
    arr = arr.copy()
    # Move target axis to the last dimension
    arr_swapped = np.moveaxis(arr, axis, -1)
    orig_shape = arr_swapped.shape
    arr_2d = arr_swapped.reshape(-1, orig_shape[-1])
    n_points = orig_shape[-1]
    x_grid = np.arange(n_points)

    for i in range(arr_2d.shape[0]):
        row = arr_2d[i]
        nans = np.isnan(row)
        if nans.any() and not nans.all():
            row[nans] = np.interp(x_grid[nans], x_grid[~nans], row[~nans])
        elif nans.all():
            row[:] = 0.0

    return np.moveaxis(arr_2d.reshape(orig_shape), -1, axis)


def _resample_signal(x, fs, fs_target):
    """
    Resample a signal to a different frequency.

    Parameters:
        x : ndarray
            Array containing the signal.
        fs : int, float
            The original sampling frequency.
        fs_target : int, float
            The target frequency.

    Returns:
        resampled_x : ndarray
            Array of the resampled signal values.

    Note:
        The method have been modified from wfdbs resample_multichan and resample_sig.
    """
    if fs == fs_target:
        return x

    new_length = int(x.shape[0] * (fs_target / fs))
    if new_length <= 0:
        return np.empty(0, dtype=x.dtype)

    nans = np.isnan(x)
    if nans.any():
        if not nans.all():
            x = x.copy().reshape(-1)
            idx = np.arange(len(x))
            x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
        else:
            return np.zeros(new_length, dtype=x.dtype)

    return resample(x, num=new_length, window="hamming")


def _check_3d_array(X):
    """
    Check that the input is a three-dimensional array.
    """
    if isinstance(X, np.ndarray) and X.ndim == 3 and np.issubdtype(X.dtype, np.floating):
        return X
    X = check_array(X, ensure_2d=False, allow_nd=True)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-dimensional (got {X.ndim}).")
    return X


def _detrend(source_ecg):
    x = np.arange(len(source_ecg))
    slope, intercept, _, _ = theilslopes(source_ecg, x, method="joint")
    source_ecg -= intercept + slope * x
    return source_ecg


def malik(bpm, type="avg"):
    """
    Estimate the QT interval duration based on heart rate using
    the linear QT-RR relationship described by Malik et al.

    The QT interval is modeled as a linear function of the RR interval
    with sex-specific parameters. For `type="avg"`, the mean of the
    male and female models is returned.

    Reference:
        Malik M, Färbom P, Batchvarov V, Hnatkova K, Camm AJ.
        Relation between QT and RR intervals is highly individual among
        healthy subjects: implications for heart rate correction of the QT interval.
        Heart. 2002;87:220-228.
    """
    RR = 60 / bpm
    if type == "avg":
        # return (malik(bpm, type="male") + malik(bpm, type="female")) / 2
        # = ((0.24 + 0.15 * RR) + (0.22 + 0.19 * RR)) / 2 = 0.23 + 0.17*RR
        return (0.23 + 0.17*RR) * 1000
    # only linear
    if type == "male":
        beta, alpha = 0.24, 0.15
    elif type == "female":
        beta, alpha = 0.22, 0.19
    else:
        raise ValueError(f"Unknown QT interval type: {type}")
    bpm = beta + alpha * RR
    return bpm * 1000

def qrs_off_to_t_off_karjalainen_to_percent(bpm):
    """
    Estimate the temporal distance from QRS offset to T-wave offset
    as a fraction of the RR interval.

    The QT interval (in ms) is estimated using the Malik QT-RR model
    with average population parameters. A fixed duration of 35 ms
    corresponding to the QR portion of the QRS complex is subtracted
    to approximate the interval from R-peak to T-wave offset.

    Assumptions:
    - Average QRS duration ~ 90 ms (QR ~ 35 ms, RS ~ 55 ms)
    - Sinus rhythm and physiological heart rates.
    - Population-average correction; individual variability is expected.
    """
    #RR = 60 / bpm * 1000
    #qt = malik(bpm, type="avg")
    #qt -= 35
    #return qt / RR

    return 0.00325 * bpm + 0.17


def p_on_to_qrs_on_carrutheres(bpm):
    """
    Estimate the temporal distance from P-wave onset to QRS onset
    as a fraction of the RR interval, based on a heart-rate-dependent
    PR interval model.

    The PR interval (in ms) is estimated according to:
        PR = -0.351 * HR + 176.7
    as reported in:
        Carruthers SG, McCall B, Cordell BA, Wu R.
        Relationships between heart rate and PR interval during
        physiological and pharmacological interventions.
        Br J Clin Pharmacol. 1987;23(3):259-265.

    A fixed offset of 35 ms is added to approximate the interval
    from P-wave onset to QRS onset.

    """
    #RR = (60 / bpm * 1000)
    #pr = -0.351 * bpm + 176.7
    #return (pr + 35) / RR

    return (-0.351 * bpm + 211.7) * (bpm / 60000.0)
