"""
    MIT License
    Copyright (c) 2024 Lucas Bickmann, Lucas Plagwitz
"""

import multiprocessing as mp
from typing import Callable, Optional, Union

import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

from rlign.utils import Template, find_rpeaks, _resample_multichannel, _check_3d_array, _detrend


class Rlign(BaseEstimator, TransformerMixin, auto_wrap_output_keys=None):
    """
    Sets up the initial configuration for ECG signal processing,
    allowing for customization of various aspects such as sampling rate, template
    ECG, lead selection, and artifact correction.

    Parameters:
        sampling_rate: Defines the sampling rate for all ECG recordings and the template.

        seconds_len: Determines the duration of all ECG recordings and the template in seconds.

        template_bpm: The desired normalized BPM value for the template. This parameter sets the
            heart rate around which the QRST pattern is structured, thereby standardizing the R-peak
            positions according to a specific BPM.

        offset: The offset specifies the starting point for the first normalized QRS complex in the
            template. In percentage of sampling_rate. Default is set to 0.01.

        select_lead: Specifies the lead (e.g., 'Lead II', 'Lead V1') for R-peak
            and QRST point detection. Different leads can provide varying levels of
            clarity for these features. Selection via channel numbers 0,1,... .

        num_workers: Determines the number of CPU cores to be utilized for
            parallel processing. Increasing this number can speed up computations
            but requires more system resources. Default is set to 4.

        neurokit_method: Chooses the algorithm for R-peak detection from the
            NeuroKit package. Different algorithms may offer varying performance
            based on the ECG signal characteristics.

        correct_artifacts: If set to True, artifact correction is applied
            exclusively for R-peak detections, enhancing the accuracy of peak
            identification in noisy signals.

        scale_method: Selects the scaling method from options 'identity', 'linear'
            or 'hrc'. This choice dictates the interval used for resampling
            the ECG signal, which can impact the quality of the processed signal.
            Default is 'hrc'.

        remove_fails: Determines the behavior when scaling is not possible. If
            set to True, the problematic ECG is excluded from the dataset. If False,
            the original, unscaled ECG signal is returned instead. Default is False.

        agg_beat: Calculates the aggregated beat from a set of aligned beats and returns
            a single, representative beat. Default is False.

        silent: Disable all warnings. Default True.
    """

    __allowed = ("sampling_rate", "resample_method", "select_lead",
                 "num_workers", "neurokit_method", "correct_artifacts",
                 "scale_method", "seconds_len", "template_bpm", "offset",
                 "silent", "agg_beat", "remove_fails")

    def __init__(
            self,
            sampling_rate: int = 500,
            seconds_len: int = 10,
            template_bpm: int = 60,
            offset: float = .01,
            select_lead: int = 1,
            num_workers: int = 4,
            neurokit_method: str = 'neurokit',
            correct_artifacts: bool = True,
            scale_method: str = 'hrc',
            remove_fails: bool = False,
            agg_beat: Union[str, Callable[[int], np.ndarray]] = "none",
            median_beat: str = "deprecated",
            detrend: bool = False,
            silent: bool = True
    ):
        self._sampling_rate = sampling_rate
        self._offset = offset
        self._template_bpm = template_bpm
        self._seconds_len = seconds_len

        self.template = Template(
            sampling_rate=self.sampling_rate,
            offset=self.offset,
            template_bpm=self.template_bpm,
            seconds_len=self.seconds_len
        )
        self.select_lead = select_lead
        self.num_workers = num_workers
        self.neurokit_method = neurokit_method
        self.correct_artifacts = correct_artifacts
        self.remove_fails = remove_fails
        self.fails = []
        self.silent = silent
        self.detrend = detrend

        if detrend:
            warnings.warn("Not yet stable, may lead to unknown artifacts.")

        available_scale_methods = ['identity', 'linear', 'hrc']
        available_agg_beat_methods = ['median', 'mean', 'list', 'none']

        if scale_method in available_scale_methods:
            if scale_method == "identity":
                if not self.agg_beat:
                    raise ValueError(f'Scaling method "identity" only works with agg_beat==True')
            self.scale_method = scale_method
        else:
            raise ValueError(f'No such scaling method, '
                             f'please use one of the following: {available_scale_methods}')

        if agg_beat in available_agg_beat_methods or callable(agg_beat):
            self.agg_beat = agg_beat

            # Deprecation Warning
            if median_beat != "deprecated":
                warnings.warn("The setting median_beat is deprecated and will be removed in future versions."
                              "Please use agg_beat with 'median' or a lambda function. "
                              "Will overwrite 'agg_beat' with 'median' for now.",
                              category=DeprecationWarning, stacklevel=2)
                self.agg_beat = "median"
            self.median_beat = "deprecated"

        else:
            raise ValueError(f'No such aggregated beat method, '
                             f'please use one of the following: {available_agg_beat_methods}')

        if self.silent:
            warnings.filterwarnings("ignore")

    def update_configuration(self, **kwargs):
        """
        Enables modification of existing configuration settings as required.

        Parameters:
            **kwargs: Accepts any key-value pairs as arguments, similar to those
                in the init function, allowing for flexible configuration updates.

        """
        for k in kwargs.keys():
            assert (k in self.__class__.__allowed), f"Disallowed keyword passed: {k}"
            setattr(self, k, kwargs[k])

    @property
    def offset(self):
        return self._offset

    @property
    def template_bpm(self):
        return self._template_bpm

    @property
    def seconds_len(self):
        return self._seconds_len

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @offset.setter
    def offset(self, val):
        self._offset = val
        self._update_template()

    @sampling_rate.setter
    def sampling_rate(self, val):
        self._sampling_rate = val
        self._update_template()

    @seconds_len.setter
    def seconds_len(self, val):
        self._seconds_len = val
        self._update_template()

    @template_bpm.setter
    def template_bpm(self, val):
        self._template_bpm = val
        self._update_template()

    def _update_template(self):
        self.template = Template(
            sampling_rate=self.sampling_rate,
            offset=self.offset,
            template_bpm=self.template_bpm,
            seconds_len=self.seconds_len
        )

    def _normalize_interval(
        self,
        source_ecg: np.ndarray,
        source_rpeaks: np.ndarray,
        source_rpeaks_intervals: np.ndarray,
        fallback: Optional[bool] = False,
        hr: int = None,
    ) -> tuple[np.ndarray, int]:
        """
        Normalizes each interval independently based on the scale_method between rpeaks.

        Parameters:
            source_ecg: The input-ecg which intervals should be normalized.
            source_rpeaks: The found rpeak positions in the source-ecg.
            source_rpeaks_intervals: The length of each interval between rpeaks.
            fallback: If Yes, the safety mode is initiated with the basic transformation 'linear'.
        Returns:
            The normalized ecg, based on the template with rescaled segments between rpeaks.
        """

        scale_method = 'linear' if fallback else self.scale_method

        # switch to defined resample method
        match scale_method:
            case 'linear':
                normalized_ecg = self._scale_linear(source_ecg, source_rpeaks, source_rpeaks_intervals)
            case "hrc":
                normalized_ecg = self._scale_hrc(source_ecg, source_rpeaks, hr)
            case "identity":
                normalized_ecgs = []
                for rpeak in source_rpeaks[:]:
                    if rpeak < 200 or rpeak > len(source_ecg[0])-400:
                        continue
                    else:
                        normalized_ecgs.append(source_ecg[:, rpeak-200:rpeak+400].reshape((1, len(source_ecg), 600)))
                normalized_ecg = np.median(np.concatenate(normalized_ecgs, axis=0), axis=0)
                return normalized_ecg
            case _:
                raise Exception("No such resampling method implemented!")

        return normalized_ecg, 0

    def _scale_hrc(self, source_ecg, source_rpeaks, hr):
        normalized_ecg = np.full(source_ecg.shape, fill_value=np.nan)
        n_channel = len(normalized_ecg)
        template_starts = self.template.rpeaks

        template_rr_dist = template_starts[2] - template_starts[1]
        dist_upper_template = int((self.template.bpm / 280+0.14) * template_rr_dist)
        dist_lower_template = int((-self.template.bpm / 330 + 0.96) * template_rr_dist)

        max_len = self.seconds_len * self.sampling_rate

        beats = []
        for idx, rpeak in enumerate(source_rpeaks[:-1]):

            soruce_rr_dist = source_rpeaks[idx+1] - rpeak
            dist_upper_or = int(np.clip((hr / 280 + 0.14), 0, 0.5) * soruce_rr_dist)
            dist_lower_or = int(np.clip((-hr / 330 + 0.96), 0.6, 1) * soruce_rr_dist)

            source_st = rpeak + dist_upper_or
            source_sp = np.min([rpeak + dist_lower_or, max_len])
            source_fs = source_sp - source_st

            target_st = template_starts[idx] + dist_upper_template
            target_sp = np.min([template_starts[idx] + dist_lower_template, max_len])
            target_fs = target_sp - target_st

            overlap = 2  # remove artifacts

            # resampling between T-offset - P-onset
            normalized_ecg[:, target_st:target_sp] = _resample_multichannel(
                    source_ecg[:, source_st-overlap:source_sp+overlap].transpose(1, 0),
                    source_fs+2*overlap,
                    target_fs+2*overlap
            ).transpose(1, 0)[:, overlap:-overlap]

            # resampling between R-peak - T-offset
            normalized_ecg[:, template_starts[idx]:target_st] = _resample_multichannel(
                source_ecg[:, rpeak-overlap: source_st+overlap].transpose(1, 0),
                source_st - rpeak+2*overlap,
                target_st - template_starts[idx]+2*overlap
            ).transpose(1, 0)[:, overlap:-overlap]

            # resampling between P-Onset - R-peak
            if source_rpeaks[idx+1] - source_sp > 10 and template_starts[idx + 1] - target_sp > 10 \
                    and source_rpeaks[idx+1]+overlap <= max_len:
                normalized_ecg[:, target_sp:template_starts[idx+1]] = _resample_multichannel(
                    source_ecg[:, source_sp-overlap:source_rpeaks[idx+1]+overlap].transpose(1, 0),
                    source_rpeaks[idx+1] - source_sp+2*overlap,
                    template_starts[idx + 1] - target_sp+2*overlap,
                ).transpose(1, 0)[:, overlap:-overlap]

            if target_st:
                normalized_ecg[:, target_st] = np.mean(normalized_ecg[:, [target_st-1, target_st+1]], axis=1)
            if target_sp:
                normalized_ecg[:, target_sp] = np.mean(normalized_ecg[:, [target_sp-1, target_sp+1]], axis=1)

            if self.agg_beat != "none":
                if (template_starts[idx] - int(self.template.intervals[0] / 3) < 0 or
                        template_starts[idx] + self.template.intervals[0] - int(self.template.intervals[0] / 3) >
                        len(source_ecg[0])) or idx == 0:
                    continue
                else:
                    beats.append(normalized_ecg[:, template_starts[idx] - int(self.template.intervals[0] / 3):
                                                        template_starts[idx] + self.template.intervals[0] -
                                                        int(self.template.intervals[0] / 3)].reshape((1, n_channel, -1)))

        if self.agg_beat != "none":
            return self._agg_beats(source_ecg, normalized_ecg, beats)
        else:
            return normalized_ecg

    def _scale_linear(self, source_ecg, source_rpeaks, source_rpeaks_intervals):
        normalized_ecg = np.full(source_ecg.shape, fill_value=np.nan)
        n_channel = len(normalized_ecg)
        beats = []
        template_starts = self.template.rpeaks
        template_intervals = self.template.intervals

        # iterate over all peaks
        for idx, (source_st, source_fs) in enumerate(zip(source_rpeaks, source_rpeaks_intervals)):
            # define stop of source
            source_sp = source_st + source_fs

            # define target points and target frequency
            target_st = template_starts[idx]
            target_fs = template_intervals[idx]
            target_sp = target_st + target_fs

            # call resampling
            normalized_ecg[:, target_st:target_sp] = _resample_multichannel(
                    source_ecg[:, source_st:source_sp].transpose(1, 0),
                    source_fs,
                    target_fs
            ).transpose(1, 0)

            if self.agg_beat != "none":
                if (template_starts[idx] - int(self.template.intervals[0] / 3) < 0 or
                        template_starts[idx] + self.template.intervals[0] - int(self.template.intervals[0] / 3) > len(
                            source_ecg[0])):
                    continue
                else:
                    beats.append(normalized_ecg[:, template_starts[idx] - int(self.template.intervals[0] / 3):
                                                        template_starts[idx] + self.template.intervals[0] - int(
                                                            self.template.intervals[0] / 3)].reshape((1, n_channel, -1)))

        if self.agg_beat != "none":
            return self._agg_beats(source_ecg, normalized_ecg, beats)
        else:
            return normalized_ecg

    def _agg_beats(self, source_ecg, normalized_ecg, beats):
        if len(beats):
            beats = np.nan_to_num(beats, nan=np.nanmean(beats, axis=(-1), keepdims=True))

            if self.detrend:
                beats = np.apply_along_axis(_detrend, axis=-1, arr=beats)

            match self.agg_beat:
                case "median":
                    normalized_ecg = np.nanmedian(np.concatenate(beats, axis=0), axis=0)
                case "mean":
                    normalized_ecg = np.nanmean(np.concatenate(beats, axis=0), axis=0)
                case "list":
                    beats = np.transpose(np.asarray(beats), (1, 2, 3, 0))
                    normalized_ecg = np.concatenate(beats, axis=-2)
                case _:
                    normalized_ecg = self.agg_beat(np.concatenate(beats, axis=0), axis=0)

            return normalized_ecg
        else:
            if self.agg_beat == "list":
                return np.expand_dims(source_ecg[:, :self.template.intervals[0]], axis=-1)
            else:
                return source_ecg[:, :self.template.intervals[0]]

    def fit(self, X: np.ndarray, y=None):
        """
        Scikit-learn compatible function.

        Parameters:
            X: All multilead ECGs which should be normalized with format [samples, leads, time].
                 Single ecgs have to be unsqueezed with a single dimension.
            y: The corresponding labels. Only exists for compatibility reasons.

        Returns:
            self
        """
        return self

    def _template_transform(self, ecg: np.ndarray):
        """
        Normalizes the ECG by recentering R-peaks at equally spaced intervals, as defined in the template.
        The QRST-complexes are added, and the baselines are interpolated to match the new connections between complexes.

        Parameters:
            ecg: The multilead ecg with format of [leads, time].

        Returns:
            A normalized ECG with equally spaced QRST-complexes based on the defined template.
        """

        ecg_lead = np.array(ecg[self.select_lead])
        rpeaks = find_rpeaks(ecg_lead, self.sampling_rate)
        if rpeaks is None or all(element is None for element in rpeaks):
            if not self.remove_fails:
                if self.agg_beat == "none":
                    return ecg, 1
                else:
                    if self.scale_method != "identity":
                        return ecg[:, :self.template.intervals[0]], 1
                    else:
                        return ecg[:, :600], 1
            else:
                return None, 1

        try:
            # just some basic heart rate approximation
            min_factor = 60 / (ecg.shape[1] / self.sampling_rate)
            hr = int(len(rpeaks) * min_factor)
        except:
            if self.remove_fails:
                return None, 1
            elif self.agg_beat != "none" and self.scale_method != "identity":
                return ecg[:, :self.template.intervals[0]], 1
            elif self.agg_beat != "none":
                return ecg[:, :600], 1

            return ecg, 1

        # limit number of R-peaks to the targets
        rpeaks = rpeaks[: len(self.template.rpeaks+1)]

        # get the interval lengths of found rpeaks
        rpeak_intervals = np.diff([*rpeaks], axis=0)

        # call the normalization function
        return self._normalize_interval(
                            ecg,
                            rpeaks,
                            rpeak_intervals,
                            hr=hr
                        )

    def transform(self, X: np.ndarray, y=None) -> Union[np.ndarray, list]:
        """
        Normalizes and returns the ECGs with multiprocessing.

        Parameters:
            X:   All multilead ECGs which should be normalized with format [samples, leads, time].
                 Single ecgs have to be unsqueezed with a single dimension.
            y: The corresponding labels. Only exists for compatibility reasons.

        Returns:
            The beat normalized ecgs.
        """
        X = _check_3d_array(X)
        # Apply multiprocessing
        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                results = pool.map(self._template_transform, [d for d in X])
        else:
            results = [self._template_transform(X[i]) for i in range(len(X))]

        self.fails = [res[1] for res in results]
        d = [res[0] for res in results if res[0] is not None]
        if self.agg_beat != "list":
            d = np.asarray(d)
        return d
