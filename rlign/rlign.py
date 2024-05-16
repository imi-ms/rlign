"""
    MIT License
    Copyright (c) 2024 Lucas Bickmann, Lucas Plagwitz
"""

import multiprocessing as mp
from typing import Optional

import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

from rlign.utils import Template, find_rpeaks, _resample_multichannel, _check_3d_array


class Rlign(BaseEstimator, TransformerMixin, auto_wrap_output_keys=None):
    """
    Sets up the initial configuration for ECG signal processing,
    allowing for customization of various aspects such as sampling rate, template
    ECG, lead selection, and artifact correction.

    Parameters:
        sampling_rate: Defines the sampling rate for all ECG recordings. 

        template: Path or identifier for the template ECG. This template is 
            used as a reference for identifying R-peaks in the ECG signals.

        select_lead: Specifies the lead (e.g., 'Lead II', 'Lead V1') for R-peak 
            and QRST point detection. Different leads can provide varying levels of 
            clarity for these features. Selection via channel numbers 0,1,... .

        num_workers: Determines the number of CPU cores to be utilized for 
            parallel processing. Increasing this number can speed up computations
            but requires more system resources.

        neurokit_method: Chooses the algorithm for R-peak detection from the 
            NeuroKit package. Different algorithms may offer varying performance 
            based on the ECG signal characteristics.

        correct_artifacts: If set to True, artifact correction is applied 
            exclusively for R-peak detections, enhancing the accuracy of peak
            identification in noisy signals.

        scale_method: Selects the scaling method from options 'identity', 'linear' 
            or 'hrc'. This choice dictates the interval used for resampling
            the ECG signal, which can impact the quality of the processed signal.

        remove_fails: Determines the behavior when scaling is not possible. If
            set to True, the problematic ECG is excluded from the dataset. If False, 
            the original, unscaled ECG signal is returned instead.
            
        median_beat: Calculates the median from a set of aligned beats and returns
            a single, representative beat.

        silent: Disable all warnings.
    """

    __allowed = ("sampling_rate", "template", "resample_method",
                 "select_lead", "num_workers", "neurokit_method",
                 "correct_artifacts", "scale_method")

    def __init__(
            self, 
            sampling_rate: int = 500,
            template: Template = None,
            select_lead: int = 1,
            num_workers: int = 8,  # only applies for multiprocessing
            neurokit_method: str = 'neurokit',
            correct_artifacts: bool = True,
            scale_method: str = 'hrc',
            remove_fails: bool = False,
            median_beat: bool = False,
            silent: bool = True
    ):
        self.sampling_rate = sampling_rate
        if template:
            self.template = template
        else:
            self.template = Template()
        self.select_lead = select_lead
        self.num_workers = num_workers
        self.neurokit_method = neurokit_method
        self.correct_artifacts = correct_artifacts
        self.remove_fails = remove_fails
        self.fails = []
        self.median_beat = median_beat

        available_scale_methods = ['identity', 'linear', 'hrc']
        if scale_method in available_scale_methods:
            if scale_method == "identity":
                if not self.median_beat:
                    raise ValueError(f'Scaling method "identity" only works with median_beat==True')
            self.scale_method = scale_method
        else:
            raise ValueError(f'No such scaling method, '
                             f'please use one of the following: {available_scale_methods}')

        if silent:
            warnings.filterwarnings("ignore")

    def update_configuration(
            self,
            **kwargs
    ):
        """
        Enables modification of existing configuration settings as required.

        Parameters:
            **kwargs: Accepts any key-value pairs as arguments, similar to those
                in the init function, allowing for flexible configuration updates.

        """
        for k in kwargs.keys():
            assert (k in self.__class__.__allowed), f"Disallowed keyword passed: {k}"
            setattr(self, k, kwargs[k])

    def _normalize_interval(
        self,
        source_ecg: np.ndarray,
        source_rpeaks: np.ndarray,
        source_rpeaks_intervals: np.ndarray,
        fallback: Optional[bool] = False,
        hr: int = None,
    ) -> (np.ndarray, int):
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

        # get default definitions and create zeroed normalized array
        normalized_ecg = np.zeros(source_ecg.shape)
        n_channel = len(normalized_ecg)
        template_starts = self.template.rpeaks
        template_intervals = self.template.intervals

        # switch to defined resample method
        match scale_method:
            case 'linear':
                # limit number of peaks
                #source_rpeaks = source_rpeaks[:-1]

                medians = []
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

                   
                    if self.median_beat:
                        if (template_starts[idx] - int(self.template.intervals[0] / 3) < 0 or
                                template_starts[idx] + self.template.intervals[0] - int(self.template.intervals[0] / 3) > len(
                                    source_ecg[0])):
                            continue
                        else:
                            medians.append(normalized_ecg[:, template_starts[idx] - int(self.template.intervals[0] / 3):
                                                             template_starts[idx] + self.template.intervals[0] - int(
                                                                 self.template.intervals[0] / 3)].reshape((1, n_channel, -1)))

                if self.median_beat:
                    if len(medians):
                        normalized_ecg = np.median(np.concatenate(medians, axis=0), axis=0)
                    else:
                        return source_ecg[:, :self.template.intervals[0]], 1

            case "hrc":

                template_rr_dist = template_starts[2] - template_starts[1]
                dist_upper_template = int((self.template.bpm / 280+0.14) * template_rr_dist)
                dist_lower_template = int((-self.template.bpm / 330 + 0.96) * template_rr_dist)

                medians = []
                for idx, rpeak in enumerate(source_rpeaks[:-1]):

                    soruce_rr_dist = source_rpeaks[idx+1] - rpeak
                    dist_upper_or = int(np.clip((hr / 280 + 0.14), 0, 0.5) * soruce_rr_dist)
                    dist_lower_or = int(np.clip((-hr / 330 + 0.96), 0.6, 1) * soruce_rr_dist)

                    source_st = rpeak + dist_upper_or
                    source_sp = np.min([rpeak + dist_lower_or, 5000])
                    source_fs = source_sp - source_st

                    target_st = template_starts[idx] + dist_upper_template
                    target_sp = np.min([template_starts[idx] + dist_lower_template, 5000])
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
                            and source_rpeaks[idx+1]+overlap <= 5000:
                        normalized_ecg[:, target_sp:template_starts[idx+1]] = _resample_multichannel(
                            source_ecg[:, source_sp-overlap:source_rpeaks[idx+1]+overlap].transpose(1, 0),
                            source_rpeaks[idx+1] - source_sp+2*overlap,
                            template_starts[idx + 1] - target_sp+2*overlap,
                        ).transpose(1, 0)[:, overlap:-overlap]

                    if self.median_beat:
                        if (template_starts[idx] - int(self.template.intervals[0] / 3) < 0 or
                                template_starts[idx] + self.template.intervals[0] - int(self.template.intervals[0] / 3) >
                                len(source_ecg[0])) or idx == 0:
                            continue
                        else:
                            medians.append(normalized_ecg[:, template_starts[idx] - int(self.template.intervals[0] / 3):
                                                             template_starts[idx] + self.template.intervals[0] -
                                                             int(self.template.intervals[0] / 3)].reshape((1, n_channel, -1)))

                if self.median_beat:
                    if len(medians):
                        normalized_ecg = np.median(np.concatenate(medians, axis=0), axis=0)
                    else:
                        return source_ecg[:, :self.template.intervals[0]], 1

            case "identity":
                normalized_ecgs = []
                for idx, rpeak in enumerate(source_rpeaks[:]):
                    if rpeak < 200 or rpeak > len(source_ecg[0])-400:
                        continue
                    else:
                        normalized_ecgs.append(source_ecg[:, rpeak-200:rpeak+400].reshape((1, n_channel, 600)))
                normalized_ecg = np.median(np.concatenate(normalized_ecgs, axis=0), axis=0)
            case _:
                raise Exception("No such resampling method implemented!")
                
        return normalized_ecg, 0

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

    def _template_transform(
            self,
            ecg: np.ndarray,
    ):
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
        if all(element is None for element in rpeaks):
            if not self.remove_fails:
                if not self.median_beat:
                    return ecg, 1
                else:
                    if self.scale_method != "identity":
                        return ecg[:, :self.template.intervals[0]], 1
                    else:
                        return ecg[:, :600], 1
            else:
                return None, 1

        try:
            #hr = ecg_process(ecg_lead, sampling_rate=self.sampling_rate)[0]["ECG_Rate"].mean()
            
            # just some basic approximation
            min_factor = 60 / (ecg.shape[1] / self.sampling_rate)
            hr = int(len(rpeaks) * min_factor)
        except:
            if self.remove_fails:
                return None, 1
            elif self.median_beat and self.scale_method != "identity":
                return ecg[:, :self.template.intervals[0]], 1
            elif self.median_beat:
                return ecg[:, :600], 1
            
            return ecg, 1                


        # limit number of rpeaks to the targets
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

    def transform(
            self,
            X: np.ndarray,
            y=None
    ) -> np.ndarray:
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
            args = [[d] for d in X]
            with mp.Pool(self.num_workers) as pool:
                results = pool.starmap(self._template_transform, args)
            self.fails = [res[1] for res in results]
            d = np.asarray([res[0] for res in results if not res[0] is None], dtype=np.float16)
        else:
            results = [self._template_transform(X[i]) for i in range(len(X))]
            self.fails = [res[1] for res in results]
            d = np.concatenate([res[0].reshape((1, len(X[0]), -1)) for res in results if not res[0] is None])

        return d
