
<img src='./logo.svg?sanitize=true' width=30%/>


# Rlign: R peak alignment and ECG transformation framework

This scikit-learn compatible framework `rlign` is designed to synchronize the temporal variations across ECG recordings. This alignment enables the direct application of simpler machine learning models, like support vector machines and logistic regression, on R-peak aligned ECG signals, bypassing the need for complex and potentially biased feature extraction and allowing for interpretable, efficient analysis with enhanced small sample size convergence. Moreover, the alignment facilitates clustering of ECG time series, overcoming the challenges posed by unaligned data, where clusters are obscured by temporal misalignments of cardiac cycles. `Rlign` can also be used for improved interpretability of CNNs by aggregating importance maps from Integrated Gradients across all instances of a data set, instead of only reviewing individual ECGs.
For more in-depth insights, please refer to our paper available on [arXiv](https://arxiv.org/abs/2407.15555).

## Installation
### From PyPI
    pip install rlign

### From source
    git clone https://github.com/imi-ms/rlign.git
    cd rlign
    pip install .

## Quick start
1. Install Rlign
2. Import `Rlign` from this package with the corresponding `sampling_rate` of your data.
3. Call `transform` for ECGs with a numpy array of `[samples, channels, len]`.

### Examples
You can check out full example notebooks in the [example](./examples/) folder.
```python
import rlign

# Create a Normalizer
normalizer = rlign.Rlign(scale_method='hrc')

# call transform with an ecg 
# Input shape has to be (samples, channels, len)
ecg_aligned = normalizer.transform(ecg)

# You can set different configuration like median_beat-averaging or the template_bpm
normalizer = rlign.Rlign(scale_method='hrc', agg_beat='median', template_bpm=80)

ecg_aligned_80bpm = normalizer.transform(ecg)
```

### Configurations

* `sampling_rate`: Defines the sampling rate for all ECG recordings and the template. Default is set to 500.

* `seconds_len`: Determines the duration of all ECG recordings and the template in seconds. Default is 10 (sec).

* `template_bpm`: The desired normalized BPM value for the template. 
    This parameter sets the heart rate around which the QRST pattern 
    is structured, thereby standardizing the R-peak positions according to a specific BPM.

* `offset`: The offset specifies the starting point for the first normalized QRS complex in the 
    template. In percentage of sampling_rate. Default is set to 0.01.

* `select_lead`: Specifies the lead (e.g., 'Lead II', 'Lead V1') for R-peak detection. 
    Different leads can provide varying levels of clarity for these features. 
    Selection via channel numbers 0,1,... .

* `num_workers`: Determines the number of CPU cores to be utilized for 
    parallel processing. Increasing this number can speed up computations
    but requires more system resources.

* `neurokit_method`: Chooses the algorithm for R-peak detection from the 
    NeuroKit package. Different algorithms may offer varying performance 
    based on the ECG signal characteristics. Default is 'neurokit'.

* `correct_artifacts`: If set to True, artifact correction is applied 
    exclusively for R-peak detections, enhancing the accuracy of peak
    identification in noisy signals. Default is True.

* `scale_method`: Selects the scaling method from options like 'resampling' 
    or 'hrc'. This choice dictates the interval used for resampling
    the ECG signal, which can impact the quality of the processed signal.
    Default is 'hrc'.

* `remove_fails`: Determines the behavior when scaling is not possible. If
    set to True, the problematic ECG is excluded from the dataset. If False, 
    the original, unscaled ECG signal is returned instead. Default is False.
    
* `agg_beat`: Calculates the aggregated beat from a set of aligned beats 
    and returns a single, representative beat if using 'mean' or 'median'.
    'list' returns the additional dimension of aligned beats without per-lead
    aggregation. 'none' disables any aggregation and returns as a time-series.
    Also works with a lambda function, such as 'np.std'. Default is 'none'.

* `detrend`: Detrend each beat individually using the robust median of slopes.
    This is only computed if using `agg_beat`. Default is True.

* `silent`: Disable all warnings. Default is True.

## Citation
Please use the following citation:

```
@misc{plagwitz2024rlignalgorithmenhancedelectrocardiogram,
      title={The Rlign Algorithm for Enhanced Electrocardiogram Analysis through R-Peak Alignment for Explainable Classification and Clustering}, 
      author={Lucas Plagwitz and Lucas Bickmann and Michael Fujarski and Alexander Brenner and Warnes Gobalakrishnan and Lars Eckardt and Antonius Büscher and Julian Varghese},
      year={2024},
      eprint={2407.15555},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2407.15555}, 
}
```

## License
[MIT License](LICENSE.txt)