
<img src='./logo.svg?sanitize=true' width=30%/>


# Rlign: R peak alignment and ECG transformation framework

This scikit-learn compatible framework `rlign` is designed to synchronize the temporal variations across ECG recordings. This alignment enables the direct application of simpler machine learning models, like support vector machines and logistic regression, on R-peak aligned ECG signals, bypassing the need for complex and potentially biased feature extraction and allowing for interpretable, efficient analysis with enhanced small sample size convergence. Moreover, the alignment facilitates clustering of ECG time series, overcoming the challenges posed by unaligned data, where clusters are obscured by temporal misalignments of cardiac cycles. `Rlign` can also be used for improved interpretability of CNNs by aggregating importance maps from Integrated Gradients across all instances of a data set, instead of only reviewing individual ECGs.


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

# You can update some configurations later on
template_ = rlign.Template(template_bpm=80)
normalizer.update_configuration(template=template_)

ecg_aligned_80hz = normalizer.transform(ecg)
```

### Configurations

* `sampling_rate`: Defines the sampling rate for all ECG recordings. 

* `template`: A template ECG created with `create_template()` method. This template is 
    used as a reference for aligning R-peaks in the ECG signals.

* `select_lead`: Specifies the lead (e.g., 'Lead II', 'Lead V1') for R-peak 
    and QRST point detection. Different leads can provide varying levels of 
    clarity for these features. Selection via channel numbers 0,1,... .

* `num_workers`: Determines the number of CPU cores to be utilized for 
    parallel processing. Increasing this number can speed up computations
    but requires more system resources.

* `neurokit_method`: Chooses the algorithm for R-peak detection from the 
    NeuroKit package. Different algorithms may offer varying performance 
    based on the ECG signal characteristics.

* `correct_artifacts`: If set to True, artifact correction is applied 
    exclusively for R-peak detections, enhancing the accuracy of peak
    identification in noisy signals.

* `scale_method`: Selects the scaling method from options like 'resampling' 
    or 'hrc'. This choice dictates the interval used for resampling
    the ECG signal, which can impact the quality of the processed signal.

* `remove_fails`: Determines the behavior when scaling is not possible. If
    set to True, the problematic ECG is excluded from the dataset. If False, 
    the original, unscaled ECG signal is returned instead.
    
* `median_beat`: Calculates the median from a set of aligned beats 
    and returns a single, representative beat.

## Citation
Please use the following citation:

## License
[MIT License](LICENSE.txt)