import unittest
import numpy as np

from neurokit2.ecg import ecg_simulate

from rlign.utils import Template, find_rpeaks
from rlign.utils import malik, qrs_off_to_t_off_karjalainen_to_percent, p_on_to_qrs_on_carrutheres
from rlign.utils import _detrend, _resample_signal, _check_3d_array


class UtilsTest(unittest.TestCase):

    def setUp(self):
        """
        Set default start setting for all tests.
        """
        self.ecg_500hz_10s = ecg_simulate(sampling_rate=500)
        self.ecg_1000hz_10s = ecg_simulate(sampling_rate=1000)

    def test_template(self):

        for sl in [10, 20, 50]:
            for sr in [100, 500, 1000]:
                template60 = Template(seconds_len=sl, sampling_rate=sr, template_bpm=60)
                template40 = Template(seconds_len=sl, sampling_rate=sr, template_bpm=40)
                self.assertLess(template60.intervals[0], template40.intervals[0])
                self.assertEqual(sum(np.array(template60.intervals)-template60.intervals[0]), 0)

    def test_find_rpeaks(self):
        r_peaks = find_rpeaks(self.ecg_500hz_10s, sampling_rate=500)
        r_peaks2 = find_rpeaks(self.ecg_1000hz_10s, sampling_rate=1000)
        self.assertEqual(len(r_peaks), len(r_peaks2))

        # Exception
        ret = find_rpeaks(0*self.ecg_500hz_10s, sampling_rate=500)
        self.assertIsNone(ret)

    def test_interval_length(self):
        offset = .5
        sampling_rate = 500
        template60 = Template(seconds_len=10, sampling_rate=sampling_rate, template_bpm=60, offset=offset)
        self.assertEqual(np.min(template60.intervals), sampling_rate)
        self.assertEqual(np.max(template60.intervals), sampling_rate)
        self.assertEqual(np.min(template60.rpeaks), int(offset*sampling_rate))
        self.assertEqual(np.min(np.diff(template60.rpeaks)), sampling_rate)
        self.assertEqual(np.max(np.diff(template60.rpeaks)), sampling_rate)

    def test_malik_ranges(self):
        # At 60 BPM (RR=1000ms), QT should be roughly 360-420ms
        qt = malik(60, type="avg")
        self.assertGreater(qt, 300)
        self.assertLess(qt, 500)
        
        # Test sex specific
        self.assertNotEqual(malik(70, type="male"), malik(70, type="female"))

    def test_fractions(self):
        # Fractions of RR interval should be between 0 and 1
        val_t = qrs_off_to_t_off_karjalainen_to_percent(60)
        val_p = p_on_to_qrs_on_carrutheres(60)
        self.assertTrue(0 < val_t < 1)
        self.assertTrue(0 < val_p < 1)

    def test_detrend(self):
        # Create a signal with a strong linear trend: y = 2x + 5
        x = np.arange(100)
        y = 2.0 * x + 5.0 + np.random.normal(0, 0.1, 100)
        detrended = _detrend(y.copy())
        # The slope of the detrended signal should be near 0
        slope = np.polyfit(x, detrended, 1)[0]
        self.assertAlmostEqual(slope, 0, places=1)

    def test_resample_with_nans(self):
        # Test that NaNs are interpolated before resampling
        sig = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        # This shouldn't crash and should return finite values
        resampled = _resample_signal(sig, 100, 200)
        self.assertFalse(np.isnan(resampled).any())

    def test_check_3d_array(self):
        with self.assertRaises(ValueError):
            _check_3d_array(np.random.randn(10, 10)) # 2D instead of 3D
        
        valid = np.random.randn(1, 1, 100)
        self.assertTrue(np.array_equal(_check_3d_array(valid), valid))