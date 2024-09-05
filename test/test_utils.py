import unittest
import numpy as np

from neurokit2.ecg import ecg_simulate

from rlign.utils import Template, find_rpeaks


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
                self.assertEquals(sum(np.array(template60.intervals)-template60.intervals[0]), 0)

    def test_find_rpeaks(self):
        r_peaks = find_rpeaks(self.ecg_500hz_10s, sampling_rate=500)
        r_peaks2 = find_rpeaks(self.ecg_1000hz_10s, sampling_rate=1000)
        self.assertEquals(len(r_peaks), len(r_peaks2))

        # Exception
        ret = find_rpeaks(0*self.ecg_500hz_10s, sampling_rate=500)
        self.assertIsNone(ret[0])
        self.assertIsNone(ret[1])

    def test_interval_length(self):
        offset = .5
        sampling_rate = 500
        template60 = Template(seconds_len=10, sampling_rate=sampling_rate, template_bpm=60, offset=offset)
        self.assertEquals(np.min(template60.intervals), sampling_rate)
        self.assertEquals(np.max(template60.intervals), sampling_rate)
        self.assertEquals(np.min(template60.rpeaks), int(offset*sampling_rate))
        self.assertEquals(np.min(np.diff(template60.rpeaks)), sampling_rate)
        self.assertEquals(np.max(np.diff(template60.rpeaks)), sampling_rate)
