import unittest
import time
import numpy as np
import rlign.rlign as rlign

from neurokit2.ecg import ecg_simulate


class UtilsTest(unittest.TestCase):

    def setUp(self):
        """
        Set default start setting for all tests.
        """
        self.ecg_500hz_10s = ecg_simulate(sampling_rate=500)
        self.ecg_1000hz_10s = ecg_simulate(sampling_rate=1000)

        self.X = np.concatenate([ecg_simulate(sampling_rate=500).reshape(1,1,5000)]*3000, axis=0)
        print(self.X.shape)

    def test_multi_processing(self):
        print(self.X.shape)


        normalizer_single_cpu = rlign.Rlign(num_workers=1, select_lead=0, template_bpm=40)
<<<<<<< HEAD
        normalizer_multiple_cpu = rlign.Rlign(num_workers=4, select_lead=0, template_bpm=40)
=======
        normalizer_multiple_cpu = rlign.Rlign(num_workers=8, select_lead=0, template_bpm=40)
>>>>>>> origin/main

        start_time = time.time()
        normalizer_single_cpu.transform(self.X)
        end_time = time.time()
        diff_single = end_time - start_time

        start_time = time.time()
        normalizer_multiple_cpu.transform(self.X)
        end_time = time.time()
        diff_multiple = end_time - start_time

        self.assertLess(diff_multiple, diff_single)

    def test_scale_method(self):
        normalizer_hrc = rlign.Rlign(num_workers=1, select_lead=0, scale_method="hrc", template_bpm=40)
        X_trans = normalizer_hrc.transform(self.X[:10])
        self.assertEqual(X_trans.shape, (10, 1, 5000))

        with self.assertRaises(ValueError):
            rlign.Rlign(num_workers=1, select_lead=0, scale_method="equal")

    def test_zero(self):
        normalizer_hrc = rlign.Rlign(num_workers=1, select_lead=0, scale_method="hrc",
                                     remove_fails=False, template_bpm=40)
        X_trans = normalizer_hrc.transform(np.zeros((1, 12, 5000)))
        self.assertTrue(np.array_equal(X_trans, np.zeros((1, 12, 5000))))

        a = np.concatenate([np.zeros((1, 1, 5000)), self.X[:10]], axis=0)
        X_trans = normalizer_hrc.transform(a)
        self.assertEqual(len(X_trans), 11)
        self.assertEqual(normalizer_hrc.fails, [])
        self.assertEqual(np.sum(normalizer_hrc.fails[1:]), 0)

        normalizer_hrc = rlign.Rlign(num_workers=1, select_lead=0, scale_method="hrc", remove_fails=True)
        a = np.concatenate([np.zeros((1, 1, 5000)), self.X[:10]], axis=0)
        X_trans = normalizer_hrc.transform(a)
        self.assertEqual(len(X_trans), 10)
        self.assertEqual(normalizer_hrc.fails, [1])
