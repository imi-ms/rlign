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

        self.X = np.concatenate([ecg_simulate(sampling_rate=500, method="multileads").to_numpy().reshape(1,12,5000)]*3000, axis=0)
        self.X_single = np.concatenate([ecg_simulate(sampling_rate=500).reshape(1,1,5000)]*3000, axis=0)

    def test_multi_processing(self):
        normalizer_single_cpu = rlign.Rlign(num_workers=1, select_lead=0, template_bpm=40)
        normalizer_multiple_cpu = rlign.Rlign(num_workers=4, select_lead=0, template_bpm=40)

        start_time = time.time()
        normalizer_single_cpu.transform(self.X_single)
        end_time = time.time()
        diff_single = end_time - start_time

        start_time = time.time()
        normalizer_multiple_cpu.transform(self.X_single)
        end_time = time.time()
        diff_multiple = end_time - start_time

        self.assertLess(diff_multiple, diff_single)

    def test_scale_method(self):
        normalizer_hrc = rlign.Rlign(num_workers=1, select_lead=0, scale_method="hrc", template_bpm=40)
        X_trans = normalizer_hrc.transform(self.X[:10])
        self.assertEqual(X_trans.shape, (10, 12, 5000))

        with self.assertRaises(ValueError):
            rlign.Rlign(num_workers=1, select_lead=0, scale_method="equal")

    def test_zero(self):
        normalizer_hrc = rlign.Rlign(num_workers=1, select_lead=0, scale_method="hrc",
                                     remove_fails=False, template_bpm=40)
        X_trans = normalizer_hrc.transform(np.zeros((1, 12, 5000)))
        self.assertTrue(np.array_equal(X_trans, np.zeros((1, 12, 5000))))

        a = np.concatenate([np.zeros((1, 1, 5000)), self.X_single[:10]], axis=0)
        X_trans = normalizer_hrc.transform(a)
        self.assertEqual(len(X_trans), 11)
        self.assertEqual(normalizer_hrc.fails[0], 1)
        self.assertEqual(np.sum(normalizer_hrc.fails[1:]), 0)

        normalizer_hrc = rlign.Rlign(num_workers=1, select_lead=0, scale_method="hrc", remove_fails=True)
        a = np.concatenate([np.zeros((1, 1, 5000)), self.X_single[:10]], axis=0)
        X_trans = normalizer_hrc.transform(a)
        self.assertEqual(len(X_trans), 10)
        self.assertEqual(normalizer_hrc.fails, [1]+[0]*10)

    def test_agg_beat_modes(self):
        # Test 'mean'
        model = rlign.Rlign(agg_beat="mean")
        out = model.transform(self.X[:10])
        self.assertEqual(out.ndim, 3)

        # Test custom callable (e.g., standard deviation across beats)
        model = rlign.Rlign(agg_beat=np.std)
        out = model.transform(self.X[:10])
        self.assertEqual(out.ndim, 3)

        # Test 'list'
        model = rlign.Rlign(agg_beat="list")
        out = model.transform(self.X[:10])
        self.assertIsInstance(out, list)
        self.assertEqual(out[0].ndim, 2) # [Leads, Time, Beats]

        

    def test_invalid_init_params(self):
        # Test invalid scale method
        with self.assertRaises(ValueError):
            rlign.Rlign(scale_method="invalid_method")
        
        # Test invalid agg_beat method
        with self.assertRaises(ValueError):
            rlign.Rlign(agg_beat="invalid_agg")

        # Test identity scaling without agg_beat
        with self.assertRaises(ValueError):
            rlign.Rlign(scale_method="identity", agg_beat="none")

    def test_property_setters(self):
        model = rlign.Rlign(sampling_rate=500)
        original_template_id = id(model.template)
        
        # Changing a property should trigger _update_template
        model.sampling_rate = 1000
        self.assertNotEqual(id(model.template), original_template_id)
        self.assertEqual(model.template.rpeaks[1] - model.template.rpeaks[0], 1000)

    def test_update_configuration(self):
        model = rlign.Rlign(num_workers=1)
        model.update_configuration(num_workers=8, neurokit_method="kalidas")
        self.assertEqual(model.num_workers, 8)
        self.assertEqual(model.neurokit_method, "kalidas")
        
        with self.assertRaises(AssertionError):
            model.update_configuration(unknown_param=True)