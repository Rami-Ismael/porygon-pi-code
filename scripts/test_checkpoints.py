import sys
import tempfile
import unittest
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
import numpy as np
import pickle
import torch
from checkpoints import load_checkpoint


class LoadCheckpointTests(unittest.TestCase):
    """load_checkpoint refuses pickled objects beyond tensors unless the caller trusts the file."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.dir=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tensor_checkpoint_loads_by_default(self):
        path=self.dir/'tensors.pt'
        torch.save({'sd':{'w':torch.arange(3.)},'cfg':{'d':8},'itos':['a','b']},path)
        ck=load_checkpoint(path,map_location='cpu')
        self.assertTrue(torch.equal(ck['sd']['w'],torch.arange(3.)))
        self.assertEqual(ck['cfg'],{'d':8})

    def test_numpy_checkpoint_refused_by_default(self):
        path=self.dir/'numpy.pt'
        torch.save({'counts':np.arange(3)},path)
        with self.assertRaises(pickle.UnpicklingError):load_checkpoint(path,map_location='cpu')

    def test_numpy_checkpoint_loads_when_trusted(self):
        path=self.dir/'numpy.pt'
        torch.save({'counts':np.arange(3)},path)
        ck=load_checkpoint(path,map_location='cpu',trusted_objects=True)
        np.testing.assert_array_equal(ck['counts'],np.arange(3))


if __name__=='__main__':
    unittest.main()
