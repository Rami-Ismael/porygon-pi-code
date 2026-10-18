import sys
import unittest
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
for m in ('corpus','encode'):   # another test may have imported the /tmp/vgc-pilot/src staging copy
    if m in sys.modules and Path(sys.modules[m].__file__).resolve().parent!=SRC.resolve():del sys.modules[m]
import numpy as np
from beartype import beartype
from jaxtyping import Int64, TypeCheckError, jaxtyped
from corpus import load_corpus
from encode import Vocab, NF


@jaxtyped(typechecker=beartype)
def ids(a) -> Int64[np.ndarray, "fields"]:
    return a


class ShapeCheckTests(unittest.TestCase):
    """jaxtyping with beartype rejects a return value whose dtype or rank breaks the annotation."""
    def test_one_dimensional_int64_passes(self):
        ids(np.zeros(48,dtype=np.int64))

    def test_wrong_dtype_raises(self):
        with self.assertRaises(TypeCheckError):ids(np.zeros(48,dtype=np.int32))

    def test_wrong_rank_raises(self):
        with self.assertRaises(TypeCheckError):ids(np.zeros((6,8),dtype=np.int64))

    def test_real_encoder_on_corpus_team(self):
        teams,_=load_corpus()
        x=Vocab(teams).encode(teams[0])
        self.assertEqual((x.dtype,x.shape),(np.int64,(6*NF,)))


if __name__=='__main__':
    unittest.main()
