import sys
import unittest
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
for m in ('encode','fullteam','spreaddiffusion'):   # another test may have imported the /tmp/vgc-pilot/src staging copy
    if m in sys.modules and Path(sys.modules[m].__file__).resolve().parent!=SRC.resolve():del sys.modules[m]
from encode import Vocab
from fullteam import FullVocab
from spreaddiffusion import SpreadVocab


def team(species='Kangaskhan'):
    slot=dict(species=species,ability='Scrappy',item='Leftovers',nature='Jolly',
              moves=['Fake Out','Last Resort'],evs={'HP':32,'Atk':2,'Spe':32})
    return [dict(slot) for _ in range(6)]


class StrictEncodeTests(unittest.TestCase):
    """Every team encoder rejects an unknown value instead of encoding it as 0, the [MASK] token."""
    def encoders(self):
        corpus=[team()]
        return [Vocab(corpus),FullVocab(corpus),SpreadVocab(corpus)]

    def test_known_team_has_no_mask_token(self):
        for v in self.encoders():
            with self.subTest(type(v).__name__):self.assertFalse((v.encode(team())==0).any())

    def test_unknown_species_raises_naming_field_and_value(self):
        for v in self.encoders():
            with self.subTest(type(v).__name__):
                with self.assertRaisesRegex(ValueError,r"outside corpus vocabulary: .*'species', 'pikachu'"):
                    v.encode(team('Pikachu'))

    def test_unknown_spread_raises(self):
        v=FullVocab([team()]);t=team();t[0]=dict(t[0],evs={'HP':1})
        with self.assertRaisesRegex(ValueError,"'spread'"):v.encode(t)


if __name__=='__main__':unittest.main()
