"""Guard against hidden-value leakage and silently masked OOV context."""
import copy
import unittest
import diffusion_baseline as D


class Vocabulary:
    stoi = {k:{'known':1, 'alpha':2, 'zulu':3} for k in ['species','ability','item','nature','move']}


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.team = [dict(species='known',ability='known',item='known',nature='known',
                          moves=['alpha','zulu','alpha','zulu'],evs={}) for _ in range(6)]

    def test_masked_move_value_does_not_determine_rank(self):
        a = D.encode_context(self.team,[[0,'moves',0]],Vocabulary())
        other = copy.deepcopy(self.team); other[0]['moves'][0]='zzzz-unseen-secret'
        self.assertEqual(a,D.encode_context(other,[[0,'moves',0]],Vocabulary()))

    def test_masked_whole_slot_is_independent_of_hidden_species(self):
        a = D.encode_context(self.team,[[0]],Vocabulary())
        other = copy.deepcopy(self.team); other[0]['species']='unseen'
        self.assertEqual(a,D.encode_context(other,[[0]],Vocabulary()))
        self.assertEqual(a[0][:8],[0]*8)

    def test_oov_fixed_context_is_explicitly_unsupported(self):
        self.team[1]['nature']='Rash'
        _,_,missing = D.encode_context(self.team,[[0,'item']],Vocabulary())
        self.assertEqual(missing,[dict(path=[1,'nature'],value='Rash')])
        self.assertFalse(D.encode_context(self.team,[[1,'nature']],Vocabulary())[2])

    def test_move_mapping_preserves_original_path(self):
        self.team[0]['moves']=['zulu','alpha','zulu','alpha']
        vals,paths,missing = D.encode_context(self.team,[[0,'moves',1]],Vocabulary())
        self.assertFalse(missing)
        self.assertEqual(paths[3:7],[[0,'moves',3],[0,'moves',0],[0,'moves',2],[0,'moves',1]])
        self.assertEqual(vals[3:7],[2,3,3,0])


if __name__=='__main__': unittest.main()
