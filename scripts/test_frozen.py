import gzip
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import frozen as F


class FrozenCopyTests(unittest.TestCase):
    def test_every_copy_matches_its_recorded_hash(self):
        sums = F.recorded()
        self.assertTrue(sums)
        self.assertEqual({p.relative_to(F.FIXTURES).as_posix()[:-3] for p in F.FIXTURES.rglob('*.gz')}, set(sums))
        for rel, expected in sums.items():
            self.assertEqual(hashlib.sha256(F.read_bytes(rel)).hexdigest(), expected, rel)

    def test_copies_equal_the_originals_where_results_exist(self):
        present = [rel for rel in F.recorded() if (F.RESULTS / rel).exists()]
        for rel in present:
            self.assertEqual(F.read_bytes(rel), (F.RESULTS / rel).read_bytes(), rel)
        if not present:
            self.skipTest('no original results/ files in this checkout to compare with')

    def test_a_corrupted_copy_is_refused(self):
        rel = next(iter(F.recorded()))
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / (rel + '.gz')
            copy.parent.mkdir(parents=True)
            copy.write_bytes(gzip.compress(F.read_bytes(rel) + b'tampered'))
            shutil.copy2(F.FIXTURES / 'SHA256SUMS', Path(tmp) / 'SHA256SUMS')
            with patch.object(F, 'FIXTURES', Path(tmp)):
                with self.assertRaisesRegex(ValueError, 'does not match SHA256SUMS'):
                    F.read_bytes(rel)


if __name__ == '__main__':
    unittest.main()
