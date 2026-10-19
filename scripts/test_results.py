import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import repo_paths  # puts src/ first on sys.path
import results


class SaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'nested' / 'record.json'

    def test_writes_json_and_creates_the_folder(self):
        results.save(self.path, {'wins': 3, 'battles': 50})
        self.assertEqual(json.loads(self.path.read_text()), {'wins': 3, 'battles': 50})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])  # no .tmp left behind

    def test_a_crash_before_the_rename_keeps_the_old_file_whole(self):
        results.save(self.path, {'version': 1})
        with patch.object(Path, 'replace', side_effect=OSError('power cut')):
            with self.assertRaises(OSError):
                results.save(self.path, {'version': 2})
        self.assertEqual(json.loads(self.path.read_text()), {'version': 1})

    def test_an_unserializable_value_never_touches_the_old_file(self):
        results.save(self.path, {'version': 1})
        with self.assertRaises(TypeError):
            results.save(self.path, {'version': object()})
        self.assertEqual(json.loads(self.path.read_text()), {'version': 1})


class DigestTests(unittest.TestCase):
    def test_key_order_does_not_change_the_digest(self):
        a = {'config': {'seed': 1, 'format': 'x'}, 'schedule': [3, 1, 2]}
        b = {'schedule': [3, 1, 2], 'config': {'format': 'x', 'seed': 1}}
        self.assertEqual(results.digest(a), results.digest(b))

    def test_values_and_list_order_do_change_it(self):
        base = results.digest({'schedule': [1, 2, 3]})
        self.assertNotEqual(base, results.digest({'schedule': [1, 2, 4]}))
        self.assertNotEqual(base, results.digest({'schedule': [3, 2, 1]}))
        self.assertEqual(len(base), 64)


if __name__ == '__main__':
    unittest.main()
