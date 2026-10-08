"""Storage regression checks: no battle execution, no changes to the live database."""
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
import matchup_db as M
import smoothness_db as D

PILOT=ROOT/'results/smoothness_pilot'
VISUAL=Path('/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Blog/visuals/smoothness/data.js')

class Persistence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'matrix.sqlite'
        M.connect(self.path).close()
        self.con=D.open_db(self.path)
        self.con.executescript(D.SCHEMA)
        self.result=D.import_run(self.con,PILOT,VISUAL)
        self.run=self.result['run_id']

    def tearDown(self):
        self.con.close()
        self.temp.cleanup()

    def test_idempotent_import_does_not_touch_pair_cells(self):
        before=self.con.execute('SELECT COUNT(*) FROM team').fetchone()[0]
        again=D.import_run(self.con,PILOT,VISUAL)
        self.assertEqual(again['new_panels'],0)
        self.assertEqual(again['saved_panels'],96)
        self.assertEqual(self.con.execute('SELECT SUM(battles) FROM smoothness_panel').fetchone()[0],4800)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM cell').fetchone()[0],0)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM contribution').fetchone()[0],0)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM team').fetchone()[0],before)
        self.assertEqual(self.con.execute('PRAGMA foreign_key_check').fetchall(),[])

    def test_export_has_no_file_or_runtime_dependency(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('export touched original files')):
            document=D.snapshot(self.con,self.run)
        self.assertEqual(document['battles'],4800)
        self.assertEqual(document['measuredMembers'],14)
        self.assertEqual(len(document['teams']),8)
        self.assertEqual(len(document['benchmarks']),6)
        for team in document['teams']:
            for edge in team['edges']:
                observed=sum(p['wins'] for p in edge['panels'])/150-team['original']
                self.assertAlmostEqual(observed,edge['delta'])

    def test_exact_paste_and_duplicate_opponents_retained(self):
        manifest=json.loads((PILOT/'manifest.json').read_text())
        for c in manifest['candidates']:
            saved=self.con.execute('SELECT * FROM smoothness_candidate WHERE run_id=? AND candidate_id=?',(self.run,c['id'])).fetchone()
            self.assertEqual(saved['exact_paste'],Path(c['file']).read_text())
            self.assertEqual(saved['exact_sha256'],c['sha256'])
            self.assertIsNotNone(saved['matrix_team_id'])
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM smoothness_opponent').fetchone()[0],50)
        self.assertEqual(self.con.execute('SELECT COUNT(DISTINCT matrix_team_id) FROM smoothness_opponent').fetchone()[0],49)

    def test_held_items_come_from_battled_slots_and_support_itemless_members(self):
        self.assertEqual(D.held_items_from_paste('Buddy (Pikachu) (M) @ Light Ball\nAbility: Static\n- Protect\n\nCharizard\nAbility: Blaze\n- Protect\n'), ['Light Ball', None])
        D.save_reference(self.con,self.run,'reference/item-sprites.json',json.dumps({'charizarditey':{'sprite':'items/item-charizarditey.png'}}).encode())
        with patch.object(Path,'read_text',side_effect=AssertionError('export touched original files')):
            document=D.snapshot(self.con,self.run)
        team=next(t for t in document['teams'] if t['id']=='tournament-000')
        self.assertEqual([p['heldItem'] for p in team['roster']], ['Charizardite Y','Life Orb','Expert Belt','Sitrus Berry','Fairy Feather','Aerodactylite'])
        self.assertEqual(team['roster'][0]['itemSprite'],'items/item-charizarditey.png')
        self.assertIsNone(team['roster'][1]['itemSprite'])

    def test_changed_panel_refused_and_transaction_rolled_back(self):
        changed=Path(self.temp.name)/'modified'
        (changed/'labels').mkdir(parents=True)
        shutil.copy2(PILOT/'manifest.json',changed/'manifest.json')
        label=next((PILOT/'labels').glob('*.json'))
        row=json.loads(label.read_text())
        row['wins']=row['wins']+1 if row['wins']<row['battles'] else row['wins']-1
        (changed/'labels'/label.name).write_text(json.dumps(row))
        before=self.con.execute('SELECT SUM(wins) FROM smoothness_panel').fetchone()[0]
        with self.assertRaisesRegex(ValueError,'conflicting saved smoothness_panel'):
            D.import_run(self.con,changed)
        self.assertEqual(self.con.execute('SELECT SUM(wins) FROM smoothness_panel').fetchone()[0],before)
        self.assertEqual(self.con.execute('SELECT status FROM smoothness_run').fetchone()[0],'complete')

    def test_complete_analysis_with_missing_panels_refused(self):
        incomplete=Path(self.temp.name)/'incomplete'
        incomplete.mkdir()
        shutil.copy2(PILOT/'manifest.json',incomplete/'manifest.json')
        shutil.copy2(PILOT/'analysis.json',incomplete/'analysis.json')
        with self.assertRaisesRegex(ValueError,'missing battle panels'):
            D.import_run(self.con,incomplete)

    def test_backup_is_a_valid_independent_snapshot(self):
        saved=D.backup(self.path)
        with sqlite3.connect(saved) as con:
            self.assertEqual(con.execute('PRAGMA integrity_check').fetchone()[0],'ok')
            self.assertEqual(con.execute('SELECT COUNT(*) FROM smoothness_panel').fetchone()[0],96)

    def test_member_extension_reuses_baselines_and_preserves_primary_summary(self):
        before=D.snapshot(self.con,self.run)['summary']
        result=D.import_run(self.con,ROOT/'results/smoothness_member_coverage')
        self.assertEqual(result['new_panels'],102)
        with patch.object(Path,'read_text',side_effect=AssertionError('export touched source files')):
            view=D.snapshot(self.con,self.run)
        self.assertEqual(view['battles'],9900)
        self.assertEqual(view['primaryBattles'],4800)
        self.assertEqual(view['measuredMembers'],48)
        self.assertEqual(view['summary'],before)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM cell').fetchone()[0],0)
        for team in view['teams']:
            self.assertEqual({e['edit']['slot'] for e in team['edges'] if e.get('edit')},set(range(6)))
            for edge in team['edges']:
                self.assertAlmostEqual(sum(p['wins'] for p in edge['panels'])/150-team['original'],edge['delta'])
        again=D.import_run(self.con,ROOT/'results/smoothness_member_coverage')
        self.assertEqual(again['new_panels'],0)

if __name__=='__main__':
    unittest.main()
