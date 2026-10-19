import copy
import json
import unittest
from repo_paths import ROOT  # also puts src/ first on sys.path
import llm_completion
import team_masks
import prompt_versions as V


class PromptVersionTests(unittest.TestCase):
    def test_released_prompts_are_frozen(self):
        for version in V.VERSIONS:
            self.assertEqual(V.system_text(version),(V.SNAPSHOTS/(version+'.system.txt')).read_text(),version)

    def test_every_version_is_logged(self):
        log=(ROOT/'docs/prompt-changelog.md').read_text()
        for version in V.VERSIONS:
            self.assertIn(version,log)
            self.assertIn(V.system_sha256(version)[:12],log)

    def test_regulation_version_only_appends_the_reference(self):
        old=V.messages(V.ORIGINAL,V.FIXTURE,V.FIXTURE_TRIAL);new=V.messages(V.REGULATION,V.FIXTURE,V.FIXTURE_TRIAL)
        self.assertEqual(new[0]['content'],old[0]['content']+'\n\n'+V.regulation_reference())
        self.assertEqual(new[1:],old[1:])

    def test_reference_does_not_depend_on_the_hidden_values(self):
        altered=copy.deepcopy(V.FIXTURE);team_masks.set_path(altered['starts'][0],[0,'item'],'SECRET-HIDDEN-VALUE')
        self.assertEqual(V.messages(V.REGULATION,V.FIXTURE,V.FIXTURE_TRIAL),V.messages(V.REGULATION,altered,V.FIXTURE_TRIAL))

    def test_reference_matches_the_simulator_vocabulary(self):
        reg=json.loads((ROOT/'data/regmb-vocabulary.json').read_text());text=V.regulation_reference()
        for field in ('species','item','nature'):
            for value in reg['values'][field]:
                if value:self.assertIn(reg['names'].get(value,value),text)

    def test_unknown_version_is_refused(self):
        with self.assertRaises(ValueError):llm_completion.prompt(dict(V.FIXTURE,prompt_version='mb-typo'),V.FIXTURE_TRIAL)


if __name__=='__main__':unittest.main()
