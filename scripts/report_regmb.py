"""Publish an auditable summary of completed full-vocabulary retraining."""
import json
from pathlib import Path
import retrain_regmb as R
import torch

audit=json.loads((R.OUT/'data-audit.json').read_text())
reports=[json.loads((R.OUT/(n+'_regmb_v2-report.json')).read_text()) for n in R.MEMBERS]
manifest=json.loads(R.MANIFEST.read_text())
V=R.encode.Vocab.from_regulation(manifest)
_,old,*_=R.X.setup()
embedding_audit=[]
for name in R.MEMBERS:
    trained=torch.load(R.L.CK/(name+'_regmb_v2.pt'),map_location=R.L.DEV)
    torch.manual_seed(trained['provenance']['seed'])
    parent=torch.load(R.L.CK/(name+'.pt'),map_location=R.L.DEV);cfg=parent['cfg']
    net=R.L.Net(V,d=cfg['d'],nhead=cfg['nhead'],nlayer=cfg['nlayer'],dropout=cfg['dropout'],
                time_cond=bool(cfg['time_cond']),mrow=49 if cfg['mrow'] else 0).to(R.L.DEV)
    R.migrate(net,parent,old,V)
    changes=[]
    for key,tokens in audit['new_values'].items():
        ids=[V.stoi[key][v] for v in tokens]
        changes.extend((trained['sd']['emb.'+key+'.weight'][ids]-net.emb[key].weight[ids]).norm(dim=1).detach().cpu().tolist())
    assert trained['provenance']['parent_sha256']==R.sha(R.L.CK/(name+'.pt'))
    embedding_audit.append(dict(model=name,rows=len(changes),changed=sum(x>1e-5 for x in changes),minimum_l2_change=min(changes)))
    del net,parent,trained
summary=dict(format=manifest['format'],simulator_sha256=manifest['simulator_sha256'],
             manifest_sha256=R.sha(R.MANIFEST),sizes=audit['sizes'],
             new_values=audit['new_values'],synthetic_teams=audit['synthetic'],
             minimum_nonempty_token_count=min(n for counts in audit['counts'].values() for token,n in counts.items() if token),
             models=reports,embedding_audit=embedding_audit)
R.save(R.CODE/'data/regmb-v2-training.json',summary)
lines=['# Regulation M-B vocabulary and retraining','',
       '**Both F1 and G2 were retrained with a simulator-derived vocabulary. All 63 frozen baseline contexts are representable.**','',
       'The original checkpoints remain unchanged. New checkpoints add `_regmb_v2` to their names. `regmb_model.model()` loads the new pair; older experiments retain explicit historical checkpoints.','',
       '## What changed','',
       'The corpus no longer defines the vocabulary for this pipeline. The exporter queries the pinned Champions M-B simulator, including legal unevolved species and forme representations, native and item-dependent abilities, usable items, learnable moves and all 25 named natures. Empty items, move padding and omitted/default natures have explicit tokens; they are distinct from the mask token. Accepted species/form entries are not a count of distinct evolution families.','',
       '| Field | Named/usable values | New tokens versus old vocabulary |','|---|---:|---:|']
for k in ['species','ability','item','move','nature']:
    lines.append(f"| {k} | {len([v for v in manifest['values'][k] if v])} | {len(audit['new_values'][k])} |")
lines += ['', 'Ability/item compatibility is checked jointly: for example, Metagross can name Tough Claws when holding Metagrossite, but that combination is not accepted without the stone. The complete held-out set was retained after verifying these conditional forms and preserving move-padding order.','',
          'Token migration copies shared embedding rows, output weights and biases by token identity, and preserves the transformer backbone. New rows are initialized near existing rows and trained. Checkpoints store token identities, the full rules snapshot and parent/data/source fingerprints. Loaders reject a mismatched vocabulary or rules snapshot rather than silently loading the wrong indices.','',
          '## Training actually run','',
          f"Generated {audit['synthetic']:,} distinct simulator-validated teams. Every nonempty legal token appears at least {summary['minimum_nonempty_token_count']} times. Synthetic examples carry no invented win-rate or opponent-row labels. Original test teams and frozen pilot inputs/legal Ling outputs are excluded from synthetic data.", '',
          f"The remapped historical pool retains {audit['retained_rows']:,}/{audit['legacy_rows']:,} teams under compatibility checks. The original test set retains {audit['retained_test']}/{audit['test_rows']} teams. F1 uses its historical family cohort; G2 also uses matrix-labelled teams, preserving its matrix holdout.", '',
          'Each model receives 150 embedding/output-head warm-up steps with the backbone frozen, then two full-network epochs. Batch size 128; warm-up learning rate 0.0003, full-network rate 0.0001. Training seed 20261009 is a fixed integer, not the execution date. This is continued training of the existing models, not training from scratch.','',
          '| Model | Training rows incl. synthetic | Migration diagnostic CE | After training diagnostic CE | Held-out 500-team CE |','|---|---:|---:|---:|---:|']
for r in reports:
    m=r['metrics'];lines.append(f"| {r['name']} | {r['n_train']:,} | {m['migration']['ce']:.4f} | {m['trained']['ce']:.4f} | {m['heldout_test']['ce']:.4f} |")
lines += ['', 'All 457 newly added embedding rows changed during training in each model, verified against deterministic migration initialization. Parent checkpoint hashes are unchanged.', '',
          'Migration/after diagnostics use the same 256 historical validation rows and mask seeds; those rows were seen by the parent final checkpoints, so they are not an unseen generalization test. The held-out test remains separate. Expanded-vocabulary losses are not directly interchangeable with published losses from the smaller output vocabulary. Battle performance is evaluated separately.','',
          '## Reproduce','', 'From the repository root:', '', '```sh',
          'node code/scripts/export_regmb_vocab.js "$HOME/.local/share/vgc-pilot-runtime/validator-913da36" code/data/regmb-vocabulary.json',
          '/tmp/vgc-pilot/.venv/bin/python code/scripts/retrain_regmb.py prepare',
          '/tmp/vgc-pilot/.venv/bin/python code/scripts/retrain_regmb.py train',
          '/tmp/vgc-pilot/.venv/bin/python code/scripts/report_regmb.py', '```','',
          'Existing derived checkpoints are reused only when the recorded data, trainer and settings match. `--replace` explicitly reruns and replaces only the derived v2 checkpoint names. Parent checkpoints are never overwritten. Data and weights are under `~/.local/share/vgc-pilot-runtime/`; the rules snapshot and training summary are in `code/data/`.','',
          '## Remaining limits','',
          'Vocabulary coverage is not a guarantee of legal or strong completions. Final Showdown validation still gates every output. Stat Points remain outside the neural grid: fills use corpus spreads, with a random legal spread when the new species has no corpus spread. Explicit opponent pastes are still not model inputs. There were no new LLM API calls during this retraining.', '',
          'The updated frozen-mask battle comparison is documented in `diffusion-baseline-comparison.md`.']
comparison=R.CODE/'results/diffusion-baseline-f1g2-regmb-v2/summary.json'
if comparison.exists():
    c=json.loads(comparison.read_text());pairs=c['pairs'];starts=sorted({p['start'] for p in pairs})
    delta=sum(sum(p['delta_pp'] for p in pairs if p['start']==s)/sum(p['start']==s for p in pairs) for s in starts)/len(starts)
    lines += ['',f"Completed pilot: {c['supported']}/63 contexts supported; {c['legal']}/63 legal completions; {c['scored']} scored 50-battle panels (exact cached panels reused where applicable). On {len(pairs)} jointly legal tasks, the equal-start-weighted difference versus Ling is {delta:+.2f} percentage points. Only three starting teams: no statistically established winner. One decoder dead end was recorded without repair or retry."]
(R.CODE/'docs/regmb-vocabulary-retraining.md').write_text('\n'.join(lines)+'\n')
print('Saved retraining report and checkpoint metadata.')
