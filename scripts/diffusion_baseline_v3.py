"""V3 decoder: exact Mega Stone ownership and repeatable unused move slots."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import time
from repo_paths import ROOT  # also puts src/ first on sys.path
import battles
import results
import showdown_validator
import team_masks

MEMBERS = ["F1_final_medium_noheldout_regmb_v3", "G2_family_matrix_noprtrain_noheldout_regmb_v3"]


def restrict_move_duplicates(ok, x, col, empty):
    for other in range(col//8*8+3, col//8*8+7):
        token=int(x[0,other])
        if token and token!=empty: ok[0,token]=False


def encode_context(team, paths, vocab):
    """Never inspect a masked value, including to determine its move rank."""
    from corpus import norm
    context = team_masks.masked(team, paths)
    values, mapping, missing = [], [], []
    for s, slot in enumerate(context):
        slot = slot or dict(species=None, ability=None, item=None, nature=None, moves=[None]*4)
        moves = sorted(range(4), key=lambda m: (slot['moves'][m] is None, norm(slot['moves'][m] or '')))
        fields = [('species',), ('ability',), ('item',)] + [('moves', m) for m in moves] + [('nature',)]
        for field in fields:
            path = [s, *field]
            value = slot[field[0]] if len(field) == 1 else slot['moves'][field[1]]
            key = 'move' if field[0] == 'moves' else field[0]
            token = 0 if value is None else vocab.stoi[key].get(norm(value))
            if token is None:
                missing.append(dict(path=path, value=value))
                token = 0
            values.append(token)
            mapping.append(path)
    return values, mapping, missing


def generate(args):
    import numpy as np
    import torch
    import lossdown as L
    import asked_vs_got as X
    import diffusion as D
    from corpus import norm
    from checkpoints import load_checkpoint
    torch.set_num_threads(1)
    L.DEV = D.DEV = torch.device('cpu')
    corpus, vocab, _, look, spreads = X.setup()
    first = load_checkpoint(L.CK/(args.members[0]+'.pt'),map_location='cpu')
    if first.get('regulation',{}).get('training_policy',{}).get('version') != 'v3':
        raise ValueError('V3 decoder requires a v3 training-policy checkpoint')
    if first.get('regulation'):
        from encode import Vocab
        vocab = Vocab.from_regulation(first['regulation'])
        look.update(vocab.regulation['names'])
    tables = L.Tables(vocab, corpus)
    # Partial move sets have no known rank for masked moves. Preserve known values,
    # disable alphabetic rank constraints and enforce distinctness explicitly.
    rules = ('clause', 'compat')
    funcs = []
    config = dict(members=args.members, seed=args.manifest['seed'], attempts=1,
                  target_win_rate=.5, guidance=2, temperature=1, rules=rules,
                  stat_points='same-species corpus spread; random legal spread if species has no corpus spread; not neural prediction',
                  input='strict known categorical context; no Stat Points or opponent-paste input',
                  masked_move_order='known sorted first; blanks last; no hidden-value ranks',
                  vocab_sha256=results.digest(vocab.itos), corpus_sha256=results.digest(corpus), checkpoints={})
    for name in args.members:
        net, cfg, _ = L.load_net(name, vocab)
        funcs.append(L.logprobs_fn(net, tables, rules))
        config['checkpoints'][name] = dict(sha256=hashlib.sha256((L.CK / (name+'.pt')).read_bytes()).hexdigest(), cfg=cfg)
    config['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [Path(__file__), ROOT/'src/lossdown.py', ROOT/'src/encode.py']}
    config_path = args.output/'generation-config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('generation configuration changed; use a fresh output directory')
    results.save(config_path, config)
    f = L.ensemble_fn(funcs)
    import hps_generate as h
    validator = h.Validator()
    try:
        for i, trial in enumerate(args.manifest['trials']):
            dest = args.output/'completions'/(trial['id']+'.json')
            if dest.exists(): continue
            start = args.manifest['starts'][trial['start']]
            vals, mapping, missing = encode_context(start, trial['paths'], vocab)
            row = dict(trial=trial['id'], valid=False, supported=not missing, missing=missing)
            if missing:
                row['error'] = 'unmasked categorical value outside checkpoint vocabulary'
                results.save(dest, row)
                continue
            t0 = time.perf_counter()
            torch.manual_seed(args.manifest['seed']+i)
            rng = np.random.default_rng(args.manifest['seed']+i)
            try:
                x = torch.tensor([vals], dtype=torch.long, device=L.DEV)
                todo = [c for c in D.ORDER if vals[c] == 0]
                with torch.no_grad():
                    for step, col in enumerate(todo):
                        t = torch.tensor([(len(todo)-step)/48], device=L.DEV)
                        w, wn = torch.tensor([.5], device=L.DEV), torch.tensor([float('nan')], device=L.DEV)
                        key = L.KEY_OF_COL[col]; j = L.COLS_BY_KEY[key].index(col)
                        logits = 2*f(x,t,w)[key][:,j] - f(x,t,wn)[key][:,j]
                        ok = tables.allowed(x, rules)[key][:,j].clone()
                        ok[:,0] = False
                        if key == 'move':
                            restrict_move_duplicates(ok,x,col,tables.EMV)
                        if not ok.any(): raise ValueError('no legal token under clamped context')
                        x[:,col] = torch.multinomial(torch.softmax(logits.masked_fill(~ok,-float('inf')), -1),1).squeeze(1)
                decoded = copy.deepcopy(start)
                for col in todo:
                    value = vocab.decode_field(col, int(x[0,col]))
                    team_masks.set_path(decoded, mapping[col], look.get(value, value))
                for path in trial['paths']:
                    if len(path) == 1 or path[1] == 'evs':
                        s = path[0]; choices = spreads.get(norm(decoded[s]['species']))
                        raw = choices[int(rng.integers(len(choices)))] if choices else h.sample_spread(rng)
                        decoded[s]['evs'] = {k:int(raw.get(k,0)) for k in team_masks.STATS}
                team_masks.validate_shape(decoded)
                if team_masks.masked(decoded,trial['paths']) != team_masks.masked(start,trial['paths']):
                    raise ValueError('unmasked field changed')
                error = validator(showdown_validator.paste(decoded,h))
                row.update(team=decoded, valid=error is None, error=error)
            except ValueError as exc:
                row['error'] = str(exc)
            row['seconds'] = time.perf_counter()-t0
            results.save(dest,row)
            print(trial['id'], row['valid'], row.get('error'), flush=True)
    finally:
        validator.close()



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['generate','battle'])
    p.add_argument('--baseline',type=Path,default=ROOT/'results/llm-baseline-instant-pilot')
    p.add_argument('--output',type=Path,default=ROOT/'results/diffusion-baseline-v3')
    p.add_argument('--members',nargs='+',default=MEMBERS)
    args = p.parse_args(); args.output=args.output.resolve(); args.baseline=args.baseline.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    args.manifest=json.loads((args.baseline/'manifest.json').read_text())
    dest=args.output/'manifest.json'
    if dest.exists() and json.loads(dest.read_text()) != args.manifest: raise ValueError('frozen manifest changed')
    results.save(dest,args.manifest)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='battle':
            candidates=[]
            for path in sorted((args.output/'completions').glob('*.json')):
                row=json.loads(path.read_text())
                if row['valid']: candidates.append(('diffusion-'+row['trial'],row['team']))
            battles.battle(args,candidates_override=candidates)
        else: globals()[args.command](args)


if __name__=='__main__': main()
