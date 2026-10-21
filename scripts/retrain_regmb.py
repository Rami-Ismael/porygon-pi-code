"""Regulation-derived vocabulary, identity migration and coverage-aware retraining.

prepare builds validated synthetic training examples; train updates both F1/G2
without overwriting them. Legacy token IDs are remapped, never reused by position.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
import encode
import heldout as HO
from showdown_validator import SHOWDOWN
import numpy as np
import torch
import lossdown as L
import asked_vs_got as X
import hps_generate as H
from corpus import norm
from regmb_training_policy import mega_stone_users,training_regulation,sample_moves,encode_training_team
from checkpoints import load_checkpoint

MANIFEST = CODE/'data/regmb-vocabulary.json'
OUT = Path.home()/'.local/share/vgc-pilot-runtime/regmb-v2'
MEMBERS = ['F1_final_medium_noheldout','G2_family_matrix_noprtrain_noheldout']
SEED = 20261009


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def remap(x, old, new):
    y=np.zeros_like(x); valid=np.ones(len(x),dtype=bool)
    for key,cols in L.COLS_BY_KEY.items():
        lookup=np.array([new.stoi[key].get(v,-1) for v in old.itos[key]])
        values=lookup[x[:,cols]]
        valid &= (values>=0).all(1)
        y[:,cols]=np.maximum(values,0)
    # New alphabetic IDs can change relative ranks; sort move columns again.
    moves=y.reshape(-1,6,8)[:,:,3:7]
    empty=new.stoi['move'].get('')
    ranked=np.where(moves==empty,new.sizes['move'],moves) if empty is not None else moves
    ranked=np.sort(ranked,axis=-1)
    moves[:]=np.where(ranked==new.sizes['move'],empty,ranked) if empty is not None else ranked
    return y,valid


def migrate(net, checkpoint, old, new):
    """Copy every shared row by identity, including [MASK], and retain backbone."""
    dest=net.state_dict(); src=checkpoint['sd']; copied={}
    for name,target in dest.items():
        if name.startswith(('emb.','head.')):
            key=name.split('.')[1]
            pairs=[(i,new.stoi[key][token]) for i,token in enumerate(old.itos[key]) if token in new.stoi[key]]
            # New rows start near the distribution of existing rows, not nn defaults.
            if target.ndim==2: target.copy_(src[name][1:].mean(0)+torch.randn_like(target)*.01)
            else: target.fill_(float(src[name][1:].mean()))
            for oi,ni in pairs: target[ni]=src[name][oi]
            copied[name]=len(pairs)
        else:
            if name not in src or target.shape!=src[name].shape: raise ValueError('incompatible backbone '+name)
            target.copy_(src[name])
    net.load_state_dict(dest)
    return copied


def prepare(args):
    OUT.mkdir(parents=True,exist_ok=True)
    reg=json.loads(MANIFEST.read_text())
    if args.version=='v3':
        reg=training_regulation(reg,mega_stone_users(SHOWDOWN))
    save(OUT/'training-regulation.json',reg)
    V=encode.Vocab.from_regulation(reg)
    corpus,old,*_=X.setup(); data=L.load_data(include_heldout=True)   # remapped.npz is aligned to every row
    train,ok=remap(data['X'],old,V); test,tok=remap(data['xte'],old,V)
    # Freeze the original test and current pilot contexts out of synthetic training.
    forbidden={x.tobytes() for x in test[tok]}
    baseline=CODE/'results/llm-baseline-instant-pilot'
    for team in json.loads((baseline/'manifest.json').read_text())['starts']:
        forbidden.add(encode_training_team(V,team).tobytes())
    for p in (baseline/'completions').glob('*.json'):
        r=json.loads(p.read_text())
        if r.get('valid'):
            try: forbidden.add(encode_training_team(V,r['team']).tobytes())
            except ValueError: pass
    tables=L.Tables(V,corpus)
    rules=('clause','compat','order')
    ix=np.flatnonzero(ok); ok[ix[tables.check(train[ix],rules)]]=False
    ti=np.flatnonzero(tok); tok[ti[tables.check(test[ti],rules)]]=False
    np.savez_compressed(OUT/'remapped.npz',X=train,valid=ok,xte=test,test_valid=tok)
    rng=np.random.default_rng(SEED)
    validator=H.Validator()
    coverage={k:np.zeros(V.sizes[k],dtype=int) for k in L.KEYS}
    examples=[]; seen=set(); rejected=0
    species=reg['values']['species']; names=reg['names']
    # Target every legal non-padding token, not only frequent or newly added ones.
    targets=[(k,v) for k in L.KEYS for v in reg['values'][k] if v]
    def slot(sp, target=None):
        rules=reg['slots'][sp]; w=rules['witness']
        ab=str(rng.choice(rules['abilities'])); item=str(rng.choice(rules['abilityItems'][ab]))
        move_target=target[1] if target and target[0]=='move' else None
        moves=sample_moves(rules['moves'],rng,target=move_target,variable=args.version=='v3')
        nature=str(rng.choice(reg['values']['nature']))
        if target:
            key,value=target
            if key=='ability': ab=value;item=str(rng.choice(rules['abilityItems'][ab]))
            elif key=='item':
                item=value;ab=str(rng.choice([a for a,items in rules['abilityItems'].items() if item in items]))
            elif key=='move': pass  # Already included by sample_moves, without duplicates.
            elif key=='nature': nature=value
        return dict(species=names[sp],ability=names[ab],item=names.get(item,item),
                    nature=names.get(nature,nature),moves=[names[x] for x in moves],evs=H.sample_spread(rng))
    try:
        for attempt in range(args.examples*30):
            deficits=[(k,v) for k,v in targets if coverage[k][V.stoi[k][v]]<args.minimum]
            if len(examples)>=args.examples and not deficits: break
            target=deficits[int(rng.integers(len(deficits)))] if deficits else None
            if target:
                k,v=target
                choices=[sp for sp in species if k=='nature' or (k=='species' and sp==v) or
                         (k in ['ability','item','move'] and v in reg['slots'][sp][{'ability':'abilities','item':'items','move':'moves'}[k]])]
                chosen=str(rng.choice(choices))
            else: chosen=str(rng.choice(species))
            team=[slot(chosen,target)]; used={reg['slots'][chosen]['base']}; items={norm(team[0]['item'])}
            for _ in range(100):
                if len(team)==6:break
                sp=str(rng.choice(species)); s=slot(sp)
                if reg['slots'][sp]['base'] in used or s['item'] and norm(s['item']) in items:continue
                team.append(s);used.add(reg['slots'][sp]['base']);items.add(norm(s['item']))
            if len(team)!=6 or validator('\n\n'.join(H.slot_to_text(s) for s in team)):
                rejected+=1;continue
            x=encode_training_team(V,team); key=x.tobytes()
            if key in forbidden or key in seen:continue
            seen.add(key);examples.append(x)
            for k,cols in L.COLS_BY_KEY.items(): np.add.at(coverage[k],x[cols],1)
            if len(examples)%1000==0:print('synthetic',len(examples),'uncovered targets',len(deficits),flush=True)
        else:raise RuntimeError('coverage target not reached; inspect legality witnesses')
    finally:validator.close()
    synthetic=np.stack(examples)
    occupied=(synthetic.reshape(-1,6,8)[:,:,3:7]!=V.stoi['move']['']).sum(2)
    move_counts={str(n):int((occupied==n).sum()) for n in range(5)}
    if args.version=='v3' and (move_counts['0'] or any(not move_counts[str(n)] for n in range(1,5))):
        raise ValueError('Missing variable-move coverage')
    assert not tables.check(synthetic,rules,reps=4).any()
    np.save(OUT/'synthetic.npy',synthetic)
    save(OUT/'data-audit.json',dict(manifest_sha256=sha(MANIFEST),legacy_vocabulary=old.itos,
         sizes=V.sizes,new_values={k:sorted(set(V.itos[k])-set(old.itos[k])) for k in L.KEYS},
         legacy_rows=len(train),retained_rows=int(ok.sum()),test_rows=len(test),retained_test=int(tok.sum()),
         synthetic=len(synthetic),rejected=rejected,minimum=args.minimum,seed=SEED,
         version=args.version,move_counts=move_counts,training_regulation_sha256=sha(OUT/'training-regulation.json'),
         counts={k:{v:int(coverage[k][i]) for i,v in enumerate(V.itos[k]) if i} for k in L.KEYS},
         data_sha256=sha(L.DATA),synthetic_sha256=sha(OUT/'synthetic.npy')))
    print('Prepared',len(synthetic),'validated teams; vocabulary',V.sizes,'test retained',int(tok.sum()),flush=True)


def train(args):
    V=encode.Vocab.from_regulation(OUT/'training-regulation.json');corpus,old,*_=X.setup()
    audit=json.loads((OUT/'data-audit.json').read_text())
    assert audit['manifest_sha256']==sha(MANIFEST) and audit['legacy_vocabulary']==old.itos
    # the v2 audit predates the version and regulation hashes; its regulation is the manifest, which rebuilds the v2 vocabulary exactly
    if 'training_regulation_sha256' in audit:
        assert audit['version']==args.version and audit['training_regulation_sha256']==sha(OUT/'training-regulation.json')
    else:
        assert args.version=='v2' and sha(OUT/'training-regulation.json')==sha(MANIFEST)
    if audit['data_sha256']!=sha(L.DATA) or audit['synthetic_sha256']!=sha(OUT/'synthetic.npy'):
        raise ValueError('training data changed since preparation; rebuild the audited dataset')
    d=L.load_data(include_heldout=True); mapped=np.load(OUT/'remapped.npz');synthetic=np.load(OUT/'synthetic.npy')
    assert len(mapped['X'])==len(d['X']),'remapped.npz is not aligned to lossdata.npz'
    keep=HO.row_mask(d['X'],old)                        # ranks 51-100 are not training or validation rows
    tr,val=L.splits(d);tr=tr[keep[tr]];val=val[keep[val]];tag=L.tag_of(d['S']);rules=('clause','compat','order');tables=L.Tables(V,corpus)
    for member in args.members:
        name=member+'_regmb_'+args.version; dest=L.CK/(name+'.pt')
        if dest.exists() and not args.replace:
            previous=load_checkpoint(dest,map_location='cpu')
            if previous.get('provenance',{}).get('data_audit_sha256')!=sha(OUT/'data-audit.json'):
                raise ValueError('existing checkpoint used different data; use --replace explicitly')
            if previous.get('provenance',{}).get('trainer_sha256')!=sha(Path(__file__)):
                raise ValueError('trainer changed; use --replace explicitly')
            if any(previous['cfg'][k]!=getattr(args,k) for k in ['epochs','lr','batch']) or previous['provenance']['embedding_steps']!=args.embedding_steps:
                raise ValueError('training settings changed; use --replace explicitly')
            print('Already trained:',name,flush=True);continue
        torch.manual_seed(SEED);np.random.seed(SEED)
        ck=load_checkpoint(L.CK/(member+'.pt'),map_location=L.DEV);cfg=dict(L.DEFAULT,**ck['cfg'])
        idx=tr[(tag[tr]==L.TEST_TAG)|(tag[tr]==0)|((d['M'][tr,:,1].sum(1)>0) if cfg['mrow'] else False)]
        if cfg['mrow']:idx=np.setdiff1d(idx,L.matrix_holdout(d))
        idx=idx[mapped['valid'][idx]];vi=val[mapped['valid'][val]]
        net=L.Net(V,d=cfg['d'],nhead=cfg['nhead'],nlayer=cfg['nlayer'],dropout=cfg['dropout'],
                  time_cond=bool(cfg['time_cond']),mrow=49 if cfg['mrow'] else 0).to(L.DEV)
        copied=migrate(net,ck,old,V)
        xx=np.concatenate([mapped['X'][idx],synthetic])
        ww=L.cond_matrix(d,idx) if cfg['mrow'] else d['W'][idx]
        sw=np.full((len(synthetic),50) if cfg['mrow'] else (len(synthetic),),np.nan,dtype=np.float32)
        xx=torch.tensor(xx,device=L.DEV);ww=torch.tensor(np.concatenate([ww,sw]),device=L.DEV)
        sx=torch.tensor(synthetic,device=L.DEV);sw=torch.tensor(sw,device=L.DEV)
        cfg.update(epochs=args.epochs,lr=args.lr,batch=args.batch,final=0,warm=member,force=1,
                   vocabulary='regmb-'+args.version,rules=','.join(rules))
        log=[];t0=time.perf_counter();sched=L.sched_tensor(cfg['sched']).to(L.DEV)
        vx=mapped['X'][vi];vw=L.cond_matrix(d,vi) if cfg['mrow'] else d['W'][vi]
        # This diagnostic holdout was seen by the old final checkpoints. Do not call it unseen.
        before,_=L.evaluate(L.logprobs_fn(net,tables,rules),vx[:256],vw[:256],reps=2,batch=64)
        for stage,steps in [('embeddings',args.embedding_steps),('full',math.ceil(len(xx)/args.batch)*args.epochs)]:
            for pname,p in net.named_parameters():p.requires_grad_(stage=='full' or pname.startswith(('emb.','head.')))
            opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=args.lr if stage=='full' else args.lr*3,weight_decay=.01)
            net.train();running=0
            order=None
            for step in range(steps):
                if stage=='embeddings':
                    b=torch.randint(len(sx),(args.batch,),device=L.DEV);bx,bw=sx[b],sw[b]
                else:
                    per=math.ceil(len(xx)/args.batch);part=step%per
                    if part==0:order=torch.randperm(len(xx),device=L.DEV)
                    b=order[part*args.batch:(part+1)*args.batch];bx,bw=xx[b],ww[b]
                loss=L.batch_loss(net,tables,rules,bx,bw,None,sched,cfg)
                if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
                running+=float(loss.detach())
                if (step+1)%25==0 or step+1==steps:
                    entry=dict(stage=stage,step=step+1,steps=steps,mean_loss=running/(step+1),minutes=(time.perf_counter()-t0)/60)
                    log.append(entry);save(OUT/(name+'-progress.json'),entry)
                    print(name,stage,step+1,'/',steps,'loss',round(entry['mean_loss'],4),'min',round(entry['minutes'],1),flush=True)
            net.eval()
        after,_=L.evaluate(L.logprobs_fn(net,tables,rules),vx[:256],vw[:256],reps=2,batch=64)
        ti=np.flatnonzero(mapped['test_valid'])
        test,_=L.evaluate(L.logprobs_fn(net,tables,rules),mapped['xte'][ti],d['wte'][ti],reps=4,batch=64)
        checkpoint=dict(sd=net.state_dict(),cfg=cfg,digits=None,vocabulary=V.itos,regulation=V.regulation,
             provenance=dict(parent=member,parent_sha256=sha(L.CK/(member+'.pt')),data_audit_sha256=sha(OUT/'data-audit.json'),
                             trainer_sha256=sha(Path(__file__)),seed=SEED,embedding_steps=args.embedding_steps),
             metrics=dict(migration=before,trained=after,heldout_test=test),log=log)
        tmp=dest.with_suffix('.tmp');torch.save(checkpoint,tmp);tmp.replace(dest)
        save(OUT/(name+'-report.json'),dict(name=name,checkpoint=str(dest),checkpoint_sha256=sha(dest),
             copied_rows=copied,n_train=len(xx),n_synthetic=len(synthetic),metrics=checkpoint['metrics'],log=log))
        print('SAVED',dest,'test CE',test['ce'],flush=True)
        del net,xx,ww,sx,sw
        if L.DEV=='mps':torch.mps.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--examples',type=int,default=6000);p.add_argument('--minimum',type=int,default=12)
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--embedding-steps',type=int,default=150)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--members',nargs='+',default=MEMBERS)
    p.add_argument('--version',choices=['v2','v3'],default='v3')
    p.add_argument('--replace',action='store_true',help='replace only the named derived checkpoints')
    a=p.parse_args();OUT=OUT.parent/('regmb-'+a.version);torch.set_num_threads(2);globals()[a.command](a)
