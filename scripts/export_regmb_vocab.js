// Probe the pinned simulator, never a list of values observed in the corpus.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = process.argv[2];
if (!root) throw Error('usage: node export_regmb_vocab.js SHOWDOWN_ROOT OUTPUT');
const {TeamValidator} = require(path.join(root,'dist/sim/team-validator'));
const format = 'gen9championsvgc2026regmb';
const validator = TeamValidator.get(format), dex = validator.dex;
const test = set => !validator.validateSet(structuredClone(set), {});
const slots = {}, names = {}, candidates = {};
const remember = x => { if(x.id) names[x.id]=x.name; };
const moves = dex.moves.all().filter(m=>m.exists&&!validator.checkMove({species:'Pikachu'},m,{}));
const items = [{id:'',name:''},...dex.items.all().filter(i=>i.exists&&!validator.checkItem({species:'Pikachu'},i,{}))];
for(const sp of dex.species.all()) {
  if(!sp.exists) continue;
  const base=dex.species.get(sp.baseSpecies);
  const relatives=dex.species.all().filter(s=>s.baseSpecies===sp.baseSpecies);
  const abs=[...new Set([sp,base,...relatives].flatMap(s=>Object.values(s.abilities)))];
  const item=sp.requiredItem || sp.requiredItems?.[0] || '';
  let seed={name:sp.name,species:sp.name,ability:abs[0],item,nature:'Hardy',level:50,evs:{hp:1},moves:['Protect']};
  if(validator.checkSpecies(structuredClone(seed),sp,sp,{})) continue;
  let witness=null;
  for(const ab of abs){
    for(const mv of moves){
      const s={...seed,ability:ab,moves:[mv.name]};
      if(test(s)){witness=s;break;}
    }
    if(witness)break;
  }
  if(!witness)continue;
  remember(sp);
  // A base forme may legally name its Mega ability when holding its Mega stone.
  // Probe joint ability/item contexts before taking marginal vocabularies.
  const abilityItems={};
  for(const a of abs){
    const allowed=items.filter(i=>test({...witness,ability:a,item:i.name}));
    if(!allowed.length)continue;
    const ab=dex.abilities.get(a);remember(ab);
    abilityItems[ab.id]=allowed.map(i=>{remember(i);return i.id;});
  }
  const abilities=Object.keys(abilityItems).sort();
  const legalItems=[...new Set(Object.values(abilityItems).flat())].sort();
  const contexts=[];
  for(const ab of abilities){
    const its=abilityItems[ab];
    for(const it of its.filter((id,i)=>i===0||relatives.some(s=>s.name===dex.items.get(id).megaStone)))contexts.push({ability:names[ab],item:names[it]||it});
  }
  const legalMoves=moves.filter(m=>contexts.some(ctx=>test({...witness,...ctx,moves:[m.name]}))).map(m=>{remember(m);return m.id;});
  slots[sp.id]={abilities,moves:legalMoves,items:legalItems,abilityItems,base:base.id,witness};
  if(Object.keys(slots).length%40===0)console.error('Probed',Object.keys(slots).length,'species');
}
for(const key of ['ability','item','move']) candidates[key]=[...new Set(Object.values(slots).flatMap(s=>s[key==='ability'?'abilities':key==='item'?'items':'moves']))].sort();
candidates.species=Object.keys(slots).sort();
candidates.move=['',...candidates.move]; // Empty move slots are padding, not a learnable move.
candidates.nature=dex.natures.all().filter(n=>Object.values(slots).some(s=>test({...s.witness,nature:n.name}))).map(n=>{remember(n);return n.id;}).sort();
candidates.nature.unshift(''); // Omitted nature is legal and defaults to Serious.
const files=[];
function walk(dir){for(const e of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,e.name);if(e.isDirectory())walk(p);else if(p.endsWith('.js'))files.push(p);}}
walk(path.join(root,'dist'));
const hash=crypto.createHash('sha256');for(const p of files.sort()){hash.update(path.relative(root,p));hash.update(fs.readFileSync(p));}
const out={format,simulator_sha256:hash.digest('hex'),method:'All mod-dex species/moves/items/natures; validateSet witnesses and species-conditioned probes',values:candidates,names,slots};
fs.mkdirSync(path.dirname(process.argv[3]),{recursive:true});fs.writeFileSync(process.argv[3],JSON.stringify(out,null,2)+'\n');
console.log(JSON.stringify(Object.fromEntries(Object.entries(candidates).map(([k,v])=>[k,v.length]))));
