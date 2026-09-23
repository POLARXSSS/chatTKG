"""Raw-train-only external hierarchy retrieval study; no Grapher imports."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

SOURCE_URL = 'https://raw.githubusercontent.com/openeventdata/Dictionaries/master/CAMEO.Manual.1.1b3.tex'

def norm(s):
    s = s.lower().replace('_', ' ').replace(', not specified below', '')
    return re.sub('[^a-z0-9]', '', s)

def mapping(source, relations):
    entries = {}
    for line in source.splitlines():
        match = re.search(r'(\d{3,4}):\s*([^\\]+)', line)
        if match:
            entries[norm(match[2])] = (match[1], match[2].strip())
    records = []
    for name, rid in sorted(relations.items(), key=lambda x:x[1]):
        found = entries.get(norm(name))
        # Version mismatch stays unmapped; no fuzzy label-to-code inference.
        records.append(dict(id=rid, name=name, code=found[0] if found else None,
                            source_label=found[1] if found else None))
    return records

def rank_rr(scores, answer, answers, entities):
    target = scores.get(answer, 0.)
    remaining = {k:v for k,v in scores.items() if k not in answers or k==answer}
    n = entities-len(answers)+1
    above = sum(v > target for v in remaining.values())
    tied = sum(v == target for v in remaining.values())
    if target == 0:
        tied += n-len(remaining)
    return 1./(1+above+(tied-1)/2)

def replay(facts, groups, nrel, nent, cut, alphas):
    exact, family = defaultdict(dict), defaultdict(dict)
    rows = []
    for day in sorted(set(facts[:,3])):
        original = facts[facts[:,3]==day]
        inverse = original[:,[2,1,0,3]].copy()
        inverse[:,1] += nrel
        batch = np.vstack([original,inverse])
        answers = defaultdict(set)
        for s,r,o,t in batch: answers[(int(s),int(r))].add(int(o))
        if day >= cut:
            for s,r,o,t in batch:
                s,r,o,t = map(int,(s,r,o,t))
                root = groups[r%nrel]
                key = (s,root,r//nrel)
                a = {k:np.exp(-(t-v)/30.) for k,v in exact[(s,r)].items()}
                b = {k:np.exp(-(t-v)/30.) for k,v in family[key].items()} if root is not None else a
                candidates = a.keys()|b.keys()
                values=[]
                for alpha in alphas:
                    scores={k:(1-alpha)*a.get(k,0.)+alpha*b.get(k,0.) for k in candidates}
                    scores={k:v for k,v in scores.items() if v>0}
                    values.append(rank_rr(scores,o,answers[(s,r)],nent))
                rows.append([t,r,int(o in a),int(o in b),*values])
        # Complete timestamp barrier, including inverse views.
        for s,r,o,t in batch:
            s,r,o,t=map(int,(s,r,o,t))
            exact[(s,r)][o]=t
            root=groups[r%nrel]
            if root is not None: family[(s,root,r//nrel)][o]=t
    return np.asarray(rows,dtype=float)

def summarize(rows, cutoff, alphas, nrel, seed=13):
    early=rows[:,0]<cutoff; late=~early
    chosen=int(np.argmax(rows[early,4:].mean(axis=0)))
    diff=rows[late,4+chosen]-rows[late,4]
    days=np.unique(rows[late,0]); day_sums=np.array([diff[rows[late,0]==d].sum() for d in days])
    counts=np.array([(rows[late,0]==d).sum() for d in days])
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(days),size=(5000,int(np.ceil(len(days)/7))))
    idx=((starts[:,:,None]+np.arange(7))%len(days)).reshape(5000,-1)[:,:len(days)]
    ci=np.quantile(day_sums[idx].sum(axis=1)/counts[idx].sum(axis=1),[.025,.975])
    direction=[float(diff[rows[late,1]<nrel].mean()),float(diff[rows[late,1]>=nrel].mean())]
    return dict(alpha=alphas[chosen],queries=int(late.sum()),days=len(days),
        selection_grid_mrr=rows[early,4:].mean(axis=0).tolist(),
        baseline_mrr=float(rows[late,4].mean()),selected_mrr=float(rows[late,4+chosen].mean()),
        delta=float(diff.mean()),ci95_7day_moving_block=ci.tolist(),direction_delta=direction,
        relation_macro_delta=float(np.mean([diff[rows[late,1]==r].mean() for r in np.unique(rows[late,1])])),
        exact_coverage=float(rows[late,2].mean()),family_coverage=float(rows[late,3].mean()),
        daily=[dict(day=int(d),queries=int(n),delta=float(v/n)) for d,n,v in zip(days,counts,day_sums)])

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    args=p.parse_args(); data=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=Path(args.source).read_text(encoding='latin-1')
    rel=json.loads((data/'relation2id.json').read_text(encoding='utf-8')); ent=json.loads((data/'entity2id.json').read_text(encoding='utf-8')); ts=json.loads((data/'ts2id.json').read_text(encoding='utf-8'))
    records=mapping(source,rel);groups=[r['code'][:2] if r['code'] else None for r in records]
    (out/'mapping.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    facts=[]
    for line in (data/'train.txt').read_text(encoding='utf-8').splitlines():
        s,r,o,t=line.split('\t');facts.append((ent[s],rel[r],ent[o],ts[t]))
    facts=np.array(facts);days=np.unique(facts[:,3]);cut=days[int(len(days)*.6)];late=days[int(len(days)*.8)]
    alphas=[0,.05,.1,.2,.4]; results={}
    configs=[('official',groups),('all_relations',['all']*len(groups))]
    known=np.flatnonzero([x is not None for x in groups]);rng=np.random.default_rng(13)
    for seed in range(19):
        g=list(groups); shuffled=rng.permutation(known)
        for i,j in zip(known,shuffled):g[i]=groups[j]
        configs.append((f'shuffled_{seed}',g))
    for name,g in configs:
        rows=replay(facts,g,len(rel),len(ent),cut,alphas)
        result=summarize(rows,late,alphas,len(rel));results[name]=result
        np.savez_compressed(out/(name+'.npz'),rows=rows)
        print(name,json.dumps({k:v for k,v in result.items() if k!='daily'}),flush=True)
    official=results['official'];controls=[results[f'shuffled_{i}']['delta'] for i in range(19)]
    passed=bool(official['delta']>=.001 and official['ci95_7day_moving_block'][0]>0 and min(official['direction_delta'])>0 and official['delta']>max(controls+[results['all_relations']['delta']]))
    report=dict(status='exploratory_not_fresh_holdout',source_url=SOURCE_URL,
        source_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        train_sha256=hashlib.sha256((data/'train.txt').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mapped_relations=sum(x is not None for x in groups),total_relations=len(rel),
        mapped_fact_fraction=float(np.mean([groups[r] is not None for r in facts[:,1]])),
        selection_start=int(cut),late_start=int(late),alphas=alphas,results=results,
        permutation_rank_p=(1+sum(v>=official['delta'] for v in controls))/20,
        progression_gate_passed=passed,validation_read=False,test_read=False)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
