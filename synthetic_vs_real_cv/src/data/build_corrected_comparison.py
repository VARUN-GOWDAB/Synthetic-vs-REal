"""Apply explicit visual-review decisions to versioned copies and freeze equal-size experiments."""
from pathlib import Path
from collections import Counter
import json,shutil,hashlib,math
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'results/annotation_audit_v3'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    candidates=json.loads((AUDIT/'candidates.json').read_text())
    decisions=json.loads((AUDIT/'decisions.json').read_text())
    assert decisions['reviewed_candidates']==len(candidates)
    accepted=set(decisions['accept']);assert accepted.issubset({r['id'] for r in candidates})
    mappings={'real_safety_500':'real_safety_500_v3','ai_generated':'ai_generated_v3'}
    for old,new in mappings.items():
        source=ROOT/'data'/old;target=ROOT/'data'/new
        assert not target.exists(),target
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('*.cache'))
    changes=[]
    for r in candidates:
        if r['id'] not in accepted:continue
        target=ROOT/'data'/mappings[r['source']]/'labels'/(Path(r['image']).stem+'.txt')
        lines=target.read_text().splitlines();before=lines.copy()
        if r['old']:
            assert r['old']['line'] in lines,(r['id'],'conflicting edits')
            lines.remove(r['old']['line'])
        x1,y1,x2,y2=r['box'];line=f"{r['c']} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}"
        lines.append(line);target.write_text('\n'.join(lines)+'\n')
        changes.append({**r,'new_line':line,'previous_label_lines':before})
    # Explicit human/visual corrections not proposed by the detector.
    for r in decisions.get('manual',[]):
        target=ROOT/'data'/mappings[r['source']]/'labels'/(Path(r['image']).stem+'.txt')
        lines=target.read_text().splitlines();before=lines.copy()
        for line in r.get('remove',[]):assert line in lines;lines.remove(line)
        lines.extend(r.get('add',[]));target.write_text('\n'.join(lines)+'\n');changes.append({**r,'previous_label_lines':before})
    for old,new in mappings.items():
        source=ROOT/'data'/old;target=ROOT/'data'/new
        manifest=json.loads((target/'annotation_manifest.json').read_text());manifest.update(status='targeted_visual_corrections_v3_not_exhaustive_ground_truth',parent_dataset=old,correction_log=str(AUDIT/'applied_changes.json'));(target/'annotation_manifest.json').write_text(json.dumps(manifest,indent=2))
        if old=='real_safety_500':
            for split in ['train','val','test']:assert (target/f'{split}.txt').read_bytes()==(source/f'{split}.txt').read_bytes()
            for split in ['val','test']:
                for line in (source/f'{split}.txt').read_text().splitlines():
                    stem=Path(line).stem;assert digest(source/'labels'/f'{stem}.txt')==digest(target/'labels'/f'{stem}.txt')
            (target/'data.yaml').write_text('path: '+json.dumps(target.resolve().as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
    inventory=[]
    for source in [*mappings.values(),'3d_rendered']:
        base=ROOT/'data'/source;names=json.loads((base/'annotation_manifest.json').read_text())['images']
        for name in names:
            p=base/'images'/name;lp=base/'labels'/(p.stem+'.txt');counts=Counter()
            for line in lp.read_text().splitlines():
                c,x,y,w,h=map(float,line.split());assert c in [0,1,2] and all(math.isfinite(v) for v in [x,y,w,h]) and w>0 and h>0 and min(x-w/2,y-h/2)>=-1e-5 and max(x+w/2,y+h/2)<=1.00001
                counts[int(c)]+=1
            assert counts
            inventory.append({'source':source,'image':name,'path':p.resolve().as_posix(),'sha256':digest(p),'label_sha256':digest(lp),'objects':dict(counts)})
    bypath={r['path']:r for r in inventory}
    previous=json.loads((ROOT/'configs/comparison_v2.json').read_text());config={**previous,'version':3,'real_dataset':mappings['real_safety_500'],'ai_dataset':mappings['ai_generated'],'audit_directory':'comparison_preparation_v3','annotation_status':'Targeted visually reviewed training-label corrections; original real holdouts unchanged and not exhaustively verified','experiments':[]}
    def remap(path):
        for old,new in mappings.items():path=path.replace('/data/'+old+'/', '/data/'+new+'/')
        return path
    for exp in previous['experiments']:
        old=Path(exp['data']).parent;name=exp['name']+'_v3';out=ROOT/'data/processed/comparison_v3'/name;out.mkdir(parents=True)
        for split in ['train','val','test']:
            lines=[remap(line) for line in (old/f'{split}.txt').read_text().splitlines()];(out/f'{split}.txt').write_text('\n'.join(lines)+'\n')
        rows=[bypath[line] for line in (out/'train.txt').read_text().splitlines()]
        (out/'data.yaml').write_text('path: '+json.dumps(out.resolve().as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
        info={**exp,'name':name,'data':(out/'data.yaml').resolve().as_posix(),'sources':dict(Counter(r['source'] for r in rows))}
        (out/'manifest.json').write_text(json.dumps({**info,'train':rows},indent=2));config['experiments'].append(info)
    out=ROOT/'results/comparison_preparation_v3';out.mkdir(parents=True,exist_ok=True);(out/'inventory.json').write_text(json.dumps(inventory,indent=2))
    (ROOT/'configs/comparison_v3.json').write_text(json.dumps(config,indent=2));(AUDIT/'applied_changes.json').write_text(json.dumps(changes,indent=2))
    print('Applied corrections:',len(changes),'files:',len({(r['source'],r['image']) for r in changes}))

if __name__=='__main__':main()
