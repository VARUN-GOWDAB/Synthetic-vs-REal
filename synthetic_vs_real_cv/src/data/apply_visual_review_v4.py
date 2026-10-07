"""Apply explicit saved edits to isolated v4 datasets; retain incomplete-review status."""
import hashlib,json,math,shutil
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'results/full_visual_review_v4'
MAPPING={'real_safety_500_v3':'real_safety_500_v4','ai_generated_v3':'ai_generated_v4'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(b):
    c,x1,y1,x2,y2=b
    assert c in (0,1,2) and all(math.isfinite(v) for v in b)
    assert 0<=x1<x2<=1 and 0<=y1<y2<=1,b
    return f'{int(c)} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}'
def validate(lines):
    for line in lines:
        c,x,y,w,h=map(float,line.split())
        assert c in (0,1,2) and all(math.isfinite(v) for v in (x,y,w,h))
        assert w>0 and h>0 and min(x-w/2,y-h/2)>=-0.00001 and max(x+w/2,y+h/2)<=1.00001,line

def main():
    rows=json.loads((AUDIT/'inventory.json').read_text());dec=json.loads((AUDIT/'decisions.json').read_text())
    plans={};log=[];canonical={}
    for r in rows:
        source=ROOT/'data'/r['source'];im=source/'images'/r['image'];lp=source/'labels'/(im.stem+'.txt')
        assert sha(im)==r['image_sha256'],im
        before=lp.read_text().splitlines();assert before==r['labels'],lp
        d=dec[str(r['id'])];remove=set(d.get('remove',[]));replace=d.get('replace',{})
        assert all(isinstance(i,int) and 0<=i<len(before) for i in remove)
        assert all(0<=int(i)<len(before) and int(i) not in remove for i in replace)
        after=[encode(replace[str(i)]) if str(i) in replace else line for i,line in enumerate(before) if i not in remove]
        after.extend(encode(b) for b in d.get('add',[]));validate(after)
        key=(r['source'],r['image_sha256'])
        if key in canonical:assert canonical[key][1]==after,'Conflicting exact-duplicate decisions'
        canonical[key]=(r,after)
        plans[(r['source'],im.name)]=after
        if before!=after:log.append({'review_id':r['id'],'source':r['source'],'image':im.name,'review_status':d['status'],'before':before,'after':after,'notes':d.get('notes','')})
    # Unselected exact copies receive the same annotation as their reviewed canonical image.
    propagated=[]
    for old,new in MAPPING.items():
        source=ROOT/'data'/old;target=ROOT/'data'/new
        assert not target.exists(),f'Refusing to overwrite {target}'
        for im in (source/'images').iterdir():
            if not im.is_file():continue
            if (old,im.name) not in plans:
                key=(old,sha(im));assert key in canonical,f'Unreviewed nonduplicate image: {im}'
                r,after=canonical[key];plans[(old,im.name)]=after
                propagated.append({'source':old,'image':im.name,'canonical_review_id':r['id'],'canonical_image':r['image']})
    for old,new in MAPPING.items():
        source=ROOT/'data'/old;target=ROOT/'data'/new
        target.mkdir();shutil.copytree(source/'images',target/'images');(target/'labels').mkdir()
        for (s,name),lines in plans.items():
            if s==old:(target/'labels'/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''))
        (target/'classes.txt').write_text('worker\nhelmet\nvest\n')
        for split in ['train','val','test']:
            if (source/f'{split}.txt').exists():shutil.copy2(source/f'{split}.txt',target/f'{split}.txt')
        if (source/'data.yaml').exists():
            (target/'data.yaml').write_text('path: '+json.dumps(target.as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
        manifest=json.loads((source/'annotation_manifest.json').read_text())
        manifest.update(complete=False,status='partial_visual_review_corrections_applied',parent_dataset=old,ready_for_training=False,correction_log=(AUDIT/'applied_changes.json').as_posix(),review_decisions_sha256=sha(AUDIT/'decisions.json'))
        (target/'annotation_manifest.json').write_text(json.dumps(manifest,indent=2))
        (target/'README.md').write_text('# Partially reviewed corrected dataset\n\nSaved explicit corrections have been applied. Unresolved review notes remain; this is not fully verified ground truth. Images and matching YOLO text labels are in images/ and labels/. Existing train/val/test membership is preserved. This dataset is not activated for training.\n')
        for im in (target/'images').iterdir():
            assert sha(im)==sha(source/'images'/im.name)
            lp=target/'labels'/(im.stem+'.txt');validate(lp.read_text().splitlines())
            assert lp.read_text().splitlines()==plans[(old,im.name)]
    report={'status':'partial_corrections_applied','ready_for_training':False,'changed_reviewed_files':len(log),'changed_by_source':dict(Counter(r['source'] for r in log)),'exact_duplicate_annotations_propagated':len(propagated),'unresolved_images':sum(d['status']=='needs_close_review' for d in dec.values()),'datasets':MAPPING,'changes':log,'duplicate_propagation':propagated}
    (AUDIT/'applied_changes.json').write_text(json.dumps(report,indent=2))
    status=json.loads((AUDIT/'review_status.json').read_text());status.update(dataset_edits_applied=True,stage='partial_corrections_applied_review_incomplete',applied_changes='applied_changes.json',ready_for_training=False);(AUDIT/'review_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['changes','duplicate_propagation']},indent=2))
if __name__=='__main__':main()
