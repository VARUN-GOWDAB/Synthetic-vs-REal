"""Audit current sources and prepare reproducible, equal-size comparison datasets."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, random
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/processed/comparison_v2'
AUDIT = ROOT / 'results/comparison_preparation_v2'
SOURCES = ['real_safety_500', 'ai_generated', '3d_rendered']

def digest_image(p):
    with Image.open(p) as image:
        image.load()
        arr = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float32)
        dct = cv2.dct(arr)[:8, :8].flatten()[1:]
        bits = dct > np.median(dct)
        return sum(int(b) << i for i, b in enumerate(bits))

def scan():
    records = []
    for source in SOURCES:
        base = ROOT / 'data' / source
        manifest = json.loads((base/'annotation_manifest.json').read_text())
        selected = set(manifest['images'])
        partitions = {}
        if source == SOURCES[0]:
            for split in ['train','val','test']:
                for line in (base/f'{split}.txt').read_text().splitlines():
                    name = Path(line).name
                    assert name not in partitions, f'Duplicate split membership: {name}'
                    partitions[name] = split
            assert set(partitions) == selected
        for name in sorted(selected):
            image = base/'images'/name
            label = base/'labels'/f'{image.stem}.txt'
            counts = Counter()
            for line in label.read_text().splitlines():
                f = line.split()
                assert len(f) == 5, (label,line)
                c = int(f[0]); x,y,w,h = map(float,f[1:])
                assert c in (0,1,2) and all(math.isfinite(v) for v in (x,y,w,h))
                assert w>0 and h>0 and min(x-w/2,y-h/2)>=-1e-5 and max(x+w/2,y+h/2)<=1.00001
                counts[c] += 1
            assert counts, f'Empty label: {label}'
            records.append(dict(source=source,image=name,path=image.resolve().as_posix(),
                sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
                phash=digest_image(image), split=partitions.get(name,'synthetic_pool'),objects=dict(counts)))
    return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');a=parser.parse_args()
    AUDIT.mkdir(parents=True,exist_ok=True)
    records=scan();pairs=[]
    for i,left in enumerate(records):
        for right in records[i+1:]:
            distance=(left['phash']^right['phash']).bit_count()
            if distance<=6:
                pairs.append(dict(left=left['path'],right=right['path'],distance=distance,
                    exact=left['sha256']==right['sha256'],
                    cross_real_split=left['source']==right['source']==SOURCES[0] and left['split']!=right['split']))
    (AUDIT/'inventory.json').write_text(json.dumps(records,indent=2))
    (AUDIT/'similarity_candidates.json').write_text(json.dumps(pairs,indent=2))
    risky=[r for r in pairs if r['cross_real_split']]
    for page in range((len(risky)+9)//10):
        canvas=Image.new('RGB',(1200,1250),'#222');draw=ImageDraw.Draw(canvas)
        for j,pair in enumerate(risky[page*10:page*10+10]):
            y=j*125
            for col,key in enumerate(['left','right']):
                im=Image.open(pair[key]).convert('RGB');im.thumbnail((580,100));canvas.paste(im,(col*600,y+22))
                draw.text((col*600,y),f'{page*10+j+1} {Path(pair[key]).name} d={pair["distance"]}',fill='white')
        canvas.save(AUDIT/f'leakage_candidates_{page+1:02}.jpg')
    print(json.dumps({'images':len(records),'similar_pairs':len(pairs),'real_cross_split_candidates':len(risky)},indent=2),flush=True)
    if a.audit_only:return
    assert not risky, 'Review cross-split candidates before preparing training.'
    assert not OUT.exists(),'Comparison output already exists; do not overwrite frozen manifests.'
    real={s:[r for r in records if r['source']==SOURCES[0] and r['split']==s] for s in ['train','val','test']}
    ai=[r for r in records if r['source']==SOURCES[1]]
    rendered=[r for r in records if r['source']==SOURCES[2]]
    # Synthetic sources are training-only; related synthetic scenes never enter val/test.
    holdout_hashes={r['sha256'] for r in real['val']+real['test']}
    assert all(r['sha256'] not in holdout_hashes for r in real['train']+ai+rendered)
    for pool in [real['train'],ai,rendered]:random.Random(42).shuffle(pool)
    n=min(400,len(real['train']),len(ai),len(rendered));assert n%4==0
    experiments={
        f'real_only_{n}':real['train'][:n],
        f'ai_only_{n}':ai[:n],
        f'mixed_ai50_rendered25_real25_{n}':ai[:n//2]+rendered[:n//4]+real['train'][:n//4],
        f'rendered_only_{n}':rendered[:n],
        f'mixed_real50_ai25_rendered25_{n}':real['train'][:n//2]+ai[:n//4]+rendered[:n//4],
    }
    plan=[]
    for name,train in experiments.items():
        out=OUT/name;out.mkdir(parents=True)
        for split,members in [('train',train),('val',real['val']),('test',real['test'])]:
            counts=Counter()
            for r in members:counts.update({int(k):v for k,v in r['objects'].items()})
            assert all(counts[c]>0 for c in range(3)),(name,split,counts)
            (out/f'{split}.txt').write_text('\n'.join(r['path'] for r in members)+'\n')
        (out/'data.yaml').write_text('path: '+json.dumps(out.resolve().as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
        info={'name':name,'train_images':len(train),'sources':dict(Counter(r['source'] for r in train)),
            'val_images':len(real['val']),'test_images':len(real['test']),
            'data':(out/'data.yaml').resolve().as_posix(),'train':train}
        (out/'manifest.json').write_text(json.dumps(info,indent=2));plan.append({k:v for k,v in info.items() if k!='train'})
    config={'version':2,'real_dataset':'real_safety_500','audit_directory':'comparison_preparation_v2','experiments':plan,'model':'yolov8n.pt','epochs':50,'imgsz':640,'batch':8,'seed':42,
        'device':'cpu','threads':4,'optimizer':'AdamW','lr0':0.001,'patience':0,
        'annotation_status':'Real source labels remapped; synthetic labels model-assisted; not exhaustive manual ground truth',
        'evaluation_policy':'Same fixed real validation/test sets; test once after training, never for tuning.',
        'limitations':['Single seed preliminary comparison','Synthetic scenes exclusively in training; source scene IDs unavailable','Perceptual similarity screen is not proof of scene independence']}
    (ROOT/'configs/comparison_v2.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
