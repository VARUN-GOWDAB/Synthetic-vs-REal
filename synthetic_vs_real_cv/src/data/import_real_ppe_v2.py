"""Import the supplied PPE dataset, retaining source IDs 6, 0 and 2 only."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,shutil,random
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('C:/Users/SIC/Downloads/archive/data')
OUT=ROOT/'data/real_ppe_v2';AUDIT=ROOT/'results/real_ppe_v2_import'
MAP={6:0,0:1,2:2}

def main():
    assert not OUT.exists(),'Do not overwrite an imported dataset.'
    AUDIT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(parents=True);(OUT/'labels').mkdir()
    report={'source':str(SOURCE),'mapping':MAP,'ignored_classes':Counter(),'orphan_labels':[],
            'excluded':[],'duplicate_images':[],'clipped_boxes':0,'splits':{}}
    records=[];seen={}
    # Holdout takes precedence if identical images were supplied in multiple splits.
    for split in ['test','val','train']:
        images=sorted(p for p in (SOURCE/'images'/split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
        stems={p.stem for p in images}
        report['orphan_labels'] += [str(p) for p in (SOURCE/'labels'/split).glob('*.txt') if p.stem not in stems]
        names=[];counts=Counter();negative=0
        for image in images:
            label=SOURCE/'labels'/split/(image.stem+'.txt')
            try:
                with Image.open(image) as im:im.load()
                lines=[]
                for line in label.read_text(encoding='utf-8-sig').splitlines():
                    if not line.strip():continue
                    f=line.split();assert len(f)==5,'Not a YOLO box'
                    c=int(f[0]);x,y,w,h=map(float,f[1:])
                    if c not in MAP:report['ignored_classes'][c]+=1;continue
                    assert all(math.isfinite(v) for v in (x,y,w,h)) and w>0 and h>0,'Invalid box'
                    a,b,d,e=x-w/2,y-h/2,x+w/2,y+h/2
                    clipped=[max(0,a),max(0,b),min(1,d),min(1,e)]
                    assert clipped[2]>clipped[0] and clipped[3]>clipped[1],'Box outside image'
                    if any(abs(v-u)>1e-6 for v,u in zip(clipped,[a,b,d,e])):report['clipped_boxes']+=1
                    a,b,d,e=clipped
                    lines.append(f'{MAP[c]} {(a+d)/2:.6f} {(b+e)/2:.6f} {d-a:.6f} {e-b:.6f}')
            except (OSError,ValueError,AssertionError) as error:
                report['excluded'].append({'image':str(image),'reason':str(error)});continue
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen:
                report['duplicate_images'].append({'image':str(image),'kept':seen[digest]});continue
            name=split+'__'+image.name;seen[digest]=name;names.append(name)
            shutil.copy2(image,OUT/'images'/name)
            content='\n'.join(lines)+('\n' if lines else '')
            (OUT/'labels'/(Path(name).stem+'.txt')).write_text(content)
            tally=Counter(int(l.split()[0]) for l in lines);counts.update(tally);negative+=not lines
            records.append({'image':name,'split':split,'sha256':digest,'objects':dict(tally),
                'label_sha256':hashlib.sha256(content.encode()).hexdigest()})
        assert all(counts[c]>0 for c in range(3)),(split,counts)
        (OUT/f'{split}.txt').write_text('\n'.join('./images/'+n for n in names)+'\n')
        report['splits'][split]={'images':len(names),'objects':dict(counts),'negative_images':negative}
    (OUT/'data.yaml').write_text('path: '+json.dumps(OUT.resolve().as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
    (OUT/'classes.txt').write_text('worker\nhelmet\nvest\n')
    (OUT/'annotation_manifest.json').write_text(json.dumps({'complete':True,'names':['worker','helmet','vest'],
        'images':[r['image'] for r in records],'status':'supplied_annotations_remapped_not_exhaustively_manually_verified',
        'source_class_mapping':MAP,'split_policy':'Supplied splits preserved; exact duplicates retained only once, test before val before train.'},indent=2))
    (AUDIT/'records.json').write_text(json.dumps(records,indent=2));(AUDIT/'report.json').write_text(json.dumps(report,indent=2))
    for split in ['train','val','test']:
        sample=[r for r in records if r['split']==split];random.Random(42).shuffle(sample)
        canvas=Image.new('RGB',(1500,1050),'#222');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(sample[:12]):
            im=Image.open(OUT/'images'/r['image']).convert('RGB');w,h=im.size;pen=ImageDraw.Draw(im)
            for line in (OUT/'labels'/(Path(r['image']).stem+'.txt')).read_text().splitlines():
                c,x,y,bw,bh=map(float,line.split());pen.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline=['cyan','yellow','lime'][int(c)],width=3)
            im.thumbnail((490,230));x=i%3*500;y=i//3*260;canvas.paste(im,(x,y+25));draw.text((x,y),r['image'][:65],fill='white')
        canvas.save(AUDIT/f'{split}_sample.jpg')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
