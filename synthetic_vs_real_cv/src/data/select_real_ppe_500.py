"""Select 500 safety images with quality ranking and scene-grouped holdouts."""
from pathlib import Path
from collections import Counter
import json,math,shutil,hashlib,re
import numpy as np
import cv2,torch,clip
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'data/real_ppe_v2';OUT=ROOT/'data/real_safety_500';AUDIT=ROOT/'results/real_ppe_v2_import'

def main():
    import sys
    if "--regroup" in sys.argv:
        candidates=json.loads((AUDIT/"ranked_scene_groups.json").read_text())
        parent=list(range(len(candidates)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        leaders={}
        for i,r in enumerate(candidates):
            key=r["scene_group"]
            if key in leaders:parent[find(i)]=find(leaders[key])
            else:leaders[key]=i
    else:
        torch.set_num_threads(4)
        model,prep=clip.load(str(ROOT.parent/'weights/clip/ViT-B-32.pt'),device='cpu')
        records=json.loads((AUDIT/'records.json').read_text());candidates=[];features=[];batch=[]
        for r in records:
            counts={int(k):v for k,v in r['objects'].items()}
            # This source appends unrelated person/cycling/military examples above image1120.
            number=int(re.search(r'image(\d+)',r['image']).group(1))
            if number>=1120 or counts.get(0,0)>6:continue
            if not counts.get(0) or not (counts.get(1) or counts.get(2)):continue
            p=SOURCE/'images'/r['image'];im=Image.open(p).convert('RGB');arr=np.asarray(im);gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
            blur=float(cv2.Laplacian(cv2.resize(gray,(256,256)),cv2.CV_64F).var())
            visible=float(np.mean((gray>20)&(gray<240)))
            smooth=cv2.medianBlur(gray,3);noise=float(np.mean(np.abs(gray.astype(float)-smooth.astype(float))))
            score=math.log1p(min(blur,250))*.3+visible*5-min(noise,30)*.12+min(len(counts),3)*.3
            candidates.append({**r,'quality_score':score,'blur':blur,'visible_fraction':visible});batch.append(prep(im))
            if len(batch)==32:
                with torch.no_grad():f=model.encode_image(torch.stack(batch));f/=f.norm(dim=-1,keepdim=True);features.extend(f.numpy())
                batch=[]
        if batch:
            with torch.no_grad():f=model.encode_image(torch.stack(batch));f/=f.norm(dim=-1,keepdim=True);features.extend(f.numpy())
        f=np.asarray(features);similar=f@f.T;parent=list(range(len(candidates)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i in range(len(candidates)):
            for j in np.flatnonzero(similar[i,i+1:]>=.88)+i+1:parent[find(int(j))]=find(i)
    # Also group perceptual duplicates, even if CLIP disagrees on an augmentation.
    hashes=[]
    for r in candidates:
        im=Image.open(SOURCE/'images'/r['image']);gray=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32);dct=cv2.dct(gray)[:8,:8].flatten()[1:];bits=dct>np.median(dct);hashes.append(sum(int(b)<<k for k,b in enumerate(bits)))
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            if (hashes[i]^hashes[j]).bit_count()<=8:parent[find(j)]=find(i)
    groups={}
    for i,r in enumerate(candidates):r['scene_group']=find(i);groups.setdefault(find(i),[]).append(r)
    for group in groups.values():group.sort(key=lambda r:r['quality_score'],reverse=True)
    print('Candidates',len(candidates),'scene groups',len(groups),'largest',[len(g) for g in sorted(groups.values(),key=len,reverse=True)[:12]],flush=True)
    (AUDIT/'ranked_scene_groups.json').write_text(json.dumps(candidates,indent=2))
    # Reserve complete small scene groups for holdouts, distributing quality/size.
    pools={s:[] for s in ['train','val','test']};used=set()
    eligible=sorted(groups.items(),key=lambda item:sum(r['quality_score'] for r in item[1])/len(item[1]),reverse=True)
    for split in ['test','val']:
        for key,group in eligible:
            if key in used or len(group)>35:continue
            if len(pools[split])>=50:break
            used.add(key);pools[split].extend(group[:50-len(pools[split])])
    train_groups=[g for k,g in eligible if k not in used]
    # Round-robin highest-quality examples across scenes instead of letting one repeated scene dominate.
    for rank in range(max(map(len,train_groups))):
        for group in train_groups:
            if rank<len(group) and len(pools['train'])<400:pools['train'].append(group[rank])
        if len(pools['train'])==400:break
    assert {k:len(v) for k,v in pools.items()}=={'train':400,'val':50,'test':50}
    assert not OUT.exists()
    (OUT/'images').mkdir(parents=True);(OUT/'labels').mkdir();selected=[]
    for split,pool in pools.items():
        for r in pool:
            selected.append({**r,'original_split':r['split'],'split':split})
            for folder in ['images','labels']:
                name=r['image'] if folder=='images' else Path(r['image']).stem+'.txt';shutil.copy2(SOURCE/folder/name,OUT/folder/name)
        (OUT/f'{split}.txt').write_text('\n'.join('./images/'+r['image'] for r in pool)+'\n')
        for page in range((len(pool)+19)//20):
            canvas=Image.new('RGB',(1600,1250),'#222');pen=ImageDraw.Draw(canvas)
            for j,r in enumerate(pool[page*20:page*20+20]):
                im=Image.open(OUT/'images'/r['image']).convert('RGB');w,h=im.size;draw=ImageDraw.Draw(im)
                for l in (OUT/'labels'/(Path(r['image']).stem+'.txt')).read_text().splitlines():
                    c,x,y,bw,bh=map(float,l.split());draw.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline=['cyan','yellow','lime'][int(c)],width=3)
                im.thumbnail((395,220));x=j%4*400;y=j//4*250;canvas.paste(im,(x,y+25));pen.text((x,y),f"{page*20+j+1} {r['image']}",fill='white')
            canvas.save(AUDIT/f'selected_{split}_{page+1:02}.jpg')
    (OUT/'data.yaml').write_text('path: '+json.dumps(OUT.resolve().as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
    (OUT/'classes.txt').write_text('worker\nhelmet\nvest\n')
    manifest={'complete':True,'names':['worker','helmet','vest'],'images':[r['image'] for r in selected],
        'status':'provided_labels_remapped_and_quality_selected_not_exhaustive_manual_ground_truth',
        'selection':'PPE scene relevance, sharpness/exposure, CLIP similarity >=0.88 and perceptual Hamming <=8 connected scene groups, diverse training selection',
        'counts':{s:len(v) for s,v in pools.items()},'limitation':'Scene grouping is heuristic, not proof of camera/source independence; quality-selected test set is not representative of all factory conditions.'}
    (OUT/'annotation_manifest.json').write_text(json.dumps(manifest,indent=2));(AUDIT/'selected_records.json').write_text(json.dumps(selected,indent=2))
    print(json.dumps(manifest['counts']),flush=True)

if __name__=='__main__':main()
