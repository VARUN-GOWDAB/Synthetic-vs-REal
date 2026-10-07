from pathlib import Path
import json, hashlib, math
import cv2
import numpy as np
from PIL import Image,ImageDraw
root=Path('synthetic_vs_real_cv'); src=root/'data/real_3class_draft'; out=root/'results/curation_600'; out.mkdir(exist_ok=True)
names=json.loads((src/'annotation_manifest.json').read_text())['images']
rows=[]
for name in names:
    p=src/'images'/name
    im=cv2.imread(str(p))
    if im is None: continue
    h,w=im.shape[:2]
    if min(w,h)<220: continue
    boxes=[list(map(float,l.split())) for l in (src/'labels'/(p.stem+'.txt')).read_text().splitlines()]
    people=[b for b in boxes if b[0]==0]; hats=[b for b in boxes if b[0]==1]; vests=[b for b in boxes if b[0]==2]
    if not people or not hats or len(people)>12 or len(hats)>12: continue
    def inside(b,p):
        return p[1]-p[3]/2 <= b[1] <= p[1]+p[3]/2 and p[2]-p[4]/2 <= b[2] <= p[2]+p[4]/2
    hat_match=sum(any(inside(b,p) and b[2]<=p[2]+p[4]*.1 for p in people) for b in hats)/len(hats)
    vest_match=sum(any(inside(b,p) and b[3]<=p[3]*1.2 and b[4]<=p[4]*.9 for p in people) for b in vests)/max(1,len(vests))
    if hat_match<1 or (vests and vest_match<1): continue
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray,(min(640,w),max(1,round(h*min(640,w)/w))))
    sharp=float(cv2.Laplacian(small,cv2.CV_64F).var())
    if sharp<35: continue
    original=(root/'data/real/labels'/(p.stem+'.txt')).read_text().splitlines()
    gap=max(0,len(original)-len(people))
    if gap>1: continue
    tiny=sum(b[3]*w<12 or b[4]*h<12 for b in boxes)
    if tiny: continue
    score=math.log1p(sharp)*2+math.log1p(w*h)/2 -len(people)*.2-gap*4
    if vests: score+= min(math.sqrt(b[3]*b[4]) for b in vests)*20
    ph=cv2.resize(gray,(9,8)); bits=(ph[:,1:]>ph[:,:-1]).flatten(); dhash=sum(int(v)<<i for i,v in enumerate(bits))
    rows.append({'image':name,'score':score,'worker':len(people),'helmet':len(hats),'vest':len(vests),'sharpness':sharp,'size':[w,h],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'dhash':dhash})
rows.sort(key=lambda r:(r['vest']>0,r['score']),reverse=True)
selected=[]; hashes=set(); phashes=[]
for r in rows:
    if r['sha256'] in hashes or any((r['dhash']^v).bit_count()<=3 for v in phashes): continue
    selected.append(r);hashes.add(r['sha256']);phashes.append(r['dhash'])
    if len(selected)==1280: break
(out/'ranked_candidates.json').write_text(json.dumps(selected,indent=2))
print(json.dumps({'eligible':len(rows),'unique_candidates':len(selected),'vest_candidates':sum(r['vest']>0 for r in selected)},indent=2),flush=True)
for page in range((len(selected)+19)//20):
    canvas=Image.new('RGB',(1600,1250),'#202020'); draw=ImageDraw.Draw(canvas)
    for j,row in enumerate(selected[page*20:(page+1)*20]):
        p=src/'images'/row['image']; x=(j%4)*400; y=(j//4)*250
        with Image.open(p).convert('RGB') as im:
            w,h=im.size; d=ImageDraw.Draw(im)
            for line in (src/'labels'/(p.stem+'.txt')).read_text().splitlines():
                cls,cx,cy,bw,bh=map(float,line.split()); color=['cyan','yellow','lime'][int(cls)]
                d.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),outline=color,width=max(2,w//350))
            im.thumbnail((392,218));canvas.paste(im,(x,y+28))
        draw.text((x+3,y+3),f"{page*20+j+1} {p.stem} P{row['worker']} H{row['helmet']} V{row['vest']}",fill='white')
    canvas.save(out/f'page_{page+1:02}.jpg')

