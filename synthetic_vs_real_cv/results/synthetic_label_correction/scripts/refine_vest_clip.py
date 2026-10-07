import torch,clip,json
from pathlib import Path
from PIL import Image,ImageDraw
from collections import defaultdict
from torchvision.ops import nms
torch.set_num_threads(4)
m,prep=clip.load('weights/clip/ViT-B-32.pt',device='cpu')
texts=['a worker wearing a reflective safety vest','a high visibility orange or yellow safety vest over a shirt','a plain dark work shirt','a plain grey work uniform','a human face','a piece of factory machinery']
with torch.no_grad(): tf=m.encode_text(clip.tokenize(texts));tf/=tf.norm(dim=-1,keepdim=True)
root=Path('synthetic_vs_real_cv');dest=root/'results/synthetic_label_correction/ai_generated';records=[];pending=[];features=[];kept=defaultdict(list)
def flush():
 if not features:return
 with torch.no_grad():f=m.encode_image(torch.stack(features));f/=f.norm(dim=-1,keepdim=True);scores=(f@tf.T).tolist()
 for r,s in zip(pending,scores):
  margin=max(s[:2])-max(s[2:]);r['margin']=margin;r['keep']=margin>.008
  if r['keep']:kept[r['image']].append(r['line'])
  records.append(r)
 pending.clear();features.clear()
for idx,p in enumerate(sorted((root/'data/ai_generated/images').glob('*'))):
 im=Image.open(p).convert('RGB');w,h=im.size;boxes=[];lines=[]
 for lp in [dest/'labels'/(p.stem+'.txt'),root/'data/ai_generated/labels'/(p.stem+'.txt')]:
  for line in lp.read_text().splitlines():
   if not line.strip():continue
   c,x,y,bw,bh=map(float,line.split())
   if c!=2:continue
   boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]);lines.append(line)
 if boxes:
  inds=nms(torch.tensor(boxes,dtype=torch.float32),torch.linspace(1,.5,len(boxes)),.4)
  for j in inds.tolist():
   x1,y1,x2,y2=boxes[j];dx=(x2-x1)*.1;dy=(y2-y1)*.1
   crop=im.crop((max(0,x1-dx),max(0,y1-dy),min(w,x2+dx),min(h,y2+dy)))
   pending.append({'image':p.name,'line':lines[j]});features.append(prep(crop))
   if len(features)==24:flush()
 if (idx+1)%100==0:print(idx+1,flush=True)
flush()
for p in (root/'data/ai_generated/images').glob('*'):
 lp=dest/'labels'/(p.stem+'.txt');lines=[l for l in lp.read_text().splitlines() if l.strip() and l.split()[0]!='2']+kept[p.name]
 lp.write_text('\n'.join(lines)+'\n')
(dest/'vest_classification.json').write_text(json.dumps(records,indent=2))
print('Vest crops:',len(records),'kept:',sum(r['keep'] for r in records),flush=True)
# Review rejected crops nearest the decision threshold, plus the most confident retained crops.
for kind,rs in [('rejected',[r for r in records if not r['keep']]),('accepted',[r for r in records if r['keep']])]:
 rs.sort(key=lambda r:r['margin'],reverse=(kind=='rejected'))
 for page in range((min(len(rs),160)+39)//40):
  canvas=Image.new('RGB',(1600,1000),'#222');draw=ImageDraw.Draw(canvas)
  for j,r in enumerate(rs[page*40:page*40+40]):
   im=Image.open(root/'data/ai_generated/images'/r['image']).convert('RGB');w,h=im.size;c,x,y,bw,bh=map(float,r['line'].split());im=im.crop((max(0,(x-bw*.65)*w),max(0,(y-bh*.65)*h),min(w,(x+bw*.65)*w),min(h,(y+bh*.65)*h)));im.thumbnail((195,170));xx=j%8*200;yy=j//8*200;canvas.paste(im,(xx,yy+25));draw.text((xx,yy),f"{r['image']} {r['margin']:.3f}",fill='white')
  canvas.save(dest/f'vest_{kind}_{page+1}.jpg')
