import json,torch,clip
from pathlib import Path
from PIL import Image,ImageDraw
torch.set_num_threads(4)
root=Path('synthetic_vs_real_cv');d=root/'results/synthetic_label_correction/ai_generated'
m,prep=clip.load('weights/clip/ViT-B-32.pt',device='cpu')
with torch.no_grad():
 t=m.encode_text(clip.tokenize(['a construction hard hat','a safety hard helmet','a welding face shield','an aluminized protective hood','a bare human head','a cloth cap']));t/=t.norm(dim=-1,keepdim=True)
rows=[];pending=[];batch=[]
def flush():
 if not batch:return
 with torch.no_grad():
  f=m.encode_image(torch.stack(batch));f/=f.norm(dim=-1,keepdim=True);scores=(f@t.T).tolist()
 for r,s in zip(pending,scores):r['margin']=max(s[:2])-max(s[2:]);rows.append(r)
 pending.clear();batch.clear()
for p in sorted((d/'labels').glob('*.txt')):
 im=Image.open(root/'data/ai_generated/images'/(p.stem+'.jpg')).convert('RGB');w,h=im.size
 for line in p.read_text().splitlines():
  if not line.strip() or line.split()[0]!='1':continue
  c,x,y,bw,bh=map(float,line.split());crop=im.crop((max(0,(x-bw*.7)*w),max(0,(y-bh*.7)*h),min(w,(x+bw*.7)*w),min(h,(y+bh*.7)*h)))
  pending.append({'image':p.stem+'.jpg','line':line});batch.append(prep(crop))
  if len(batch)==32:flush()
flush();rows.sort(key=lambda r:r['margin']);(d/'helmet_review_order.json').write_text(json.dumps(rows,indent=2))
flagged=[r for r in rows if r['margin']<.015]
print('total',len(rows),'flagged',len(flagged),flush=True)
for page in range((len(flagged)+59)//60):
 canvas=Image.new('RGB',(1500,1000),'#222');draw=ImageDraw.Draw(canvas)
 for j,r in enumerate(flagged[page*60:page*60+60]):
  im=Image.open(root/'data/ai_generated/images'/r['image']).convert('RGB');w,h=im.size;c,x,y,bw,bh=map(float,r['line'].split());im=im.crop((max(0,(x-bw*.7)*w),max(0,(y-bh*.7)*h),min(w,(x+bw*.7)*w),min(h,(y+bh*.7)*h)));im.thumbnail((145,140));xx=j%10*150;yy=j//10*165;canvas.paste(im,(xx,yy+22));draw.text((xx,yy),f"{page*60+j+1} {r['image'][4:8]}",fill='white')
 canvas.save(d/f'helmet_review_{page+1:02}.jpg')
