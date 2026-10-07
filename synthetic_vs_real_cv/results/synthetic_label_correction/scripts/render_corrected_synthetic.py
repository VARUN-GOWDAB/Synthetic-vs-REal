from pathlib import Path
from PIL import Image,ImageDraw
root=Path('synthetic_vs_real_cv');base=root/'results/synthetic_label_correction'
for name in ['ai_generated','3d_rendered']:
 ims=sorted((root/'data'/name/'images').glob('*'));dest=base/name
 if len(list((dest/'labels').glob('*.txt')))!=len(ims):continue
 for page in range((len(ims)+19)//20):
  canvas=Image.new('RGB',(1600,1250),'#222');draw=ImageDraw.Draw(canvas)
  for j,p in enumerate(ims[page*20:page*20+20]):
   im=Image.open(p).convert('RGB');w,h=im.size;dd=ImageDraw.Draw(im)
   for line in (dest/'labels'/(p.stem+'.txt')).read_text().splitlines():
    if not line.strip():continue
    c,x,y,bw,bh=map(float,line.split());dd.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline=['cyan','yellow','lime'][int(c)],width=max(2,w//400))
   im.thumbnail((395,218));x=j%4*400;y=j//4*250;canvas.paste(im,(x,y+26));draw.text((x+3,y+3),f'{page*20+j+1} {p.name}',fill='white')
  canvas.save(dest/f'page_{page+1:02}.jpg')
 print(name,'previews ready',flush=True)
