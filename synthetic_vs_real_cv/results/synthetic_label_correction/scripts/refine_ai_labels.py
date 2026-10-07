from pathlib import Path
import cv2,numpy as np,json
root=Path('synthetic_vs_real_cv');base=root/'results/synthetic_label_correction';name='ai_generated'
def overlap(a,b):
 x=max(0,min(a[3],b[3])-max(a[1],b[1]));y=max(0,min(a[4],b[4])-max(a[2],b[2]));inter=x*y
 aa=(a[3]-a[1])*(a[4]-a[2]);bb=(b[3]-b[1])*(b[4]-b[2]);return inter/max(1,aa+bb-inter),inter/max(1,min(aa,bb))
def read(p,w,h,score):
 out=[]
 for l in p.read_text().splitlines():
  if not l.strip():continue
  c,x,y,bw,bh=map(float,l.split());out.append([int(c),(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h,score])
 return out
for p in sorted((root/'data'/name/'images').glob('*')):
 im=cv2.imread(str(p));h,w=im.shape[:2];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
 label=base/name/'labels'/(p.stem+'.txt');new=read(label,w,h,.7);old=read(root/'data'/name/'labels'/label.name,w,h,.8)
 boxes=[b for b in old if b[0]==0]+[b for b in new if b[0]==0]
 people=boxes.copy()
 def patch(b):
  return hsv[max(0,int(b[2])):min(h,int(b[4])),max(0,int(b[1])):min(w,int(b[3]))]
 def onperson(b,upper=False):
  x=(b[1]+b[3])/2;y=(b[2]+b[4])/2
  return any(a[1]-4<=x<=a[3]+4 and a[2]-5<=y<=a[4] and (not upper or y<a[2]+.32*(a[4]-a[2])) for a in people)
 for b in new+old:
  if b[0]==0:continue
  pa=patch(b)
  if not pa.size:continue
  H,S,V=cv2.split(pa)
  if b[0]==2:
   # A vest must show high-visibility material, rather than an ordinary dark shirt.
   bright=(H>=4)&(H<=48)&(S>=110)&(V>=115)
   if bright.mean()<.12 or not onperson(b):continue
  elif b[0]==1:
   if not onperson(b,True):continue
   if b in old:
    upper=pa[:max(1,int(len(pa)*.65))];H,S,V=cv2.split(upper)
    bright=((S<75)&(V>155))|((S>100)&(V>130)&(((H>=18)&(H<=45))|(H<4)))
    if bright.mean()<.18 and not any(n[0]==1 and overlap(b,n)[0]>.25 for n in new):continue
  boxes.append(b)
 clean=[]
 for b in sorted(boxes,key=lambda b:b[5],reverse=True):
  if any(b[0]==a[0] and (overlap(b,a)[0]>.45 or (b[0]!=0 and overlap(b,a)[1]>.85)) for a in clean):continue
  clean.append(b)
 label.write_text(''.join(f'{c} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n' for c,x1,y1,x2,y2,s in clean))
print('AI original person coverage retained; equipment cross-checked and duplicates suppressed.')
