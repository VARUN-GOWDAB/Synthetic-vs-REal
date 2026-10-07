from pathlib import Path
import torch
from torchvision.ops import nms
root=Path('synthetic_vs_real_cv');d=root/'results/synthetic_label_correction/3d_rendered/labels'
for p in d.glob('*.txt'):
 lines=[l for l in p.read_text().splitlines() if l.strip()]
 lines += [l for l in (root/'data/3d_rendered/labels'/p.name).read_text().splitlines() if l.split()[0]=='1']
 out=[]
 for cls in range(3):
  rows=[l for l in lines if int(l.split()[0])==cls];boxes=[]
  for line in rows:
   c,x,y,w,h=map(float,line.split());boxes.append([x-w/2,y-h/2,x+w/2,y+h/2])
  if not boxes:continue
  indexes=nms(torch.tensor(boxes,dtype=torch.float32),torch.linspace(1,.5,len(boxes)),.4)
  out.extend(rows[i] for i in indexes.tolist())
 if p.stem=='synthetic_000008':out.append('0 0.466797 0.520833 0.052344 0.361111')
 p.write_text('\n'.join(out)+'\n')
print('Restored original rendered helmet coverage and corrected occluded worker in frame 8.')
