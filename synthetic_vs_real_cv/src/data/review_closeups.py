"""Render original or proposed labels for explicitly selected review IDs."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/full_visual_review_v4'
def boxes(row,decision,proposed):
    result=[]
    for k,line in enumerate(row['labels']):
        c,x,y,w,h=map(float,line.split())
        box=[int(c),x-w/2,y-h/2,x+w/2,y+h/2]
        if proposed and k in decision.get('remove',[]):continue
        if proposed:box=decision.get('replace',{}).get(str(k),box)
        result.append((str(k),box))
    if proposed:result.extend((f'A{k}',b) for k,b in enumerate(decision.get('add',[])))
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('ids',type=int,nargs='+');ap.add_argument('--proposed',action='store_true');a=ap.parse_args()
    rows=json.loads((OUT/'inventory.json').read_text());dec=json.loads((OUT/'decisions.json').read_text())
    for i in a.ids:
        r=rows[i-1];im=Image.open(ROOT/'data'/r['source'].replace('_v3','_v4')/'images'/r['image']).convert('RGB')
        scale=1280/max(im.size);im=im.resize((round(im.width*scale),round(im.height*scale)));w,h=im.size;dr=ImageDraw.Draw(im)
        for k,b in boxes(r,dec.get(str(i),{}),a.proposed):
            c,x1,y1,x2,y2=b;color=['cyan','yellow','magenta'][int(c)];xy=[x1*w,y1*h,x2*w,y2*h];dr.rectangle(xy,outline=color,width=2);dr.text(xy[:2],k,fill=color,stroke_width=1,stroke_fill='black')
        im.save(OUT/f"{'proposed' if a.proposed else 'close'}_{i}.jpg",quality=95)
if __name__=='__main__':main()
