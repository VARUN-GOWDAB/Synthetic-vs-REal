"""Render every selected source image with numbered annotations for visual review."""
from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/full_visual_review_v4'
def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for source in ['real_safety_500_v3','ai_generated_v3']:
        base=ROOT/'data'/source;manifest=json.loads((base/'annotation_manifest.json').read_text());splits={}
        if source.startswith('real'):
            for split in ['train','val','test']:
                for line in (base/f'{split}.txt').read_text().splitlines():splits[Path(line).name]=split
        for name in manifest['images']:
            p=base/'images'/name;lp=base/'labels'/(p.stem+'.txt')
            rows.append({'id':len(rows)+1,'source':source,'image':name,'split':splits.get(name,'synthetic_pool'),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'labels':lp.read_text().splitlines()})
    (OUT/'inventory.json').write_text(json.dumps(rows,indent=2))
    for start in range(0,len(rows),4):
        canvas=Image.new('RGB',(1920,1400),'#202020');pen=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[start:start+4]):
            im=Image.open(ROOT/'data'/r['source']/'images'/r['image']).convert('RGB');im.thumbnail((950,650));w,h=im.size;draw=ImageDraw.Draw(im)
            for k,line in enumerate(r['labels']):
                c,x,y,bw,bh=map(float,line.split());color=['#00ffff','#ffff00','#ff55ff'][int(c)];box=[max(0,(x-bw/2)*w),max(0,(y-bh/2)*h),min(w-1,(x+bw/2)*w),min(h-1,(y+bh/2)*h)];draw.rectangle(box,outline=color,width=1);draw.text((box[0],box[1]),str(k),fill=color,stroke_width=1,stroke_fill='#000000')
            x=j%2*960;y=j//2*700;canvas.paste(im,(x,y+35));pen.text((x+5,y+5),f"IMAGE {r['id']} {r['image']} {r['split']} | cyan worker / yellow helmet / magenta vest | label IDs start at 0",fill='white')
        canvas.save(OUT/f"page_{start//4+1:03}.jpg",quality=92)
    print('Images',len(rows),'pages',(len(rows)+3)//4,flush=True)
if __name__=='__main__':main()
