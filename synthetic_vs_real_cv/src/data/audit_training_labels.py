"""Generate independent detector suggestions for visual review; never overwrite labels."""
from pathlib import Path
import os,json,math,sys
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'Ultralytics')
os.environ['YOLO_AUTOINSTALL']='false'
import torch,clip
from PIL import Image,ImageDraw
from ultralytics import YOLOWorld

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def main():
    torch.set_num_threads(4)
    original=clip.load
    clip.load=lambda name,*a,**kw:original(str(ROOT.parent/'weights/clip/ViT-B-32.pt'),*a,**kw)
    if '--render-only' not in sys.argv:
        model=YOLOWorld(str(ROOT.parent/'weights/yolov8s-worldv2.pt'))
        model.set_classes(['person','safety hard hat','high visibility safety vest'])
    out=ROOT/'results/annotation_audit_v3';out.mkdir(exist_ok=True)
    rows=[]
    # Audit every selected training image in both source datasets. Holdouts are not relabeled from predictions.
    for source in ['real_safety_500','ai_generated']:
        base=ROOT/'data'/source
        names=[Path(x).name for x in (base/'train.txt').read_text().splitlines()] if source=='real_safety_500' else json.loads((base/'annotation_manifest.json').read_text())['images']
        for name in names:rows.append((source,name))
    predictions=out/'predictions.jsonl';done={}
    if predictions.exists():
        for line in predictions.read_text().splitlines():
            r=json.loads(line);done[(r['source'],r['image'])]=r
    for n,(source,name) in enumerate(rows,1):
        if '--render-only' in sys.argv:break
        if (source,name) in done:continue
        base=ROOT/'data'/source;image=base/'images'/name
        result=model.predict(str(image),conf=.2,imgsz=768,device='cpu',verbose=False)[0]
        r={'source':source,'image':name,'predictions':[{'c':int(c),'box':b,'confidence':s} for c,b,s in zip(result.boxes.cls.tolist(),result.boxes.xyxyn.tolist(),result.boxes.conf.tolist())]}
        with predictions.open('a') as f:f.write(json.dumps(r)+'\n')
        done[(source,name)]=r
        if n%25==0:print(f'Audited {n}/{len(rows)}',flush=True)
    candidates=[]
    for source,name in rows:
        if (source,name) not in done:continue
        base=ROOT/'data'/source;old=[]
        for line in (base/'labels'/(Path(name).stem+'.txt')).read_text().splitlines():
            c,x,y,w,h=map(float,line.split());old.append({'c':int(c),'box':[x-w/2,y-h/2,x+w/2,y+h/2],'line':line})
        for pred in done[(source,name)]['predictions']:
            if pred['confidence']<(.5 if pred['c']==0 else .3):continue
            matches=[(iou(pred['box'],b['box']),i) for i,b in enumerate(old) if b['c']==pred['c']]
            overlap,index=max(matches,default=(0,-1))
            if overlap>=.65:continue
            candidates.append({'id':len(candidates)+1,'source':source,'image':name,**pred,'overlap':overlap,'old':old[index] if overlap>=.2 else None,'action':'adjust' if overlap>=.2 else 'add'})
    (out/'candidates.json').write_text(json.dumps(candidates,indent=2))
    # Large contextual crops permit review of both proposed (magenta) and existing (cyan) geometry.
    for start in range(0,len(candidates),24):
        canvas=Image.new('RGB',(1800,1440),'#222');pen=ImageDraw.Draw(canvas)
        for j,r in enumerate(candidates[start:start+24]):
            im=Image.open(ROOT/'data'/r['source']/'images'/r['image']).convert('RGB');w,h=im.size
            b=r['box'];old=r['old'];union=b if old is None else [min(b[0],old['box'][0]),min(b[1],old['box'][1]),max(b[2],old['box'][2]),max(b[3],old['box'][3])]
            bw=union[2]-union[0];bh=union[3]-union[1];x0=max(0,(union[0]-bw*.5)*w);y0=max(0,(union[1]-bh*.4)*h);x1=min(w,(union[2]+bw*.5)*w);y1=min(h,(union[3]+bh*.4)*h)
            draw=ImageDraw.Draw(im)
            if old:draw.rectangle([old['box'][0]*w,old['box'][1]*h,old['box'][2]*w,old['box'][3]*h],outline='cyan',width=3)
            draw.rectangle([b[0]*w,b[1]*h,b[2]*w,b[3]*h],outline='magenta',width=3)
            im=im.crop((int(x0),int(y0),int(x1),int(y1)));im.thumbnail((295,320));xx=j%6*300;yy=j//6*360;canvas.paste(im,(xx,yy+35));pen.text((xx,yy),f"{r['id']} {r['action']} {['person','helmet','vest'][r['c']]} {r['confidence']:.2f}",fill='white');pen.text((xx,yy+15),r['image'],fill='white')
        canvas.save(out/f"review_{start//24+1:03}.jpg")
    print('Review candidates:',len(candidates),flush=True)

if __name__=='__main__':main()
