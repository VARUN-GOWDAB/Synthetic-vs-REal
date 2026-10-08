"""Run the local SynthReal dashboard: python -m src.dashboard.server (project directory)."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit
import argparse,base64,hashlib,io,json,math,os,secrets,shutil,threading,time,traceback
from PIL import Image,ImageDraw,ImageOps
from .registry import ROOT,get_registry
from .alarms import PPETracker,associate

os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'Ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
STATIC=Path(__file__).parent/'static'
TOKEN=secrets.token_urlsafe(32)
INFERENCE_LOCK=threading.Lock()
COLORS=['#20c9bd','#ffbb55','#8ba7ff']

def jpeg(image):
    buf=io.BytesIO();image.save(buf,format='JPEG',quality=85)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()

def draw(image,detections):
    image=image.copy();pen=ImageDraw.Draw(image)
    for d in detections:
        color=COLORS[d['class_id']];box=d['box'];pen.rectangle(box,outline=color,width=3)
        label=d['class_name']+(f" {d['confidence']:.0%}" if 'confidence' in d else '')
        x,y=box[:2];y=max(0,y-16);text=pen.textbbox((x,y),label)
        pen.rectangle(text,fill='#10222e');pen.text((x,y),label,fill=color)
    return image

def number(value,low,high):
    result=float(value)
    if not math.isfinite(result) or not low<=result<=high:raise ValueError(f'Value must be between {low} and {high}')
    return result

class Engine:
    def __init__(self):self.model=None;self.model_id=None;self.sessions={};self.loaded=None;self.loaded_preview=True
    def load(self,entry):
        if self.model_id==entry['id'] and self.model is not None:return
        from ultralytics import YOLO
        import torch
        import ultralytics.utils.torch_utils as tu
        tu.NUM_THREADS=2;torch.set_num_threads(2)
        source=Path(entry['checkpoint']);cache=ROOT/'.runtime/dashboard_cache';cache.mkdir(parents=True,exist_ok=True)
        if entry.get('sha256') and hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Bundled checkpoint checksum mismatch. Restore the model file from the repository.')
        snapshot=cache/'selected_model.pt';temp=cache/'selected_model.tmp'
        for attempt in range(3):
            before=source.stat();shutil.copyfile(source,temp);after=source.stat()
            if (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns):break
            time.sleep(.2)
        else:raise ValueError('Checkpoint is being saved. Try again shortly.')
        temp.replace(snapshot)
        self.model=None;self.model_id=None
        model=YOLO(str(snapshot))
        if list(model.names.values())!=['worker','helmet','vest']:raise ValueError('Checkpoint class mapping does not match the registry.')
        self.model=model;self.model_id=entry['id'];self.loaded=time.strftime('%H:%M:%S');self.loaded_preview=entry['preview']
    def predict(self,payload):
        confidence=number(payload.get('confidence',.35),.05,.95)
        threshold=number(payload.get('iou',.5),.1,.9)
        raw=payload.get('image','')
        if not isinstance(raw,str) or ',' not in raw:raise ValueError('Supply an encoded image.')
        image=Image.open(io.BytesIO(base64.b64decode(raw.split(',',1)[1],validate=True)))
        if image.width*image.height>20_000_000:raise ValueError('Use an image smaller than 20 megapixels.')
        image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((1920,1920))
        registry=get_registry();entries={r['id']:r for r in registry['models']}
        ids=payload.get('models',[])
        if not isinstance(ids,list) or not 1<=len(ids)<=6 or any(not isinstance(x,str) for x in ids):raise ValueError('Select between one and six models.')
        live=bool(payload.get('live'));session=payload.get('session','')
        if live and (len(ids)!=1 or not isinstance(session,str) or not 1<=len(session)<=100):raise ValueError('Live inference needs one model and a session ID.')
        required=payload.get('required',['helmet','vest'])
        if not isinstance(required,list) or any(x not in ['helmet','vest'] for x in required):raise ValueError('Invalid PPE requirements.')
        persistence=number(payload.get('persistence',2),.5,15)
        results=[]
        for model_id in ids:
            entry=entries.get(model_id)
            if not entry or not entry['checkpoint_available']:raise ValueError('Selected model has no checkpoint yet.')
            self.load(entry);start=time.monotonic()
            prediction=self.model.predict(image,conf=confidence,iou=threshold,imgsz=640,device='cpu',verbose=False)[0]
            detections=[]
            for xyxy,cls,conf in zip(prediction.boxes.xyxy.tolist(),prediction.boxes.cls.tolist(),prediction.boxes.conf.tolist()):
                detections.append({'class_id':int(cls),'class_name':['worker','helmet','vest'][int(cls)],'confidence':float(conf),'box':[round(v,1) for v in xyxy]})
            elapsed=time.monotonic()-start;people=[]
            if live:
                now=time.monotonic();self.sessions={k:v for k,v in self.sessions.items() if now-v['last']<60}
                key=(session,model_id,tuple(required),confidence,threshold,persistence)
                if key not in self.sessions:
                    if len(self.sessions)>=32:self.sessions.pop(next(iter(self.sessions)))
                    self.sessions[key]={'tracker':PPETracker(),'last':now}
                state=self.sessions[key];state['last']=now
                people=state['tracker'].update(associate(detections,image.width,image.height,required),now,persistence)
            results.append({'model_id':model_id,'name':entry['name'],'preview':self.loaded_preview,
                'image':jpeg(draw(image,detections)),'detections':detections,'people':people,
                'width':image.width,'height':image.height,'inference_ms':round(elapsed*1000),
                'checkpoint_loaded_at':self.loaded})
        return {'results':results}

ENGINE=Engine()

class Handler(BaseHTTPRequestHandler):
    def send(self,status,data,kind='application/json'):
        encoded=json.dumps(data).encode() if kind=='application/json' else data
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(encoded)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer');self.end_headers();self.wfile.write(encoded)
    def valid_host(self):
        return self.headers.get('Host','') in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']
    def do_GET(self):
        if not self.valid_host():return self.send(403,{'error':'Local requests only.'})
        parts=urlsplit(self.path)
        try:
            if parts.path=='/api/registry':return self.send(200,{**get_registry(),'token':TOKEN})
            if parts.path=='/api/health':return self.send(200,{'status':'ok'})
            if parts.path=='/api/review':return self.send(400,{'error':'Test-set review is unavailable in inference-only mode.'})
            assets={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8'),'/comparison.css':('comparison.css','text/css; charset=utf-8')}
            if parts.path not in assets:return self.send(404,{'error':'Not found'})
            file,kind=assets[parts.path];return self.send(200,(STATIC/file).read_bytes(),kind)
        except (ValueError,OSError) as e:return self.send(400,{'error':str(e)})
    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-SynthReal-Token')!=TOKEN:return self.send(403,{'error':'Refresh this local dashboard before retrying.'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=12_000_000:return self.send(413,{'error':'Request too large; use an image below 8 MB.'})
            payload=json.loads(self.rfile.read(length))
            if not isinstance(payload,dict):raise ValueError('Expected a JSON object.')
            if self.path=='/api/predict':
                if not INFERENCE_LOCK.acquire(blocking=False):return self.send(409,{'error':'Inference is busy. Try again shortly.'})
                try:result=ENGINE.predict(payload)
                finally:INFERENCE_LOCK.release()
                return self.send(200,result)
            if self.path=='/api/reset':
                with INFERENCE_LOCK:
                    ENGINE.sessions.clear()
                    if payload.get('reload'):ENGINE.model=None;ENGINE.model_id=None
                return self.send(200,{'ok':True})
            if self.path=='/api/review':return self.send(400,{'error':'Test-set review is unavailable in inference-only mode.'})
            return self.send(404,{'error':'Not found'})
        except (ValueError,KeyError,OSError,Image.DecompressionBombError) as e:return self.send(400,{'error':str(e)})
        except Exception:
            traceback.print_exc();return self.send(500,{'error':'Inference failed. Check the server log; check that the bundled checkpoint and dependencies are available.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(f'SynthReal dashboard: http://127.0.0.1:{a.port}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
