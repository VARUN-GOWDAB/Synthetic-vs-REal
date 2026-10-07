"""Registry entries are derived from actual checkpoints and completed evaluations."""
from pathlib import Path
import hashlib,json,threading,os

ROOT=Path(__file__).resolve().parents[2]
LOCK=threading.Lock()
LABELS={'real_baseline_420':'Real baseline · 420', 'real_only_400':'D · Real only',
        'ai_only_400':'A · AI only','rendered_only_400':'Blender only',
        'mixed_real50_ai25_rendered25_400':'C · 25 / 25 / 50',
        'mixed_ai50_rendered25_real25_400':'B · 50 / 25 / 25'}

def read(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return default

def active_settings():
    return read(ROOT/'configs/active_dataset.json',{'real_dataset':'real_safety_600','config':'comparison_v1.json','run':'comparison_v1'})

def review_summary():
    decisions=read(ROOT/'results/test_review/decisions.json',{})
    base=ROOT/'data'/active_settings()['real_dataset']
    if not (base/'test.txt').exists():return {'reviewed':0,'total':0,'complete':False,'available':False}
    names=[Path(x).name for x in (base/'test.txt').read_text().splitlines()]
    verified=0
    for name in names:
        label=base/'labels'/f'{Path(name).stem}.txt'
        d=decisions.get(name,{})
        digest=hashlib.sha256(label.read_bytes()).hexdigest()
        image_digest=hashlib.sha256((base/'images'/name).read_bytes()).hexdigest()
        verified+=d.get('status')=='reviewed' and d.get('label_sha256')==digest and d.get('image_sha256')==image_digest
    return {'reviewed':verified,'total':len(names),'complete':verified==len(names),'available':True}

def deployment_registry():
    base=ROOT.parent/'deployment'
    manifest=read(base/'registry.json')
    if not manifest:raise ValueError('Deployment registry missing. Export completed models before sharing the repository.')
    models=[]
    for entry in manifest['models']:
        checkpoint=(base/entry['checkpoint']).resolve()
        if not checkpoint.is_relative_to(base.resolve()):raise ValueError('Invalid deployment checkpoint path.')
        available=checkpoint.is_file()
        models.append({**entry,'checkpoint':str(checkpoint) if available else None,
            'checkpoint_available':available,'status':'evaluated' if available else 'missing checkpoint',
            'preview':False,'epoch':None})
    return {'mode':'deployment','models':models,'queues':[],
        'test_review':{'reviewed':0,'total':0,'complete':False,'available':False},
        'classes':manifest['classes'],'b_queue':{},'dataset_summary':manifest.get('dataset_summary',{})}

def get_registry():
    with LOCK:
        active=active_settings()
        if os.environ.get('SYNTHREAL_DEPLOYMENT')=='1' or not (ROOT/'configs'/active['config']).exists():
            return deployment_registry()
        models=[];queues=[]
        for config_name,run_name in [(active['config'],active['run'])]:
            config=read(ROOT/'configs'/config_name)
            if not config:continue
            directory=ROOT/'results/runs'/run_name
            queue=read(directory/'queue_status.json',{})
            queues.append({'name':run_name,**{k:queue.get(k) for k in ['status','current','epoch','total_epochs','error','updated']}})
            for e in config['experiments']:
                checkpoint=directory/e['name']/'weights/best.pt'
                result=read(directory/e['name']/'test_metrics.json',{})
                status='evaluated' if result else ('training' if queue.get('current')==e['name'] and queue.get('status')=='training' else 'pending')
                if queue.get('current')==e['name'] and queue.get('status') in ('evaluating','failed'):status=queue['status']
                if checkpoint.exists() and status=='pending':status='checkpoint available'
                ratios={k:round(100*e['sources'].get(s,0)/e['train_images'],2) for k,s in [('AI',config.get('ai_dataset','ai_generated')),('Blender','3d_rendered'),('Real',active['real_dataset'])]}
                models.append({'id':e['name'],'name':LABELS.get(e['name'].removesuffix('_v3'),e['name'])+(' - corrected labels' if e['name'].endswith('_v3') else ''),
                    'status':status,'train_images':e['train_images'],'ratios':ratios,
                    'checkpoint':str(checkpoint) if checkpoint.exists() else None,
                    'checkpoint_available':checkpoint.exists(),'preview':bool(checkpoint.exists() and not result),
                    'metrics':result.get('metrics'),'per_class':result.get('per_class_ap50_95'),
                    'settings':{k:config[k] for k in ['model','epochs','imgsz','batch','optimizer','lr0','seed','device']},
                    'dataset':e['data'],'val_images':e['val_images'],'test_images':e['test_images'],
                    'annotation_status':config['annotation_status'],
                    'epoch':queue.get('epoch') if queue.get('current')==e['name'] else None})
        manifest=read(ROOT/'data'/active['real_dataset']/'annotation_manifest.json',{})
        data={'mode':'research','models':models,'queues':queues,'test_review':review_summary(),
            'dataset_summary':{'real':len(manifest.get('images',[])),'ai':656,'blender':400},
            'classes':['worker','helmet','vest'],'b_queue':{}}
        path=ROOT/'results/model_registry.json';path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
        return data
