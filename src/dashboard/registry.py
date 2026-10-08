"""Registry entries come from completed Colab checkpoints and evaluations."""
from pathlib import Path
import json,threading

ROOT=Path(__file__).resolve().parents[2]
LOCK=threading.Lock()

def read(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return default

def deployment_registry():
    base=ROOT/'deployment'
    manifest=read(base/'registry.json')
    if not manifest:raise ValueError('Deployment registry missing. Import a completed Colab export before sharing the repository.')
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
        return deployment_registry()
