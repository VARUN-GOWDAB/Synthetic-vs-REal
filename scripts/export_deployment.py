"""Bundle only completed/evaluated checkpoints; never launch or stop training."""
from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'synthetic_vs_real_cv'))
from src.dashboard.registry import LABELS

def main():
    project=ROOT/'synthetic_vs_real_cv';out=ROOT/'deployment';(out/'models').mkdir(parents=True,exist_ok=True)
    previous=json.loads((out/'registry.json').read_text()) if (out/'registry.json').exists() else {'models':[]}
    models={m['id']:m for m in previous['models']}
    active=json.loads((project/'configs/active_dataset.json').read_text())
    for config_name,run in [(active['config'],active['run'])]:
        file=project/'configs'/config_name
        if not file.exists():continue
        config=json.loads(file.read_text(encoding='utf-8'))
        for e in config['experiments']:
            directory=project/'results/runs'/run/e['name'];metrics=directory/'test_metrics.json';source=directory/'weights/best.pt'
            if not metrics.exists() or not source.exists():continue
            result=json.loads(metrics.read_text(encoding='utf-8'))
            target=out/'models'/f"{e['name']}.pt"
            temporary=target.with_suffix('.tmp')
            shutil.copyfile(source,temporary)
            temporary.replace(target)
            assert target.stat().st_size<95*1024*1024,'Model too large for regular Git; use a release asset instead.'
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
            models[e['name']]={'id':e['name'],'name':LABELS.get(e['name'].removesuffix('_v3'),e['name'])+(' - corrected labels' if e['name'].endswith('_v3') else ''),
                'dataset_version':config.get('version',2),
                'checkpoint':target.relative_to(out).as_posix(),'sha256':digest,
                'train_images':e['train_images'],'ratios':{k:round(100*e['sources'].get(s,0)/e['train_images'],2) for k,s in [('AI',config.get('ai_dataset','ai_generated')),('Blender','3d_rendered'),('Real',active['real_dataset'])]},
                'metrics':result['metrics'],'per_class':result.get('per_class_ap50_95'),
                'settings':{k:config[k] for k in ['model','epochs','imgsz','batch','optimizer','lr0','seed','device']},
                'dataset':'Fixed real holdout; datasets are not required or bundled for inference.',
                'val_images':e['val_images'],'test_images':e['test_images'],
                'annotation_status':result['annotation_status']}
    if not models:raise RuntimeError('No completed and evaluated models are available yet.')
    (out/'registry.tmp').write_text(json.dumps({'version':1,'classes':['worker','helmet','vest'],
        'dataset_summary':{'real':500,'ai':656,'blender':400},'models':list(models.values())},indent=2),encoding='utf-8')
    (out/'registry.tmp').replace(out/'registry.json')
    print(f'Bundled {len(models)} completed models. Add deployment/ to your Git commit.')

if __name__=='__main__':main()
