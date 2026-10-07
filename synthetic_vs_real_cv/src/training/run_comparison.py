"""Run frozen comparison experiments sequentially and save actual held-out metrics."""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, time, traceback, subprocess, sys

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(WORKSPACE/'Ultralytics'))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('YOLO_VERBOSE','false')

def save(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,default=str))
    temp.replace(path)

def main():
    active=json.loads((ROOT/'configs/active_dataset.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/active['config'])
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume interrupted queue from saved checkpoints.')
    parser.add_argument('--output',type=Path,default=ROOT/'results/runs'/active['run'])
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if cfg.get('ready_for_training') is False and not args.check_only:
        raise SystemExit('Dataset review is incomplete; resolve annotations before training.')
    import torch, ultralytics
    from ultralytics import YOLO
    import ultralytics.utils.torch_utils as tu
    tu.NUM_THREADS=cfg['threads'];torch.set_num_threads(cfg['threads'])
    weights=WORKSPACE/'weights'/cfg['model']
    assert weights.is_file(),f'Missing pretrained model: {weights}'
    # Revalidate every frozen experiment's source data before training any run.
    shared_val=shared_test=None
    for exp in cfg['experiments']:
        directory=Path(exp['data']).parent
        manifest=json.loads((directory/'manifest.json').read_text())
        assert len(manifest['train'])==exp['train_images']
        val=(directory/'val.txt').read_text();test=(directory/'test.txt').read_text()
        if shared_val is None:shared_val,shared_test=val,test
        assert (val,test)==(shared_val,shared_test),'Holdouts differ between experiments'
        train_paths=set((directory/'train.txt').read_text().splitlines())
        assert train_paths=={r['path'] for r in manifest['train']}
        assert not train_paths.intersection(val.splitlines()+test.splitlines())
        for row in manifest['train']:
            p=Path(row['path']);lp=p.parent.parent/'labels'/f'{p.stem}.txt'
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
            assert hashlib.sha256(lp.read_bytes()).hexdigest()==row['label_sha256']
    # Includes fixed holdouts and synthetic samples not selected for training.
    inventory=json.loads((ROOT/'results'/cfg.get('audit_directory','comparison_preparation')/'inventory.json').read_text())
    for row in inventory:
        p=Path(row['path']);lp=p.parent.parent/'labels'/f'{p.stem}.txt'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        assert hashlib.sha256(lp.read_bytes()).hexdigest()==row['label_sha256']
    YOLO(str(weights))
    if args.check_only:
        print('PASS: model loads; all frozen images/labels match; train/holdout paths disjoint; shared holdouts identical.')
        return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    status_path=out/'queue_status.json'
    previous=None
    if status_path.exists():
        if not args.resume:raise RuntimeError('Queue already exists. Use --resume to continue it.')
        previous=json.loads(status_path.read_text())
        if previous['config']!=cfg:raise RuntimeError('Configuration changed; cannot resume this queue.')
        import psutil
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid==os.getpid() or process.pid==os.getppid():continue
            if 'python' in (process.info['name'] or '').lower() and any(Path(arg).name=='run_comparison.py' for arg in (process.info['cmdline'] or [])):
                raise RuntimeError(f'Training process {process.pid} is already running.')
    state={'status':'starting','pid':os.getpid(),'started':time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed':[],'pending':[e['name'] for e in cfg['experiments']],
        'annotation_status':cfg['annotation_status'],'config':cfg,'environment':{
            'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
            'cuda_available':torch.cuda.is_available(),'threads':cfg['threads'],
            'pretrained_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}}
    if previous:
        state.update(started=previous['started'],resumed=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(status_path,state);scores=[]
    try:
        for exp in cfg['experiments']:
            name=exp['name']
            metrics_path=out/name/'test_metrics.json'
            if metrics_path.exists():
                result=json.loads(metrics_path.read_text())
                if not (out/name/'weights/best.pt').exists():raise RuntimeError(f'Missing completed checkpoint: {name}')
                scores.append({'experiment':name,'train_images':exp['train_images'],**result['metrics']})
                state['completed'].append(name);state['pending'].remove(name);save(status_path,state);continue
            last=out/name/'weights/last.pt'
            resume_epoch=0;finished=False
            if last.exists():
                if not args.resume:raise RuntimeError(f'Existing checkpoint: {last}')
                checkpoint=torch.load(last,map_location='cpu',weights_only=False)
                resume_epoch=checkpoint['epoch']+1
                finished=resume_epoch>=cfg['epochs'] or checkpoint['epoch']==-1
                if not finished and checkpoint.get('optimizer') is None:raise RuntimeError('Optimizer state missing.')
                del checkpoint
            elif (out/name).exists():raise RuntimeError(f'Run exists without last.pt: {name}')
            state.update(status='training',current=name,epoch=resume_epoch,total_epochs=cfg['epochs'])
            save(status_path,state);print(f'\nSTART {name}',flush=True)
            model=YOLO(str(last if last.exists() else weights))
            if last.exists():print(f'RESUME {name}: {resume_epoch} completed epochs',flush=True)
            def epoch_end(trainer):
                state.update(epoch=min(trainer.epoch+1,cfg['epochs']),updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                    validation_metrics={k:float(v) for k,v in trainer.metrics.items()})
                save(status_path,state)
            model.add_callback('on_fit_epoch_end',epoch_end)
            if last.exists():
                if not finished:model.train(resume=True)
            else:
                model.train(data=exp['data'],epochs=cfg['epochs'],imgsz=cfg['imgsz'],batch=cfg['batch'],
                    device=cfg['device'],workers=0,optimizer=cfg['optimizer'],lr0=cfg['lr0'],
                    patience=cfg['patience'],seed=cfg['seed'],deterministic=True,amp=False,cache=False,
                    project=str(out),name=name,exist_ok=False,plots=True,save=True,save_period=10,verbose=False)
            best=out/name/'weights/best.pt'
            state.update(status='evaluating',best_checkpoint=str(best));save(status_path,state)
            metrics=YOLO(str(best)).val(data=exp['data'],split='test',imgsz=cfg['imgsz'],batch=cfg['batch'],
                device=cfg['device'],workers=0,project=str(out),name=name+'_test',plots=True,verbose=False)
            result={'experiment':name,'train_images':exp['train_images'],'best_checkpoint':str(best),
                'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
                'per_class_ap50_95':{model.names[int(c)]:float(ap) for c,ap in zip(metrics.box.ap_class_index,metrics.box.ap)},
                'annotation_status':cfg['annotation_status']}
            save(out/name/'test_metrics.json',result)
            scores.append({'experiment':name,'train_images':exp['train_images'],**result['metrics']})
            with (out/'comparison_metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
            state['completed'].append(name);state['pending'].remove(name);save(status_path,state)
            if cfg.get('auto_export'):
                exported=subprocess.run([sys.executable,str(WORKSPACE/'scripts/export_deployment.py')],cwd=WORKSPACE,capture_output=True,text=True)
                state['deployment_export']={'returncode':exported.returncode,'message':exported.stdout+exported.stderr};save(status_path,state)
        state.update(status='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S'));save(status_path,state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        save(status_path,state);raise

if __name__=='__main__':main()
