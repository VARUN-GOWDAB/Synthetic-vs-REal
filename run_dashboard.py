"""Cross-platform inference launcher. First run: python run_dashboard.py --setup"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,venv
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup',action='store_true',help='Create a dedicated environment and install inference dependencies.')
    p.add_argument('--check',action='store_true',help='Verify bundled checkpoints and run a small inference check, then exit.')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--current-env',action='store_true',help='Use an already configured Python environment.')
    a=p.parse_args()
    if sys.version_info<(3,10):p.error('Python 3.10 or newer is required; Python 3.11 or 3.12 is recommended.')
    environment=ROOT/'.venv-inference'
    python=Path(sys.executable) if a.current_env else environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if a.setup:
        if not a.current_env and not python.exists():venv.create(environment,with_pip=True)
        subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements-inference.txt')],check=True)
    if not python.exists():p.error('First run: python run_dashboard.py --setup')
    manifest=ROOT/'deployment/registry.json'
    if not manifest.exists():p.error('Missing deployment/registry.json. Include deployment/ when pushing the project.')
    models=json.loads(manifest.read_text(encoding='utf-8'))['models']
    if a.check and not models:p.error('No replacement models bundled yet. Complete training and run scripts/export_deployment.py first.')
    for model in models:
        path=(ROOT/'deployment'/model['checkpoint']).resolve()
        if not path.is_relative_to((ROOT/'deployment').resolve()) or not path.is_file():p.error('A bundled model file is missing.')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=model['sha256']:p.error(f"Checkpoint checksum failed: {model['id']}")
    (ROOT/'Ultralytics').mkdir(exist_ok=True)
    env={**os.environ,'SYNTHREAL_DEPLOYMENT':'1','PYTHONUTF8':'1','YOLO_AUTOINSTALL':'false',
        'YOLO_CONFIG_DIR':str(ROOT/'Ultralytics'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    if a.check:
        code="from src.dashboard.registry import get_registry; from src.dashboard.server import Engine; from PIL import Image; r=get_registry(); e=Engine(); [(e.load(m), e.model.predict(Image.new('RGB',(64,64)),imgsz=64,device='cpu',verbose=False)) for m in r['models']]; print('PASS: bundled models load and infer without datasets or training outputs.')"
        command=[str(python),'-c',code]
    else:
        print(f'Inference only: http://127.0.0.1:{a.port} (Ctrl+C to stop)',flush=True)
        command=[str(python),'-m','src.dashboard.server','--port',str(a.port)]
    try:return subprocess.call(command,cwd=ROOT/'synthetic_vs_real_cv',env=env)
    except KeyboardInterrupt:return 0

if __name__=='__main__':sys.exit(main())
