"""Wait for the existing comparison queue, then run experiment B without overlap."""
from pathlib import Path
import json, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[2]
status=ROOT/'results/runs/comparison_v1/queue_status.json'
own=ROOT/'results/comparison_preparation/experiment_b_queue.json'
def record(state, **details):
    own.write_text(json.dumps({'status':state,**details},indent=2))
record('waiting',depends_on=str(status))
while True:
    state=json.loads(status.read_text())
    if state['status']=='complete':break
    if state['status']=='failed':
        record('blocked',reason='Original queue failed. Resolve its failure before running B.')
        sys.exit(1)
    time.sleep(30)
record('training')
run=subprocess.run([sys.executable,str(ROOT/'src/training/run_comparison.py'),
    '--config',str(ROOT/'configs/comparison_b.json'),'--output',str(ROOT/'results/runs/comparison_b')],cwd=ROOT.parent)
record('complete' if run.returncode==0 else 'failed',returncode=run.returncode)
sys.exit(run.returncode)
