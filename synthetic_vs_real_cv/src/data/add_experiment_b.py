"""Add the requested 50% AI / 25% Blender / 25% real experiment to frozen splits."""
from pathlib import Path
from collections import Counter
import json
ROOT=Path(__file__).resolve().parents[2]
def main():
    base=ROOT/'data/processed/comparison_v1'
    name='mixed_ai50_rendered25_real25_400';out=base/name
    assert not out.exists(), 'B is already prepared; keep its frozen manifest.'
    read=lambda key:json.loads((base/key/'manifest.json').read_text())['train']
    train=read('ai_only_400')[:200]+read('rendered_only_400')[:100]+read('real_only_400')[:100]
    counts=Counter()
    for r in train:counts.update(r['objects'])
    assert all(counts[str(c)]>0 for c in range(3))
    out.mkdir()
    (out/'train.txt').write_text('\n'.join(r['path'] for r in train)+'\n')
    for split in ['val','test']:(out/f'{split}.txt').write_text((base/'real_only_400'/f'{split}.txt').read_text())
    (out/'data.yaml').write_text('path: '+json.dumps(out.resolve().as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
    e={'name':name,'train_images':400,'sources':dict(Counter(r['source'] for r in train)),
       'val_images':90,'test_images':90,'data':(out/'data.yaml').resolve().as_posix()}
    (out/'manifest.json').write_text(json.dumps({**e,'train':train},indent=2))
    cfg=json.loads((ROOT/'configs/comparison_v1.json').read_text());cfg['experiments']=[e]
    (ROOT/'configs/comparison_b.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps(e,indent=2))
if __name__=='__main__':main()
