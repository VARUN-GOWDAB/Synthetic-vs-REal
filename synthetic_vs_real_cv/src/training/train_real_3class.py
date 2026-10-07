"""Split a reviewed three-class dataset, train YOLO, and evaluate best weights."""
import argparse
import os
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT.parent / 'Ultralytics'
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(CONFIG_DIR))


def prepare(source, destination):
    if destination.exists():
        raise ValueError('Split output already exists. Reuse it with --data or choose a new --output.')
    groups = defaultdict(list)
    counts = Counter()
    images = sorted(p for p in (source / 'images').iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    annotation_manifest = source / 'annotation_manifest.json'
    if annotation_manifest.exists():
        selected = set(json.loads(annotation_manifest.read_text())['images'])
        images = [p for p in images if p.name in selected]
        if len(images) != len(selected):
            raise ValueError('Some selected images are missing from the annotation dataset.')
    if len(images) < 10:
        raise ValueError('Need at least 10 labeled images.')
    for image in images:
        label = source / 'labels' / (image.stem + '.txt')
        for line in label.read_text().splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'Invalid label: {label}')
            cls = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            if not (cls in (0, 1, 2) and all(math.isfinite(v) and 0 <= v <= 1 for v in (x,y,w,h))
                    and w > 0 and h > 0 and x-w/2 >= -1e-5 and x+w/2 <= 1+1e-5
                    and y-h/2 >= -1e-5 and y+h/2 <= 1+1e-5):
                raise ValueError(f'Invalid box: {label}')
            counts[cls] += 1
        groups[hashlib.sha256(image.read_bytes()).hexdigest()].append(image.resolve())
    if any(counts[c] == 0 for c in (0,1,2)):
        raise ValueError(f'All three classes need actual boxes. Current object counts: {dict(counts)}')
    groups = list(groups.values())
    random.Random(42).shuffle(groups)
    n = len(groups)
    v = max(1, round(n*.15))
    t = max(1, round(n*.15))
    partitions = {'train': groups[v+t:], 'val': groups[:v], 'test': groups[v:v+t]}
    split_counts = {}
    split_classes = {}
    for split, members in partitions.items():
        tally = Counter(int(line.split()[0]) for group in members for image in group
                        for line in (source/'labels'/(image.stem+'.txt')).read_text().splitlines())
        if any(tally[c] == 0 for c in (0,1,2)):
            raise ValueError(f'{split} lacks a class: {dict(tally)}. Curate representative splits before training.')
        split_classes[split] = dict(tally)
    destination.mkdir(parents=True)
    for split, members in partitions.items():
        paths = [p.as_posix() for group in members for p in group]
        split_counts[split] = len(paths)
        (destination / (split+'.txt')).write_text('\n'.join(paths)+'\n')
    data = destination/'data.yaml'
    data.write_text('path: '+json.dumps(destination.resolve().as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
    (destination/'split_manifest.json').write_text(json.dumps({
        'source': str(source.resolve()), 'seed': 42, 'counts': split_counts,
        'objects': dict(counts), 'objects_per_split': split_classes,
        'limitation': 'Exact duplicates grouped. Camera/scene and near-duplicate separation requires review.'}, indent=2))
    print(json.dumps(split_counts, indent=2))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'data/real_safety_500')
    p.add_argument('--output', type=Path, default=ROOT/'data/processed/real_safety_500')
    p.add_argument('--data', type=Path, help='Reuse prepared split data.yaml')
    p.add_argument('--annotations-reviewed', action='store_true')
    p.add_argument('--allow-pseudo-labels', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if not (a.prepare_only or a.annotations_reviewed or a.allow_pseudo_labels):
        p.error('Specify --annotations-reviewed after review or --allow-pseudo-labels for an explicitly unverified experiment.')
    manifest = a.source/'annotation_manifest.json'
    if manifest.exists() and not json.loads(manifest.read_text())['complete']:
        p.error('Annotation generation is incomplete.')
    existing_data = a.source/'data.yaml'
    data = a.data.resolve() if a.data else (existing_data.resolve() if existing_data.exists()
                                           else prepare(a.source.resolve(), a.output.resolve()))
    import yaml
    if list(yaml.safe_load(data.read_text())['names']) != ['worker','helmet','vest']:
        p.error('Dataset classes must be worker, helmet, vest in that order.')
    if a.prepare_only:
        return
    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch, workers=0,
                device=a.device, optimizer='AdamW', lr0=.01, patience=20, seed=42,
                project=str(ROOT/'results/runs'), name='real_worker_helmet_vest', exist_ok=False)
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(data), split='test', workers=0, device=a.device,
                                project=str(ROOT/'results/runs'), name='real_3class_test')
    (best.parent.parent/'test_metrics.json').write_text(json.dumps({
        'metrics': metrics.results_dict, 'annotation_claim':
        'user_confirmed_reviewed' if a.annotations_reviewed else 'unverified_pseudo_labels',
        'best_checkpoint': str(best)}, indent=2))
    print(f'Best checkpoint: {best}')


if __name__ == '__main__':
    main()
