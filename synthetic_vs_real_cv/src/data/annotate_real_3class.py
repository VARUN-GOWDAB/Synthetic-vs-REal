"""Create three-class draft annotations without overwriting the SHWD source."""
from pathlib import Path
import argparse
import os
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT.parent / 'Ultralytics'
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(CONFIG_DIR))
NAMES = ['worker', 'helmet', 'vest']


def helmet_labels(path):
    result = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'Invalid source annotation: {path}')
        if fields[0] == '1':
            result.append(line)
        elif fields[0] != '0':
            raise ValueError(f'Unexpected source class: {path}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/real')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/real_3class_draft')
    parser.add_argument('--limit', type=int, default=0, help='0 processes every image')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--model', default=str(ROOT.parent / 'weights' / 'yolov8s-worldv2.pt'))
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted draft with the same model settings')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error('Use a new output folder, or --resume for an interrupted draft.')
    if args.resume:
        previous = json.loads((args.output / 'annotation_manifest.json').read_text())
        expected = {'source': str(args.source.resolve()), 'model': args.model,
                    'confidence': args.confidence, 'imgsz': args.imgsz,
                    'prompts': ['person', 'safety vest', 'reflective vest', 'high visibility vest']}
        if any(previous.get(k) != v for k,v in expected.items()):
            parser.error('Resume settings differ from the existing annotation manifest.')
        if previous.get('complete'):
            print('Draft is already complete.'); return
    images = sorted(p for p in (args.source / 'images').iterdir()
                    if p.suffix.lower() in {'.jpg', '.png', '.jpeg'})
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error('No source images found.')
    for image in images:
        if not (args.source / 'labels' / (image.stem + '.txt')).exists():
            parser.error(f'Missing source label: {image.name}')
    from ultralytics import YOLOWorld
    model = YOLOWorld(args.model)
    model.set_classes(['person', 'safety vest', 'reflective vest', 'high visibility vest'])
    for folder in ['images', 'labels', 'previews']:
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    provenance = {'names': NAMES, 'source': str(args.source.resolve()),
                  'model': args.model, 'confidence': args.confidence, 'imgsz': args.imgsz,
                  'prompts': ['person', 'safety vest', 'reflective vest', 'high visibility vest'],
                  'status': 'unreviewed_pseudo_labels', 'complete': False,
                  'helmet_source': 'existing SHWD class 1 annotations',
                  'worker_and_vest_source': 'YOLO-World predictions; requires review',
                  'discarded': 'SHWD class 0 unhelmeted head boxes', 'images': []}
    manifest = args.output / 'annotation_manifest.json'
    manifest.write_text(json.dumps(provenance, indent=2))
    counts = {name: 0 for name in NAMES}
    for number, image in enumerate(images, 1):
        existing_label = args.output / 'labels' / (image.stem + '.txt')
        if args.resume and existing_label.exists() and (args.output / 'images' / image.name).exists():
            existing_lines = existing_label.read_text().splitlines()
            for line in existing_lines:
                parts = line.split()
                if len(parts) != 5 or parts[0] not in ('0','1','2'):
                    raise ValueError(f'Invalid existing draft label: {existing_label}')
                counts[NAMES[int(parts[0])]] += 1
            provenance['images'].append(image.name)
            continue
        lines = helmet_labels(args.source / 'labels' / (image.stem + '.txt'))
        result = model.predict(str(image), conf=args.confidence, device=args.device,
                               imgsz=args.imgsz, verbose=False)[0]
        # Merge overlapping vest predictions from synonymous prompts.
        import torch
        from torchvision.ops import nms
        mapped_classes = torch.where(result.boxes.cls == 0, 0, 2)
        keep = []
        for class_id in (0, 2):
            indices = torch.where(mapped_classes == class_id)[0]
            selected = nms(result.boxes.xyxy[indices], result.boxes.conf[indices], .5)
            keep.extend(indices[selected].tolist())
        boxes = result.boxes[keep]
        for cls, box in zip(boxes.cls.tolist(), boxes.xywhn.tolist()):
            mapped = 0 if int(cls) == 0 else 2
            lines.append(str(mapped) + ' ' + ' '.join(f'{v:.6f}' for v in box))
        shutil.copy2(image, args.output / 'images' / image.name)
        (args.output / 'labels' / (image.stem + '.txt')).write_text(
            '\n'.join(lines) + ('\n' if lines else ''))
        for line in lines:
            counts[NAMES[int(line.split()[0])]] += 1
        provenance['images'].append(image.name)
        if number <= 24:
            # Render final merged labels, including retained helmets.
            from PIL import Image, ImageDraw
            with Image.open(image).convert('RGB') as preview:
                draw = ImageDraw.Draw(preview)
                width, height = preview.size
                for line in lines:
                    cls, x, y, w, h = map(float, line.split())
                    bounds = ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)
                    color = ['cyan', 'yellow', 'lime'][int(cls)]
                    draw.rectangle(bounds, outline=color, width=2)
                    draw.text((bounds[0], bounds[1]), NAMES[int(cls)], fill=color)
                preview.thumbnail((1200, 1200))
                preview.save(args.output / 'previews' / (image.stem + '.jpg'))
        if number % 100 == 0:
            manifest.write_text(json.dumps(provenance, indent=2))
            print(f'Annotated {number}/{len(images)}: {counts}', flush=True)
    provenance.update(complete=True, counts=counts)
    manifest.write_text(json.dumps(provenance, indent=2))
    (args.output / 'classes.txt').write_text('\n'.join(NAMES) + '\n')
    print(json.dumps({'output': str(args.output), 'images': len(images), 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
