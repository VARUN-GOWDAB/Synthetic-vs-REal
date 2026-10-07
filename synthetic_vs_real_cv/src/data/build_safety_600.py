"""Materialize the visually screened selection; never deletes source data."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/real_3class_draft'
OUT = ROOT / 'data/real_safety_600'
AUDIT = ROOT / 'results/curation_600'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    decisions = json.loads((AUDIT/'visual_decisions.json').read_text(encoding='utf-8-sig'))
    ranked = json.loads((AUDIT/'ranked_candidates.json').read_text())
    indexes = sorted(decisions['keep'] + decisions['remove_false_vest'])
    assert len(indexes) == len(set(indexes)) == 600
    assert not OUT.exists(), 'Refusing to overwrite an existing curated dataset.'
    (OUT/'images').mkdir(parents=True)
    (OUT/'labels').mkdir()
    counts = Counter()
    records = []
    for index in indexes:
        row = ranked[index-1]
        name = row['image']
        assert Path(name).name == name and not name.startswith(('PartA_', 'PartB_'))
        src_image = SOURCE/'images'/name
        label_name = Path(name).stem + '.txt'
        lines = (SOURCE/'labels'/label_name).read_text().splitlines()
        removed = []
        if index in decisions['remove_false_vest']:
            removed = [line for line in lines if line.split()[0] == '2']
            lines = [line for line in lines if line.split()[0] != '2']
        tally = Counter()
        for line in lines:
            fields = line.split()
            assert len(fields) == 5
            cls = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            assert cls in (0, 1, 2)
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in (x,y,w,h))
            assert w > 0 and h > 0
            assert x-w/2 >= -1e-5 and x+w/2 <= 1+1e-5
            assert y-h/2 >= -1e-5 and y+h/2 <= 1+1e-5
            tally[cls] += 1
        assert tally[0] and tally[1]
        shutil.copy2(src_image, OUT/'images'/name)
        (OUT/'labels'/label_name).write_text('\n'.join(lines)+'\n')
        assert digest(src_image) == digest(OUT/'images'/name) == row['sha256']
        with Image.open(OUT/'images'/name) as im:
            im.verify()
        counts.update(tally)
        records.append(dict(row, rank=index, objects=dict(tally), removed_false_vest_boxes=removed,
                            label_sha256=digest(OUT/'labels'/label_name)))
    assert len({r['sha256'] for r in records}) == 600
    assert len(list((OUT/'images').iterdir())) == len(list((OUT/'labels').iterdir())) == 600
    # Stratify by vest presence so the minority class appears in every split.
    rng = random.Random(42)
    strata = [[r for r in records if bool(r['objects'].get(2,0)) == vest] for vest in (True,False)]
    partitions = {k: [] for k in ('train','val','test')}
    for group in strata:
        rng.shuffle(group)
        nval = round(len(group)*.15)
        ntest = nval
        partitions['val'].extend(group[:nval])
        partitions['test'].extend(group[nval:nval+ntest])
        partitions['train'].extend(group[nval+ntest:])
    split_summary = {}
    for split, rows in partitions.items():
        rng.shuffle(rows)
        tally = Counter()
        for row in rows:
            tally.update(row['objects'])
            row['split'] = split
        assert all(tally[c] for c in range(3))
        assert len(rows) == {'train':420,'val':90,'test':90}[split]
        (OUT/f'{split}.txt').write_text(''.join('./images/'+r['image']+'\n' for r in rows))
        split_summary[split] = dict(images=len(rows), objects=dict(tally),
                                   vest_images=sum(bool(r['objects'].get(2,0)) for r in rows))
    (OUT/'data.yaml').write_text('path: '+json.dumps(OUT.resolve().as_posix())+
        '\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n')
    (OUT/'classes.txt').write_text('worker\nhelmet\nvest\n')
    provenance = json.loads((SOURCE/'annotation_manifest.json').read_text())
    shutil.copy2(SOURCE/'annotation_manifest.json', AUDIT/'source_annotation_manifest.json')
    for name in ('README.md','voc_conversion_report.json'):
        shutil.copy2(ROOT/'data/real'/name, AUDIT/('original_'+name))
    summary = dict(names=['worker','helmet','vest'], complete=True, images=[r['image'] for r in records],
                   selected_images=600, objects=dict(counts), splits=split_summary,
                   status='visually_screened_pseudo_labels', review_method=decisions['method'],
                   source_model=provenance['model'], helmet_source=provenance['helmet_source'],
                   corrected_images=sum(bool(r['removed_false_vest_boxes']) for r in records),
                   removed_false_vest_boxes=sum(len(r['removed_false_vest_boxes']) for r in records),
                   vest_images=sum(bool(r['objects'].get(2,0)) for r in records),
                   limitations=['Contact-sheet screening is not exhaustive manual ground truth.',
                                'Helmet boxes retain source SHWD extents, which can include the face.',
                                'Exact duplicates removed; near duplicates filtered by dHash. Scene-level separation is not guaranteed.'])
    (OUT/'annotation_manifest.json').write_text(json.dumps(summary,indent=2))
    (OUT/'curation_records.json').write_text(json.dumps(records,indent=2))
    (AUDIT/'final_validation.json').write_text(json.dumps({k:v for k,v in summary.items() if k!='images'},indent=2))
    (OUT/'README.md').write_text(
        '# Curated real safety dataset\n\n600 images and 600 YOLO label files.\n\n'
        'Class IDs: 0 worker/person, 1 safety helmet/hard hat, 2 safety/high-visibility vest.\n\n'
        'Splits: 420 train, 90 validation, 90 test; seed 42, stratified by vest presence.\n\n'
        f"Objects: {counts[0]} workers, {counts[1]} helmets, {counts[2]} vests. "
        f"Vests occur in {summary['vest_images']} images; all three classes need not appear in every image.\n\n"
        f"Selected after screening 1,240 ranked candidates. Removed {summary['removed_false_vest_boxes']} "
        f"false vest boxes from {summary['corrected_images']} images. Classroom images were excluded.\n\n"
        'Labels originate from YOLO-World worker/vest predictions and SHWD helmet annotations. '
        'Visual contact-sheet screening rejected obvious missing objects and false detections, '
        'but this is not exhaustive manual ground truth. Source helmet boxes can include the face. '
        'Independently review test labels before reporting accuracy. Exact duplicates were removed; '
        'scene-level separation is not guaranteed.\n\n'
        'Use data.yaml for training. Relative image lists remain portable; update the YAML path if moving this folder. '
        'annotation_manifest.json and curation_records.json contain provenance, corrections and hashes.\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('images','limitations')},indent=2))


if __name__ == '__main__':
    main()
