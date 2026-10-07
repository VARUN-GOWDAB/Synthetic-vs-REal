"""Audit draft labels and prioritize likely incomplete automatic annotations."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'data/real')
    p.add_argument('--draft', type=Path, default=ROOT/'data/real_3class_draft')
    p.add_argument('--output', type=Path, default=ROOT/'results/real_3class_review')
    a = p.parse_args()
    manifest = json.loads((a.draft/'annotation_manifest.json').read_text())
    if not manifest['complete']:
        p.error('Wait for annotation generation to complete.')
    total = Counter(); rows = []; invalid = []; helmet_changes = []
    for image in manifest['images']:
        stem = Path(image).stem
        original = (a.source/'labels'/(stem+'.txt')).read_text().splitlines()
        draft = (a.draft/'labels'/(stem+'.txt')).read_text().splitlines()
        counts = Counter()
        for line in draft:
            try:
                fields=line.split(); cls=int(fields[0]); x,y,w,h=map(float,fields[1:])
                assert len(fields)==5 and cls in (0,1,2)
                assert all(math.isfinite(v) and 0<=v<=1 for v in (x,y,w,h)) and w>0 and h>0
                assert x-w/2>=-1e-5 and x+w/2<=1+1e-5 and y-h/2>=-1e-5 and y+h/2<=1+1e-5
                counts[cls]+=1
            except (ValueError, AssertionError, IndexError):
                invalid.append({'image':image,'line':line})
        orig_helmets=[line for line in original if line.split()[0]=='1']
        new_helmets=[line for line in draft if line.split()[0]=='1']
        if orig_helmets != new_helmets:
            helmet_changes.append(image)
        total.update(counts)
        source_heads=len(original)
        gap=max(0,source_heads-counts[0])
        rows.append({'image':image,'worker_boxes':counts[0],'helmet_boxes':counts[1],
                     'vest_boxes':counts[2],'source_head_or_helmet_boxes':source_heads,
                     'review_gap_heuristic':gap,'no_worker_prediction':int(counts[0]==0)})
    rows.sort(key=lambda r:(r['no_worker_prediction'],r['review_gap_heuristic']),reverse=True)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'review_priority.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={'images':len(rows),'objects':{'worker':total[0],'helmet':total[1],'vest':total[2]},
             'images_without_worker_predictions':sum(r['no_worker_prediction'] for r in rows),
             'images_with_vest_predictions':sum(r['vest_boxes']>0 for r in rows),
             'invalid_labels':invalid,'modified_helmet_annotations':helmet_changes,
             'status':'unreviewed_pseudo_labels',
             'review_note':'The gap is only a review priority heuristic. Source boxes can include isolated helmets. Review missed people and vests, false positives, and box extents.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    # A mixture of common failure cases and vest detections for visual review.
    candidates=rows[:4]+sorted(rows,key=lambda r:r['review_gap_heuristic'],reverse=True)[:4]+[r for r in rows if r['vest_boxes']][:4]
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1200,1000),'#202020')
    for i,row in enumerate(candidates):
        x=(i%3)*400; y=(i//3)*250
        with Image.open(a.draft/'images'/row['image']).convert('RGB') as im:
            width,height=im.size; draw=ImageDraw.Draw(im)
            for line in (a.draft/'labels'/(Path(row['image']).stem+'.txt')).read_text().splitlines():
                cls,cx,cy,w,h=map(float,line.split()); color=['cyan','yellow','lime'][int(cls)]
                box=((cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height)
                draw.rectangle(box,outline=color,width=2)
            im.thumbnail((390,215)); canvas.paste(im,(x,y+30))
        ImageDraw.Draw(canvas).text((x+4,y+4),f"{row['image']} | person {row['worker_boxes']} helmet {row['helmet_boxes']} vest {row['vest_boxes']}",fill='white')
    canvas.save(a.output/'review_contact_sheet.jpg')


if __name__=='__main__':
    main()
