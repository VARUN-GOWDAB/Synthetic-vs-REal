"""Apply staged, reviewed corrections; preserve originals and verify image hashes."""
from pathlib import Path
import json,hashlib,math,zipfile
from collections import Counter,defaultdict
root=Path('synthetic_vs_real_cv');base=root/'results/synthetic_label_correction'
d=base/'ai_generated';dec=json.loads((d/'vest_visual_decisions.json').read_text(encoding='utf-8-sig'));rows=json.loads((d/'vest_review_order.json').read_text());vests=defaultdict(list)
assert dec['reviewed_through']==len(rows)
for i,r in enumerate(rows,1):
 if (r['keep'] or i in dec['restore']) and i not in dec['remove'] and r['image']!='img_0566.jpg':vests[r['image']].append(r['line'])
helmet_rows=json.loads((d/'helmet_review_order.json').read_text());helmet_dec=json.loads((d/'helmet_visual_decisions.json').read_text());removed={(helmet_rows[i-1]['image'],helmet_rows[i-1]['line']) for i in helmet_dec['remove']}
for p in (d/'labels').glob('*.txt'):
 lines=[l for l in p.read_text().splitlines() if l.strip() and l.split()[0]!='2' and (p.stem+'.jpg',l) not in removed]+vests[p.stem+'.jpg']
 p.write_text('\n'.join(lines)+'\n')
dups=json.loads((base/'duplicate_groups.json').read_text());backup=zipfile.ZipFile(base/'original_labels.zip');report={};pending=[]
for name in ['ai_generated','3d_rendered']:
 source=root/'data'/name;stage=base/name/'labels';before=Counter();after=Counter();changed=0;empty=[];records=[]
 for canonical,copies in dups[name]['duplicates'].items():
  for copy in copies:(stage/(Path(copy).stem+'.txt')).write_text((stage/(Path(canonical).stem+'.txt')).read_text())
 hashes={r['image']:r['sha256'] for r in json.loads((base/name/'records.json').read_text())}
 for image in sorted((source/'images').glob('*')):
  assert hashlib.sha256(image.read_bytes()).hexdigest()==hashes[image.name],image
  p=stage/(image.stem+'.txt');new=p.read_text();old=backup.read(f'{name}/labels/{p.name}').decode();changed+=old.strip()!=new.strip()
  before.update(int(l.split()[0]) for l in old.splitlines() if l.strip());counts=Counter()
  for l in new.splitlines():
   if not l.strip():continue
   fields=l.split();assert len(fields)==5,(p,l)
   c=int(fields[0]);x,y,w,h=map(float,fields[1:]);assert c in (0,1,2) and all(math.isfinite(v) for v in (x,y,w,h)) and w>0 and h>0 and min(x-w/2,y-h/2)>=-1e-5 and max(x+w/2,y+h/2)<=1.00001,(p,l)
   counts[c]+=1
  after.update(counts)
  if not counts:empty.append(image.name)
  records.append({'image':image.name,'sha256':hashes[image.name],'objects':dict(counts),'label_sha256':hashlib.sha256(new.encode()).hexdigest()});pending.append((source/'labels'/p.name,new))
 assert not empty,(name,empty)
 report[name]={'images':len(records),'training_unique_images':len(dups[name]['unique_images']),'changed_label_files':changed,'before_objects':dict(before),'after_objects':dict(after),'empty_labels':empty}
 (base/name/'final_records.json').write_text(json.dumps(records,indent=2))
# Only write active labels after every source passes validation.
for p,content in pending:p.write_text(content)
for name in report:
 source=root/'data'/name
 manifest={'complete':True,'classes':['worker','helmet','vest'],'images':dups[name]['unique_images'],'status':'model_assisted_corrected_with_targeted_visual_review','review_scope':'All 1674 AI vest candidate crops reviewed; flagged helmet crops reviewed; 3D worker boxes regenerated with sampled visual checks. Not exhaustive manual ground truth.','duplicates_excluded':dups[name]['duplicates'],'audit':'../../results/synthetic_label_correction'}
 (source/'annotation_manifest.json').write_text(json.dumps(manifest,indent=2));(source/'classes.txt').write_text('worker\nhelmet\nvest\n')
(base/'final_audit.json').write_text(json.dumps(report,indent=2))
lines=['# Synthetic annotation correction','', 'Classes: 0 worker/person, 1 hard hat/safety helmet, 2 safety/high-visibility vest.','', 'Original annotations: original_labels.zip. All source images preserved; SHA-256 hashes checked. Exact duplicate images are excluded by the source annotation manifests and their labels synchronized.','', 'AI person boxes retain original annotations with model additions. Vest candidates were classified and all 1,674 crops visually screened; ordinary clothing, harnesses and protective suits were removed. Flagged helmet crops were screened for welding masks and hoods. 3D worker boxes were regenerated; original equipment annotations were retained/merged.','', 'These are model-assisted corrections with targeted visual review, not exhaustive manually verified ground truth. Sampled whole-image checks may leave missed or imperfect boxes. Training has not been started.','']
for name,r in report.items():lines += [f'## {name}',json.dumps(r,indent=2),'']
(base/'README.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2))
