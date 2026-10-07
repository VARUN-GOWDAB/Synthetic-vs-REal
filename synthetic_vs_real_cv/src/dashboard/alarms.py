"""Conservative temporal PPE association; alerts are possible violations, not proof."""

def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=w*h
    return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

def associate(detections,width,height,required=('helmet','vest')):
    people=[{'box':d['box'],'confidence':d['confidence'],'present':set()} for d in detections if d['class_id']==0]
    for equipment in [d for d in detections if d['class_id'] in (1,2)]:
        box=equipment['box'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2;candidates=[]
        for i,person in enumerate(people):
            x1,y1,x2,y2=person['box'];w=x2-x1;h=y2-y1
            low,high=(-.08,.35) if equipment['class_id']==1 else (.12,.78)
            if x1<=cx<=x2 and y1+low*h<=cy<=y1+high*h:
                candidates.append((abs(cx-(x1+x2)/2)/max(w,1)+abs((cy-y1)/max(h,1)-(.08 if equipment['class_id']==1 else .4)),i))
        if candidates:people[min(candidates)[1]]['present'].add('helmet' if equipment['class_id']==1 else 'vest')
    for p in people:
        x1,y1,x2,y2=p['box']
        p['uncertain']=y2-y1<80 or x1<=2 or y1<=2 or x2>=width-2 or y2>=height-2
        p['missing']=[x for x in required if x not in p['present']]
        p['present']=sorted(p['present'])
    return people

class PPETracker:
    def __init__(self):self.tracks={};self.next_id=1
    def update(self,people,now,persistence=2.0,min_frames=3):
        self.tracks={k:v for k,v in self.tracks.items() if now-v['last']<5}
        remaining=set(self.tracks);output=[]
        for p in people:
            matches=[(iou(p['box'],self.tracks[k]['box']),k) for k in remaining]
            best=max(matches,default=(0,None));key=best[1] if best[0]>=.3 else self.next_id
            if best[0]<.3:self.next_id+=1
            else:remaining.remove(key)
            old=self.tracks.get(key,{});missing=tuple(p['missing']) if not p['uncertain'] else ()
            same=bool(missing) and old.get('missing')==missing and now-old.get('last',now)<5
            start=old['start'] if same else now;frames=old.get('frames',0)+1 if same else 1
            self.tracks[key]={'box':p['box'],'missing':missing,'start':start,'last':now,'frames':frames}
            alert=bool(missing) and frames>=min_frames and now-start>=persistence
            output.append({**p,'track_id':key,'alert':alert,'observations':frames,'duration':round(now-start,2),
                'assessment':'uncertain' if p['uncertain'] else ('possible missing PPE' if alert else ('observing' if missing else 'PPE detected'))})
        # A missed person observation breaks the consecutive-frame requirement.
        for key in remaining:
            self.tracks[key]['missing']=();self.tracks[key]['frames']=0
        return output
